#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace vllm {
namespace dispatch_combine {

// Maximum number of EP ranks supported.
constexpr int kMaxRanks = 64;

// Persistent grid size for fused kernels.
// 32 blocks fit on all modern GPUs (>=80 SMs), starting
// within nanoseconds of each other. This guarantees
// that inline barrier reads of the counter happen before
// block 0 increments it (barrier takes microseconds).
constexpr int kPersistentGrid = 32;

// ====================================================================
// P2P flag operations for cross-GPU synchronization.
// Follows custom_all_reduce.cuh pattern (lines 159-181).
// ====================================================================
using FlagType = uint32_t;

static __device__ __forceinline__ void dc_st_flag_release(
    FlagType* flag_addr, FlagType flag) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  asm volatile(
      "st.release.sys.global.u32 [%1], %0;"
      ::"r"(flag), "l"(flag_addr));
#else
  asm volatile(
      "membar.sys; st.volatile.global.u32 [%1], %0;"
      ::"r"(flag), "l"(flag_addr));
#endif
}

static __device__ __forceinline__ FlagType dc_ld_flag_acquire(
    FlagType* flag_addr) {
  FlagType flag;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  asm volatile(
      "ld.acquire.sys.global.u32 %0, [%1];"
      : "=r"(flag) : "l"(flag_addr));
#else
  asm volatile(
      "ld.volatile.global.u32 %0, [%1]; membar.gl;"
      : "=r"(flag) : "l"(flag_addr));
#endif
  return flag;
}

// Metadata for each dispatched token-expert pair.
// Packed into 16 bytes for efficient P2P transfer.
struct __align__(16) TokenMetadata {
  int32_t source_rank;       // Originating rank
  int32_t source_token_idx;  // Token index on originating rank
  int32_t expert_id;         // Global expert ID
  float topk_weight;         // Router weight for this pair
};

// P2P barrier signal buffer. One per rank, shared via IPC.
// Counter is GPU-resident and increments on each barrier
// call, naturally compatible with CUDA graph replays.
struct DispatchCombineSignals {
  // flags[i] is written by rank i to signal readiness.
  alignas(128) FlagType flags[kMaxRanks];
  // Monotonically increasing counter.
  FlagType counter;
};

// Per-rank buffer configuration passed to CUDA kernels.
struct DispatchCombineConfig {
  // Pointers to each rank's dispatch recv buffer (via IPC).
  void* remote_dispatch_recv[kMaxRanks];
  // Pointers to each rank's dispatch metadata buffer (via IPC).
  void* remote_dispatch_meta[kMaxRanks];
  // Pointers to each rank's dispatch write-offset counter (via IPC).
  int32_t* remote_dispatch_offsets[kMaxRanks];
  // Pointers to each rank's combine recv buffer (via IPC).
  void* remote_combine_recv[kMaxRanks];
  // Pointers to each rank's combine metadata buffer (via IPC).
  void* remote_combine_meta[kMaxRanks];
  // Pointers to each rank's combine write-offset counter (via IPC).
  int32_t* remote_combine_offsets[kMaxRanks];

  // P2P barrier signal buffers (via IPC).
  DispatchCombineSignals* self_signals;
  DispatchCombineSignals* peer_signals[kMaxRanks];

  int32_t rank;
  int32_t world_size;
  int32_t experts_per_rank;
  int32_t hidden_dim;
  int32_t max_num_tokens_per_rank;
  int32_t max_recv;  // max entries per recv buffer
};

// ====================================================================
// GPU-side buffer operations (CUDA-graph compatible)
// ====================================================================

// Reset local rank's dispatch and combine offset counters.
// Launch with 1 block, 1 thread.
__global__ void reset_offsets_kernel(
    const DispatchCombineConfig* __restrict__ config) {
  const int32_t rank = config->rank;
  if (config->remote_dispatch_offsets[rank]) {
    *config->remote_dispatch_offsets[rank] = 0;
  }
  if (config->remote_combine_offsets[rank]) {
    *config->remote_combine_offsets[rank] = 0;
  }
}

// Reset only the local rank's combine offset counter.
// Launch with 1 block, 1 thread.
__global__ void reset_combine_offset_kernel(
    const DispatchCombineConfig* __restrict__ config) {
  const int32_t rank = config->rank;
  if (config->remote_combine_offsets[rank]) {
    *config->remote_combine_offsets[rank] = 0;
  }
}

// ====================================================================
// P2P flag-based barrier kernels (replace NCCL AllReduce).
// Launch with 1 block, kMaxRanks threads.
// ====================================================================
enum class BarrierMode : int {
  PURE = 0,                    // Signal + wait only
  RESET_DISPATCH_COMBINE = 1,  // Reset both offsets + barrier
  RESET_COMBINE = 2,           // Reset combine offset + barrier
  RESET_DISPATCH = 3,          // Reset dispatch offset + barrier
};

template <BarrierMode mode>
__global__ void p2p_barrier_kernel(
    const DispatchCombineConfig* __restrict__ config) {
  const int32_t rank = config->rank;
  const int32_t ws = config->world_size;
  const int32_t tid = threadIdx.x;
  if (tid >= ws) return;

  // Optional reset (thread 0 only).
  if constexpr (mode == BarrierMode::RESET_DISPATCH_COMBINE) {
    if (tid == 0) {
      if (config->remote_dispatch_offsets[rank])
        *config->remote_dispatch_offsets[rank] = 0;
      if (config->remote_combine_offsets[rank])
        *config->remote_combine_offsets[rank] = 0;
    }
  } else if constexpr (mode == BarrierMode::RESET_COMBINE) {
    if (tid == 0) {
      if (config->remote_combine_offsets[rank])
        *config->remote_combine_offsets[rank] = 0;
    }
  } else if constexpr (mode == BarrierMode::RESET_DISPATCH) {
    if (tid == 0) {
      if (config->remote_dispatch_offsets[rank])
        *config->remote_dispatch_offsets[rank] = 0;
    }
  }

  // Make all preceding writes visible to peers.
  __threadfence_system();

  // Read counter and compute expected flag value.
  // Counter is GPU-resident; increments naturally on
  // each CUDA graph replay.
  FlagType flag = config->self_signals->counter + 1;

  // Write flag to peer tid's signal buffer at our rank.
  dc_st_flag_release(
      &config->peer_signals[tid]->flags[rank], flag);

  // Spin-wait on own signal buffer for peer tid's flag.
  while (dc_ld_flag_acquire(
      &config->self_signals->flags[tid]) != flag)
    ;

  __syncthreads();

  // Update counter (one thread only).
  if (tid == 0) {
    config->self_signals->counter = flag;
  }
}

// Copy dispatch recv data from IPC buffer to PyTorch tensor.
// Reads actual count from the local dispatch offset counter.
// Entries beyond actual count are zeroed.
// Launch with max_recv blocks, kBlockSize threads.
template <typename T>
__global__ void copy_dispatch_recv_kernel(
    T* __restrict__ output,  // (max_recv, K)
    const DispatchCombineConfig* __restrict__ config,
    int32_t K) {
  const int32_t entry_idx = blockIdx.x;
  const int32_t rank = config->rank;

  // Read actual count from local dispatch offset counter.
  int32_t actual_count =
      *config->remote_dispatch_offsets[rank];
  if (actual_count > config->max_recv)
    actual_count = config->max_recv;

  if (entry_idx >= actual_count) {
    // Zero entries beyond actual count.
    for (int32_t k = threadIdx.x; k < K; k += blockDim.x) {
      output[entry_idx * K + k] = T(0);
    }
    return;
  }

  const T* src = reinterpret_cast<const T*>(
      config->remote_dispatch_recv[rank]);
  for (int32_t k = threadIdx.x; k < K; k += blockDim.x) {
    output[entry_idx * K + k] = src[entry_idx * K + k];
  }
}

// Copy dispatch metadata from IPC buffer to PyTorch tensor.
// Entries beyond actual count are zeroed (weight=0).
// Launch with max_recv blocks, 1 thread per block.
__global__ void copy_dispatch_meta_kernel(
    int32_t* __restrict__ output,  // (max_recv, 4)
    const DispatchCombineConfig* __restrict__ config) {
  const int32_t entry_idx = blockIdx.x;
  const int32_t rank = config->rank;

  int32_t actual_count =
      *config->remote_dispatch_offsets[rank];
  if (actual_count > config->max_recv)
    actual_count = config->max_recv;

  if (entry_idx >= actual_count) {
    // Padding: use num_experts as expert_id sentinel
    // so moe_align_block_size skips these entries
    // (its kernel has: if expert_id >= num_experts continue).
    output[entry_idx * 4 + 0] = 0;
    output[entry_idx * 4 + 1] = 0;
    output[entry_idx * 4 + 2] =
        config->experts_per_rank * config->world_size;
    *reinterpret_cast<float*>(
        &output[entry_idx * 4 + 3]) = 0.0f;
    return;
  }

  const TokenMetadata* src =
      reinterpret_cast<const TokenMetadata*>(
          config->remote_dispatch_meta[rank]);
  const TokenMetadata& meta = src[entry_idx];
  output[entry_idx * 4 + 0] = meta.source_rank;
  output[entry_idx * 4 + 1] = meta.source_token_idx;
  output[entry_idx * 4 + 2] = meta.expert_id;
  *reinterpret_cast<float*>(
      &output[entry_idx * 4 + 3]) = meta.topk_weight;
}

// Copy combine recv data from IPC buffer to PyTorch tensor.
// Launch with max_recv blocks, kBlockSize threads.
template <typename T>
__global__ void copy_combine_recv_kernel(
    T* __restrict__ output,  // (max_recv, K)
    const DispatchCombineConfig* __restrict__ config,
    int32_t K) {
  const int32_t entry_idx = blockIdx.x;
  const int32_t rank = config->rank;

  int32_t actual_count =
      *config->remote_combine_offsets[rank];
  if (actual_count > config->max_recv)
    actual_count = config->max_recv;

  if (entry_idx >= actual_count) {
    for (int32_t k = threadIdx.x; k < K; k += blockDim.x) {
      output[entry_idx * K + k] = T(0);
    }
    return;
  }

  const T* src = reinterpret_cast<const T*>(
      config->remote_combine_recv[rank]);
  for (int32_t k = threadIdx.x; k < K; k += blockDim.x) {
    output[entry_idx * K + k] = src[entry_idx * K + k];
  }
}

// Copy combine metadata from IPC buffer to PyTorch tensor.
// Launch with max_recv blocks, 1 thread per block.
__global__ void copy_combine_meta_kernel(
    int32_t* __restrict__ output,  // (max_recv, 4)
    const DispatchCombineConfig* __restrict__ config) {
  const int32_t entry_idx = blockIdx.x;
  const int32_t rank = config->rank;

  int32_t actual_count =
      *config->remote_combine_offsets[rank];
  if (actual_count > config->max_recv)
    actual_count = config->max_recv;

  if (entry_idx >= actual_count) {
    output[entry_idx * 4 + 0] = 0;
    output[entry_idx * 4 + 1] = 0;
    output[entry_idx * 4 + 2] = 0;
    *reinterpret_cast<float*>(
        &output[entry_idx * 4 + 3]) = 0.0f;
    return;
  }

  const TokenMetadata* src =
      reinterpret_cast<const TokenMetadata*>(
          config->remote_combine_meta[rank]);
  const TokenMetadata& meta = src[entry_idx];
  output[entry_idx * 4 + 0] = meta.source_rank;
  output[entry_idx * 4 + 1] = meta.source_token_idx;
  output[entry_idx * 4 + 2] = meta.expert_id;
  *reinterpret_cast<float*>(
      &output[entry_idx * 4 + 3]) = meta.topk_weight;
}

// ====================================================================
// Dispatch P2P kernel
// ====================================================================
template <typename T>
__global__ void dispatch_p2p_kernel(
    const T* __restrict__ input,
    const int32_t* __restrict__ topk_ids,
    const float* __restrict__ topk_weights,
    const DispatchCombineConfig* __restrict__ config,
    int32_t M,
    int32_t K,
    int32_t topk) {
  const int32_t rank = config->rank;
  const int32_t experts_per_rank = config->experts_per_rank;

  const int32_t pair_idx = blockIdx.x;
  if (pair_idx >= M * topk) return;

  const int32_t token_idx = pair_idx / topk;
  const int32_t expert_slot = pair_idx % topk;

  const int32_t expert_id =
      topk_ids[token_idx * topk + expert_slot];
  const float weight =
      topk_weights[token_idx * topk + expert_slot];
  const int32_t dest_rank = expert_id / experts_per_rank;

  if (dest_rank < 0 || dest_rank >= config->world_size)
    return;

  if (!config->remote_dispatch_offsets[dest_rank] ||
      !config->remote_dispatch_recv[dest_rank] ||
      !config->remote_dispatch_meta[dest_rank]) {
    return;
  }

  __shared__ int32_t s_write_pos;
  if (threadIdx.x == 0) {
    s_write_pos = atomicAdd(
        config->remote_dispatch_offsets[dest_rank], 1);
  }
  __syncthreads();

  const int32_t write_pos = s_write_pos;
  if (write_pos >= config->max_recv) return;

  T* dest_data = reinterpret_cast<T*>(
      config->remote_dispatch_recv[dest_rank]);
  const T* src_data = input + token_idx * K;
  for (int32_t k = threadIdx.x; k < K; k += blockDim.x) {
    dest_data[write_pos * K + k] = src_data[k];
  }

  if (threadIdx.x == 0) {
    TokenMetadata* dest_meta =
        reinterpret_cast<TokenMetadata*>(
            config->remote_dispatch_meta[dest_rank]);
    dest_meta[write_pos].source_rank = rank;
    dest_meta[write_pos].source_token_idx = token_idx;
    dest_meta[write_pos].expert_id = expert_id;
    dest_meta[write_pos].topk_weight = weight;
  }

  __threadfence_system();
}

// ====================================================================
// Combine P2P kernel
// ====================================================================
// Persistent-grid combine: reads actual dispatch recv count
// from config offset counter. Grid = kPersistentGrid;
// kernel loops over entries for CUDA graph compatibility.
template <typename T>
__global__ void combine_p2p_kernel(
    const T* __restrict__ expert_output,
    const TokenMetadata* __restrict__ dispatch_meta,
    const DispatchCombineConfig* __restrict__ config,
    int32_t K) {
  int32_t M_recv =
      *config->remote_dispatch_offsets[config->rank];
  if (M_recv > config->max_recv)
    M_recv = config->max_recv;

  __shared__ int32_t s_write_pos;

  for (int32_t pair_idx = blockIdx.x;
       pair_idx < M_recv;
       pair_idx += gridDim.x) {
    const int32_t dest_rank =
        dispatch_meta[pair_idx].source_rank;
    const int32_t orig_token_idx =
        dispatch_meta[pair_idx].source_token_idx;
    const float weight =
        dispatch_meta[pair_idx].topk_weight;

    if (dest_rank < 0 ||
        dest_rank >= config->world_size)
      continue;

    if (!config->remote_combine_offsets[dest_rank] ||
        !config->remote_combine_recv[dest_rank] ||
        !config->remote_combine_meta[dest_rank])
      continue;

    if (threadIdx.x == 0) {
      s_write_pos = atomicAdd(
          config->remote_combine_offsets[dest_rank],
          1);
    }
    __syncthreads();

    const int32_t write_pos = s_write_pos;
    if (write_pos >= config->max_recv) continue;

    T* dest_data = reinterpret_cast<T*>(
        config->remote_combine_recv[dest_rank]);
    const T* src_data =
        expert_output + pair_idx * K;
    for (int32_t k = threadIdx.x; k < K;
         k += blockDim.x) {
      dest_data[write_pos * K + k] = src_data[k];
    }

    if (threadIdx.x == 0) {
      TokenMetadata* dest_meta =
          reinterpret_cast<TokenMetadata*>(
              config->remote_combine_meta[
                  dest_rank]);
      dest_meta[write_pos].source_rank =
          config->rank;
      dest_meta[write_pos].source_token_idx =
          orig_token_idx;
      dest_meta[write_pos].expert_id =
          dispatch_meta[pair_idx].expert_id;
      dest_meta[write_pos].topk_weight = weight;
    }

    __threadfence_system();
  }
}

// ====================================================================
// Scatter-add weighted kernel
// ====================================================================
// Skips entries with zero weight (padding beyond actual count).
template <typename T>
__global__ void scatter_add_weighted_kernel(
    float* __restrict__ output,
    const T* __restrict__ combine_recv,
    const TokenMetadata* __restrict__ combine_meta,
    int32_t N_recv,
    int32_t K) {
  const int32_t entry_idx = blockIdx.x;
  if (entry_idx >= N_recv) return;

  const int32_t token_idx =
      combine_meta[entry_idx].source_token_idx;
  const float weight =
      combine_meta[entry_idx].topk_weight;

  // Skip zero-weight entries (padding beyond actual count).
  if (weight == 0.0f) return;

  for (int32_t k = threadIdx.x; k < K; k += blockDim.x) {
    float val = static_cast<float>(
        combine_recv[entry_idx * K + k]);
    val *= weight;
    atomicAdd(output + token_idx * K + k, val);
  }
}

// ====================================================================
// Stamp + zero stale dispatch entries (replaces copy kernels)
// ====================================================================
// After post-dispatch barrier, stamps sentinel metadata and
// zeros stale data for entries beyond actual_count.
// Grid = mc (tight upper bound), block = kBlockSize.
template <typename T>
__global__ void stamp_and_zero_dispatch_kernel(
    T* __restrict__ dispatch_recv,  // IPC buffer, in-place
    const DispatchCombineConfig* __restrict__ config,
    int32_t mc,
    int32_t K) {
  const int32_t idx = blockIdx.x;
  if (idx >= mc) return;

  const int32_t rank = config->rank;
  int32_t actual =
      *config->remote_dispatch_offsets[rank];
  if (actual > config->max_recv)
    actual = config->max_recv;

  // Real entries: skip (written by dispatch_p2p).
  if (idx < actual) return;

  // Stale entries: zero data for quantization safety.
  T* dest = dispatch_recv + idx * K;
  for (int32_t k = threadIdx.x; k < K;
       k += blockDim.x) {
    dest[k] = T(0);
  }

  // Stamp sentinel metadata (expert_id = num_experts).
  if (threadIdx.x == 0) {
    TokenMetadata* meta =
        reinterpret_cast<TokenMetadata*>(
            config->remote_dispatch_meta[rank]);
    meta[idx].source_rank = 0;
    meta[idx].source_token_idx = 0;
    meta[idx].expert_id =
        config->experts_per_rank * config->world_size;
    meta[idx].topk_weight = 0.0f;
  }
}

// ====================================================================
// Fused prepare: barrier + stamp/zero + routing metadata
// ====================================================================
// Fuses p2p_barrier(RESET_COMBINE) + stamp_and_zero_dispatch
// + routing extraction into one kernel. Block 0 does the
// cross-GPU barrier; other blocks spin on the counter.
// Grid = kPersistentGrid, block = kBlockSize.
// expert_num_tokens must be pre-zeroed before launch.
template <typename T>
__global__ void prepare_dispatch_recv_kernel(
    T* __restrict__ dispatch_recv,
    int64_t* __restrict__ expert_topk_ids,
    float* __restrict__ expert_topk_weights,
    int32_t* __restrict__ expert_num_tokens,
    const DispatchCombineConfig* __restrict__ config,
    int32_t mc,
    int32_t K,
    int32_t num_experts) {
  // Phase 1: Inline barrier (RESET_COMBINE).
  // All blocks read counter before block 0 modifies it.
  // Safe: kPersistentGrid blocks all start on separate
  // SMs within nanoseconds; barrier takes microseconds.
  FlagType expected =
      config->self_signals->counter + 1;

  if (blockIdx.x == 0) {
    const int32_t rank = config->rank;
    const int32_t ws = config->world_size;
    const int32_t tid = threadIdx.x;

    // Reset combine offset (thread 0).
    if (tid == 0) {
      if (config->remote_combine_offsets[rank])
        *config->remote_combine_offsets[rank] = 0;
    }

    __threadfence_system();

    if (tid < ws) {
      dc_st_flag_release(
          &config->peer_signals[tid]->flags[rank],
          expected);
      while (dc_ld_flag_acquire(
          &config->self_signals->flags[tid])
              != expected)
        ;
    }

    __syncthreads();

    if (tid == 0) {
      dc_st_flag_release(
          &config->self_signals->counter,
          expected);
    }
  } else {
    // Wait for block 0 to complete barrier.
    if (threadIdx.x == 0) {
      while (dc_ld_flag_acquire(
          &config->self_signals->counter)
              != expected)
        ;
    }
    __syncthreads();
  }

  // Phase 2: Stamp/zero + routing (persistent loop).
  const int32_t rank = config->rank;
  int32_t actual =
      *config->remote_dispatch_offsets[rank];
  if (actual > config->max_recv)
    actual = config->max_recv;

  for (int32_t idx = blockIdx.x; idx < mc;
       idx += gridDim.x) {
    if (idx < actual) {
      if (threadIdx.x == 0) {
        const TokenMetadata* meta =
            reinterpret_cast<const TokenMetadata*>(
                config->remote_dispatch_meta[rank]);
        int32_t eid = meta[idx].expert_id;
        expert_topk_ids[idx] =
            static_cast<int64_t>(eid);
        expert_topk_weights[idx] = 1.0f;
        if (eid >= 0 && eid < num_experts) {
          atomicAdd(&expert_num_tokens[eid], 1);
        }
      }
    } else {
      T* dest = dispatch_recv + idx * K;
      for (int32_t k = threadIdx.x; k < K;
           k += blockDim.x) {
        dest[k] = T(0);
      }
      if (threadIdx.x == 0) {
        TokenMetadata* meta =
            reinterpret_cast<TokenMetadata*>(
                config->remote_dispatch_meta[rank]);
        meta[idx].source_rank = 0;
        meta[idx].source_token_idx = 0;
        meta[idx].expert_id = num_experts;
        meta[idx].topk_weight = 0.0f;
        expert_topk_ids[idx] =
            static_cast<int64_t>(num_experts);
        expert_topk_weights[idx] = 0.0f;
      }
    }
  }
}

// ====================================================================
// Scatter-add v2: reads directly from IPC buffers
// ====================================================================
// Replaces copy_combine_recv + copy_combine_meta +
// scatter_add_weighted. Reads IPC buffers via config ptrs.
// Grid = mc, block = kBlockSize.
template <typename T>
__global__ void scatter_add_v2_kernel(
    float* __restrict__ output,
    const DispatchCombineConfig* __restrict__ config,
    int32_t N_recv,
    int32_t K) {
  const int32_t idx = blockIdx.x;
  if (idx >= N_recv) return;

  const int32_t rank = config->rank;
  int32_t actual =
      *config->remote_combine_offsets[rank];
  if (actual > config->max_recv)
    actual = config->max_recv;
  if (idx >= actual) return;

  const TokenMetadata* meta =
      reinterpret_cast<const TokenMetadata*>(
          config->remote_combine_meta[rank]);
  const T* recv = reinterpret_cast<const T*>(
      config->remote_combine_recv[rank]);

  const int32_t token_idx =
      meta[idx].source_token_idx;
  const float weight = meta[idx].topk_weight;
  if (weight == 0.0f) return;

  for (int32_t k = threadIdx.x; k < K;
       k += blockDim.x) {
    float val = static_cast<float>(
        recv[idx * K + k]);
    atomicAdd(
        output + token_idx * K + k, val * weight);
  }
}

// ====================================================================
// Scatter-add direct: reads from IPC combine buffers
// ====================================================================
// Reads combine recv/meta via IPC pointers in config.
// Uses native bf16/fp16 atomicAdd (SM_80+/SM_70+).
// Output must be pre-zeroed via cudaMemsetAsync.
// Grid = mc, block = kBlockSize.
template <typename T>
__global__ void scatter_add_direct_kernel(
    T* __restrict__ output,
    const DispatchCombineConfig* __restrict__ config,
    int32_t N_recv,
    int32_t K) {
  const int32_t idx = blockIdx.x;
  if (idx >= N_recv) return;

  const int32_t rank = config->rank;
  int32_t actual =
      *config->remote_combine_offsets[rank];
  if (actual > config->max_recv)
    actual = config->max_recv;
  if (idx >= actual) return;

  const TokenMetadata* meta =
      reinterpret_cast<const TokenMetadata*>(
          config->remote_combine_meta[rank]);
  const T* recv = reinterpret_cast<const T*>(
      config->remote_combine_recv[rank]);

  const int32_t token_idx =
      meta[idx].source_token_idx;
  const float weight = meta[idx].topk_weight;
  if (weight == 0.0f) return;

  for (int32_t k = threadIdx.x; k < K;
       k += blockDim.x) {
    float val = static_cast<float>(
        recv[idx * K + k]);
    atomicAdd(
        output + token_idx * K + k,
        static_cast<T>(val * weight));
  }
}

// ====================================================================
// Host-callable wrappers
// ====================================================================
void dispatch_p2p(
    torch::Tensor input,
    torch::Tensor topk_ids,
    torch::Tensor topk_weights,
    torch::Tensor config_tensor,
    int64_t M, int64_t K, int64_t topk);

// combine_p2p: grid = max_recv, reads actual count from config.
void combine_p2p(
    torch::Tensor expert_output,
    torch::Tensor dispatch_meta,
    torch::Tensor config_tensor,
    int64_t max_recv, int64_t K);

void scatter_add_weighted(
    torch::Tensor output,
    torch::Tensor combine_recv,
    torch::Tensor combine_meta,
    int64_t N_recv, int64_t K);

// GPU-side buffer operations (CUDA-graph compatible).
void reset_offsets(torch::Tensor config_tensor);
void reset_combine_offset(torch::Tensor config_tensor);

// P2P flag-based barriers (replace NCCL AllReduce).
void p2p_barrier(torch::Tensor config_tensor);
void p2p_barrier_reset_offsets(
    torch::Tensor config_tensor);
void p2p_barrier_reset_combine_offset(
    torch::Tensor config_tensor);
void p2p_barrier_reset_dispatch(
    torch::Tensor config_tensor);
void copy_dispatch_recv(
    torch::Tensor output,
    torch::Tensor config_tensor,
    int64_t max_recv, int64_t K);
void copy_dispatch_meta(
    torch::Tensor output,
    torch::Tensor config_tensor,
    int64_t max_recv);
void copy_combine_recv(
    torch::Tensor output,
    torch::Tensor config_tensor,
    int64_t max_recv, int64_t K);
void copy_combine_meta(
    torch::Tensor output,
    torch::Tensor config_tensor,
    int64_t max_recv);

// New fused kernels (eliminate copy overhead).
void stamp_and_zero_dispatch(
    torch::Tensor dispatch_recv,
    torch::Tensor config_tensor,
    int64_t mc, int64_t K);
void scatter_add_v2(
    torch::Tensor output,
    torch::Tensor config_tensor,
    int64_t mc, int64_t K,
    int64_t dtype_code);
torch::Tensor wrap_cuda_ptr(
    torch::Tensor dummy,
    int64_t ptr, int64_t dim0, int64_t dim1,
    int64_t dtype_code);
void prepare_dispatch_recv(
    torch::Tensor dispatch_recv,
    torch::Tensor expert_topk_ids,
    torch::Tensor expert_topk_weights,
    torch::Tensor expert_num_tokens,
    torch::Tensor config_tensor,
    int64_t mc, int64_t K,
    int64_t num_experts);
void scatter_add_direct(
    torch::Tensor output,
    torch::Tensor config_tensor,
    int64_t mc, int64_t K,
    int64_t M);

}  // namespace dispatch_combine
}  // namespace vllm
