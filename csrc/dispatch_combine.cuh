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

  // ---- Integrated routing fields (EPLB) ----
  // Push-based all-reduce: IPC ptrs to each rank's
  // expert_counts buffer. During dispatch, each block
  // atomicAdds to ALL peers' buffers. After barrier,
  // local buffer has global sum.
  int32_t* remote_expert_counts[kMaxRanks];

  // GPU-resident routing tables (updated on EPLB rebalance).
  int32_t* logical_to_physical_map;   // [NL * max_replicas]
  int64_t* logical_replica_count;     // [NL]
  int32_t* routing_selection;         // [NL] output

  // Scalars for integrated routing.
  int32_t num_logical_experts;
  int32_t max_replicas;               // slots_per_logical
  int32_t physical_experts_per_rank;

  // Intra-kernel sync for routing completion
  // (CUDA graph compatible, monotonic counter).
  FlagType* routing_ready_flag;

  // Grid-wide sync: all blocks increment this counter
  // after Phase A completes. Block 0 spins until
  // counter == base + gridDim.x before entering the
  // P2P barrier. Monotonic for CUDA graph replay.
  FlagType* phase_a_done_counter;

  // Local expert counts buffer for batched all-reduce.
  // Each block atomicAdds its shared-mem counts here;
  // block 0 reads the aggregate and pushes to all ranks'
  // remote_expert_counts in one pass (after grid-wide sync).
  // Zeroed by Phase E; first invocation by cudaMemset.
  int32_t* local_expert_counts;  // [NL]
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

    // Skip filtered entries (integrated routing sets
    // weight=0 for tokens whose chosen replica is not
    // on this rank).
    if (weight == 0.0f) continue;

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

// ====================================================================
// Fused dispatch + route + filter kernel (integrated EPLB)
// ====================================================================
// Single persistent-grid kernel that:
//   Phase A: Broadcast-dispatches tokens to all replica ranks
//            + push-based all-reduce (remote atomicAdd to all
//            peers' expert_counts buffers).
//   Phase B: P2P barrier (shared: covers dispatch + all-reduce).
//   Phase C: Deterministic router (block 0, sequential).
//   Phase D: Filter + stamp/zero + routing metadata.
//
// Grid = kPersistentGrid, block = kBlockSize.
// expert_num_tokens must be pre-zeroed before launch.
// remote_expert_counts[rank] is self-zeroing via Phase E
// (first invocation: zeroed during init_integrated_routing).
template <typename T>
__global__ void dispatch_and_route_kernel(
    const T* __restrict__ input,
    const int32_t* __restrict__ topk_ids,
    const float* __restrict__ topk_weights,
    T* __restrict__ dispatch_recv,
    int64_t* __restrict__ expert_topk_ids,
    float* __restrict__ expert_topk_weights,
    int32_t* __restrict__ expert_num_tokens,
    int32_t* __restrict__ data_remap,
    const DispatchCombineConfig* __restrict__ config,
    int32_t M, int32_t K, int32_t topk,
    int32_t mc,
    int32_t num_physical_experts) {
  const int32_t rank = config->rank;
  const int32_t ws = config->world_size;
  const int32_t max_rep = config->max_replicas;
  const int32_t epr = config->physical_experts_per_rank;
  const int32_t NL = config->num_logical_experts;

  // Shared memory layout (reused across phases):
  // Phase A: s_expert_counts[NL] + s_grp_dest[ws]
  //        + s_grp_count[ws] + s_ent_lid[64]
  //        + s_ent_wt[64] + s_ent_grp[64]
  // Phase C: routing_selection_smem[NL]
  //        + rank_active_counts[ws]
  // Phases don't overlap, so same memory is reused.
  extern __shared__ int32_t shared[];
  int32_t* routing_selection_smem = shared;
  int32_t* rank_active_counts = shared + NL;

  // Read monotonic counter base values BEFORE Phase A
  // (for CUDA graph replay compatibility).
  FlagType rf_expected = 0;
  FlagType pa_base = 0;
  if (threadIdx.x == 0) {
    rf_expected =
        static_cast<FlagType>(*config->routing_ready_flag) + 1;
    pa_base =
        static_cast<FlagType>(*config->phase_a_done_counter);
  }
  // Broadcast to all threads via shared mem.
  __shared__ FlagType s_rf_expected;
  __shared__ FlagType s_pa_base;
  if (threadIdx.x == 0) {
    s_rf_expected = rf_expected;
    s_pa_base = pa_base;
  }
  __syncthreads();
  rf_expected = s_rf_expected;
  pa_base = s_pa_base;

  // ---- Phase A: Broadcast dispatch + batched all-reduce ----
  // Per-token loop: group (slot, replica) by dest_rank to
  // write token data ONCE per unique (token, dest_rank).
  // Expert counts accumulated in shared memory, flushed to
  // local device buffer after loop, pushed to all ranks by
  // block 0 after grid-wide sync.
  // Reuses extern shared[] (Phase C only runs after barrier).
  constexpr int32_t kMaxEntries = 64;  // topk * max_rep
  int32_t* s_expert_counts = shared;              // [NL]
  int32_t* s_grp_dest  = shared + NL;             // [ws]
  int32_t* s_grp_count = shared + NL + ws;        // [ws]
  int32_t* s_ent_lid   = shared + NL + 2 * ws;    // [kMaxEntries]
  float*   s_ent_wt    = reinterpret_cast<float*>(
      s_ent_lid + kMaxEntries);                    // [kMaxEntries]
  int32_t* s_ent_grp   = reinterpret_cast<int32_t*>(
      s_ent_wt + kMaxEntries);                     // [kMaxEntries]

  __shared__ int32_t s_num_groups;
  __shared__ int32_t s_total_entries;
  __shared__ int32_t s_grp_base[64];   // claimed write_pos

  // Step 0: Zero shared expert counts (persistent across
  // all token iterations; flushed to device mem after loop).
  for (int32_t e = threadIdx.x; e < NL;
       e += blockDim.x) {
    s_expert_counts[e] = 0;
  }
  __syncthreads();

  for (int32_t t = blockIdx.x; t < M; t += gridDim.x) {
    // Step 1: Thread 0 scans slots+replicas, groups by
    // dest_rank, accumulates expert counts in shared mem.
    if (threadIdx.x == 0) {
      s_num_groups = 0;
      s_total_entries = 0;
      for (int32_t slot = 0; slot < topk; slot++) {
        int32_t lid = topk_ids[t * topk + slot];
        if (lid < 0 || lid >= NL) continue;

        // Local count (shared mem, ~1 cycle).
        s_expert_counts[lid]++;

        int32_t rc = static_cast<int32_t>(
            config->logical_replica_count[lid]);
        if (rc > max_rep) rc = max_rep;
        for (int32_t rep = 0; rep < rc; rep++) {
          int32_t phys =
              config->logical_to_physical_map[
                  lid * max_rep + rep];
          int32_t dr = phys / epr;
          if (dr < 0 || dr >= ws) continue;
          if (!config->remote_dispatch_offsets[dr] ||
              !config->remote_dispatch_recv[dr] ||
              !config->remote_dispatch_meta[dr])
            continue;
          // Find or create group for dest_rank.
          int32_t g = -1;
          for (int32_t i = 0; i < s_num_groups; i++) {
            if (s_grp_dest[i] == dr) {
              g = i; break;
            }
          }
          if (g == -1) {
            g = s_num_groups++;
            s_grp_dest[g] = dr;
            s_grp_count[g] = 0;
          }
          int32_t ei = s_total_entries;
          if (ei < kMaxEntries) {
            s_ent_lid[ei] = lid;
            s_ent_wt[ei] =
                topk_weights[t * topk + slot];
            s_ent_grp[ei] = g;
            s_grp_count[g]++;
            s_total_entries = ei + 1;
          }
        }
      }
    }
    __syncthreads();

    // Step 2: Thread 0 claims contiguous positions.
    if (threadIdx.x == 0) {
      for (int32_t g = 0; g < s_num_groups; g++) {
        s_grp_base[g] = atomicAdd(
            config->remote_dispatch_offsets[
                s_grp_dest[g]],
            s_grp_count[g]);
      }
    }
    __syncthreads();

    // Step 3: Write data ONCE + N metadata per group.
    // Entries are interleaved across groups in the flat
    // arrays, so filter by s_ent_grp[ei] == g.
    for (int32_t g = 0; g < s_num_groups; g++) {
      int32_t dr = s_grp_dest[g];
      int32_t base = s_grp_base[g];
      int32_t n = s_grp_count[g];
      if (base >= config->max_recv) continue;
      if (base + n > config->max_recv)
        n = config->max_recv - base;

      // All threads: write token data ONCE at base.
      T* dest = reinterpret_cast<T*>(
          config->remote_dispatch_recv[dr]);
      const T* src = input + t * K;
      for (int32_t k = threadIdx.x; k < K;
           k += blockDim.x) {
        dest[base * K + k] = src[k];
      }

      // Thread 0: write metadata for matching entries.
      if (threadIdx.x == 0) {
        TokenMetadata* meta =
            reinterpret_cast<TokenMetadata*>(
                config->remote_dispatch_meta[dr]);
        int32_t mi = 0;
        for (int32_t ei = 0;
             ei < s_total_entries && mi < n;
             ei++) {
          if (s_ent_grp[ei] != g) continue;
          meta[base + mi].source_rank = rank;
          meta[base + mi].source_token_idx = t;
          meta[base + mi].expert_id = s_ent_lid[ei];
          meta[base + mi].topk_weight = s_ent_wt[ei];
          mi++;
        }
      }
    }
    __syncthreads();  // before next token overwrites smem
  }

  // Flush shared-mem expert counts to local device buffer.
  // All blocks contribute; local_expert_counts accumulates
  // this rank's total expert counts for the batch push.
  for (int32_t e = threadIdx.x; e < NL;
       e += blockDim.x) {
    int32_t count = s_expert_counts[e];
    if (count > 0) {
      atomicAdd(&config->local_expert_counts[e],
                count);
    }
  }

  // ---- Grid-wide sync: wait for ALL blocks to finish
  //      Phase A before block 0 enters P2P barrier. ----
  // Covers dispatch data writes + local_expert_counts
  // flushes. Block 0 then pushes aggregated counts to
  // all remote ranks before the P2P barrier.
  __syncthreads();
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    atomicAdd(config->phase_a_done_counter,
              static_cast<FlagType>(1));
  }
  if (blockIdx.x == 0) {
    if (threadIdx.x == 0) {
      FlagType target = pa_base + gridDim.x;
      while (dc_ld_flag_acquire(
                 config->phase_a_done_counter)
              < target)
        ;
    }
    __syncthreads();

    // Batch all-reduce push: block 0 reads aggregated
    // local counts and pushes to ALL ranks' buffers.
    // P2P barrier's __threadfence_system() will make
    // these writes visible to peers.
    for (int32_t e = threadIdx.x; e < NL;
         e += blockDim.x) {
      int32_t count =
          config->local_expert_counts[e];
      if (count > 0) {
        for (int32_t r = 0; r < ws; r++) {
          atomicAdd(
              &config->remote_expert_counts[r][e],
              count);
        }
      }
    }
    __syncthreads();
  }

  // ---- Phase B: P2P barrier (RESET_COMBINE) ----
  // Shared barrier covers dispatch data writes AND
  // remote expert_counts batch push.
  FlagType expected =
      config->self_signals->counter + 1;

  if (blockIdx.x == 0) {
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
    if (threadIdx.x == 0) {
      while (dc_ld_flag_acquire(
          &config->self_signals->counter)
              != expected)
        ;
    }
    __syncthreads();
  }

  // ---- Phase C: Deterministic router (block 0) ----
  if (blockIdx.x == 0) {
    // Read global expert counts from local buffer
    // (already has sum from push all-reduce).
    int32_t* expert_counts =
        config->remote_expert_counts[rank];

    // Initialize shared memory.
    for (int32_t e = threadIdx.x; e < NL;
         e += blockDim.x) {
      routing_selection_smem[e] = -1;
    }
    for (int32_t r = threadIdx.x; r < ws;
         r += blockDim.x) {
      rank_active_counts[r] = 0;
    }
    __syncthreads();

    // Sequential routing (thread 0 only).
    // Process experts in ascending order for
    // determinism across all ranks.
    if (threadIdx.x == 0) {
      for (int32_t e = 0; e < NL; e++) {
        const int32_t count = expert_counts[e];
        if (count == 0) continue;

        int32_t rc = static_cast<int32_t>(
            config->logical_replica_count[e]);
        if (rc <= 0) continue;
        if (rc > max_rep) rc = max_rep;

        if (rc == 1) {
          const int32_t phys =
              config->logical_to_physical_map[
                  e * max_rep];
          routing_selection_smem[e] = phys;
          rank_active_counts[phys / epr] += count;
          continue;
        }

        // Multiple replicas: pick minimum-loaded rank
        // (ties: lower rank for determinism).
        int32_t best_phys = -1;
        int32_t best_rank = -1;
        int32_t best_cost = INT_MAX;
        for (int32_t i = 0; i < rc; i++) {
          const int32_t phys =
              config->logical_to_physical_map[
                  e * max_rep + i];
          const int32_t r = phys / epr;
          const int32_t c = rank_active_counts[r];
          if (c < best_cost ||
              (c == best_cost && r < best_rank)) {
            best_cost = c;
            best_rank = r;
            best_phys = phys;
          }
        }
        routing_selection_smem[e] = best_phys;
        rank_active_counts[best_rank] += count;
      }
    }
    __syncthreads();

    // Write routing_selection to global memory.
    for (int32_t e = threadIdx.x; e < NL;
         e += blockDim.x) {
      config->routing_selection[e] =
          routing_selection_smem[e];
    }
    __threadfence();

    // Signal routing complete.
    if (threadIdx.x == 0) {
      dc_st_flag_release(
          config->routing_ready_flag, rf_expected);
    }
  } else {
    // Other blocks: spin until routing complete.
    if (threadIdx.x == 0) {
      while (dc_ld_flag_acquire(
              config->routing_ready_flag)
              != rf_expected)
        ;
    }
    __syncthreads();
    __threadfence();
  }

  // ---- Phase D: Filter + stamp/zero + data_remap ----
  int32_t actual =
      *config->remote_dispatch_offsets[rank];
  if (actual > config->max_recv)
    actual = config->max_recv;

  for (int32_t idx = blockIdx.x; idx < mc;
       idx += gridDim.x) {
    if (idx < actual) {
      if (threadIdx.x == 0) {
        // Compute data_remap: find group leader.
        // Contiguous entries with same (source_rank,
        // source_token_idx) share data at the first
        // entry (Phase A writes data once per group).
        const TokenMetadata* meta_r =
            reinterpret_cast<const TokenMetadata*>(
                config->remote_dispatch_meta[rank]);
        int32_t leader = idx;
        if (idx > 0) {
          int32_t sr = meta_r[idx].source_rank;
          int32_t st = meta_r[idx].source_token_idx;
          int32_t check = idx - 1;
          while (check >= 0
                 && meta_r[check].source_rank == sr
                 && meta_r[check].source_token_idx
                     == st) {
            leader = check;
            check--;
          }
        }
        data_remap[idx] = leader;

        // Filtering logic: route to selected replica.
        TokenMetadata* meta =
            reinterpret_cast<TokenMetadata*>(
                config->remote_dispatch_meta[rank]);
        const int32_t logical_id =
            meta[idx].expert_id;
        if (logical_id < 0 || logical_id >= NL) {
          expert_topk_ids[idx] =
              static_cast<int64_t>(
                  num_physical_experts);
          expert_topk_weights[idx] = 0.0f;
          meta[idx].topk_weight = 0.0f;
        } else {
        const int32_t selected_phys =
            config->routing_selection[logical_id];

        // Guard: if routing produced -1 (e.g. expert
        // count was zero due to race), treat as FILTER.
        if (selected_phys < 0 ||
            selected_phys >= num_physical_experts) {
          expert_topk_ids[idx] =
              static_cast<int64_t>(
                  num_physical_experts);
          expert_topk_weights[idx] = 0.0f;
          meta[idx].topk_weight = 0.0f;
        } else if (selected_phys / epr == rank) {
          // KEEP: this token's replica is local.
          expert_topk_ids[idx] =
              static_cast<int64_t>(selected_phys);
          expert_topk_weights[idx] =
              meta[idx].topk_weight;
          atomicAdd(
              &expert_num_tokens[selected_phys],
              1);
        } else {
          // FILTER: not our replica.
          expert_topk_ids[idx] =
              static_cast<int64_t>(
                  num_physical_experts);
          expert_topk_weights[idx] = 0.0f;
          meta[idx].topk_weight = 0.0f;
        }
        }  // close logical_id bounds else
      }
    } else {
      // Stale entry: identity remap + zero + sentinel.
      if (threadIdx.x == 0) {
        data_remap[idx] = idx;
      }
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
        meta[idx].expert_id = num_physical_experts;
        meta[idx].topk_weight = 0.0f;
        expert_topk_ids[idx] =
            static_cast<int64_t>(
                num_physical_experts);
        expert_topk_weights[idx] = 0.0f;
      }
    }
  }

  // ---- Phase E: Zero counts for next invocation ----
  // Must happen AFTER Phase C reads the counts and AFTER
  // Phase B barrier guarantees no more remote atomicAdds.
  // The next layer's combine barrier (RESET_DISPATCH)
  // includes __threadfence_system() which ensures this
  // zeroing is visible to all peers before they start
  // the next dispatch_and_route.
  {
    int32_t* remote_ec =
        config->remote_expert_counts[rank];
    int32_t* local_ec =
        config->local_expert_counts;
    for (int32_t e = blockIdx.x * blockDim.x + threadIdx.x;
         e < NL;
         e += gridDim.x * blockDim.x) {
      remote_ec[e] = 0;
      local_ec[e] = 0;
    }
  }
}

// Host-callable wrapper for fused dispatch+route+filter.
void dispatch_and_route(
    torch::Tensor input,
    torch::Tensor topk_ids,
    torch::Tensor topk_weights,
    torch::Tensor dispatch_recv,
    torch::Tensor expert_topk_ids,
    torch::Tensor expert_topk_weights,
    torch::Tensor expert_num_tokens,
    torch::Tensor expert_counts,
    torch::Tensor data_remap,
    torch::Tensor config_tensor,
    int64_t M, int64_t K, int64_t topk,
    int64_t mc,
    int64_t num_physical_experts,
    int64_t num_logical_experts,
    int64_t world_size);

}  // namespace dispatch_combine
}  // namespace vllm
