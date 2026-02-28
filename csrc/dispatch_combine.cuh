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
  // twice per invocation (after Pass 1 and Pass 2).
  // Block 0 spins until counter == base + gridDim.x
  // (Pass 1 done) then base + 2*gridDim.x (Pass 2
  // done + fence). Monotonic for CUDA graph replay.
  FlagType* phase_a_done_counter;

  // Local expert counts buffer for batched all-reduce.
  // Each block atomicAdds its shared-mem counts here;
  // block 0 reads the aggregate and pushes to all ranks'
  // remote_expert_counts in one pass (after grid-wide sync).
  // Zeroed by Phase E; first invocation by cudaMemset.
  int32_t* local_expert_counts;  // [NL]

  // ---- Per-sender section support ----
  // Eliminates remote atomicAdd for write-position claiming.
  // Each source rank owns a section of size section_size in
  // each destination's recv buffer. Position = rank *
  // section_size + local_offset. Local counters track
  // per-destination write counts; pushed to remote
  // per-sender offset arrays before each barrier.
  int32_t dispatch_section_size;        // max_recv / ws
  int32_t combine_section_size;         // max_recv / ws
  int32_t* local_dispatch_counters;     // [kMaxRanks] local
  int32_t* local_combine_counters;      // [kMaxRanks] local

  // Grid-wide sync for fused combine_and_scatter kernel.
  // All blocks increment after Phase 1 (combine P2P writes);
  // block 0 spins until all done before entering barrier.
  FlagType* combine_done_counter;

  // ---- Two-pass dispatch support ----
  // Pass 1 writes per-block per-rank entry counts here;
  // block 0 reads them during aggregation for prefix sum.
  int32_t* block_dispatch_counts;      // [kPersistentGrid * kMaxRanks]
  // Block 0 writes per-block starting write positions here
  // after prefix sum; all blocks read in Pass 2.
  int32_t* block_start_positions;      // [kPersistentGrid * kMaxRanks]
  // Block 0 signals this flag after computing positions;
  // blocks 1-31 spin until ready. Monotonic for CUDA graph.
  FlagType* positions_ready_flag;
};

// ====================================================================
// P2P flag-based barrier kernels (replace NCCL AllReduce).
// Launch with 1 block, kMaxRanks threads.
// ====================================================================
enum class BarrierMode : int {
  PURE = 0,            // Signal + wait only
  RESET_DISPATCH = 1,  // Reset dispatch offset + barrier
};

template <BarrierMode mode>
__global__ void p2p_barrier_kernel(
    const DispatchCombineConfig* __restrict__ config) {
  const int32_t rank = config->rank;
  const int32_t ws = config->world_size;
  const int32_t tid = threadIdx.x;
  if (tid >= ws) return;

  // Optional reset (per-sender sections).
  if constexpr (mode == BarrierMode::RESET_DISPATCH) {
    if (tid < ws) {
      // Push combine section counts to remote ranks.
      // Thread tid pushes this rank's count for dest tid.
      if (config->remote_combine_offsets[tid])
        config->remote_combine_offsets[tid][rank] =
            config->local_combine_counters[tid];
      // Reset per-sender dispatch offsets at our rank.
      if (config->remote_dispatch_offsets[rank])
        config->remote_dispatch_offsets[rank][tid] = 0;
      // Reset local counters for next layer.
      config->local_dispatch_counters[tid] = 0;
      config->local_combine_counters[tid] = 0;
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
    // Local atomicAdd for position claiming.
    int32_t local_off = atomicAdd(
        &config->local_dispatch_counters[dest_rank], 1);
    int32_t ss = config->dispatch_section_size;
    // Section overflow: drop if section full.
    s_write_pos = (local_off < ss)
        ? rank * ss + local_off : config->max_recv;
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
    const int32_t* __restrict__ compact_reverse,
    const DispatchCombineConfig* __restrict__ config,
    int32_t K) {
  // Section-aware iteration: loop only over real entries
  // in each sender's section. Avoids iterating all mc
  // entries (65536) when only ~800 are real, eliminating
  // expensive runtime integer div/mod per stale entry.
  const int32_t rank_c = config->rank;
  const int32_t ws_c = config->world_size;
  const int32_t ss_d = config->dispatch_section_size;

  __shared__ int32_t s_write_pos;

  for (int32_t s = 0; s < ws_c; s++) {
    int32_t section_start = s * ss_d;
    // Clamp to section size: raw counter may exceed ss_d
    // due to overflow counting in dispatch atomicAdd.
    int32_t count = config->
        remote_dispatch_offsets[rank_c][s];
    if (count > ss_d) count = ss_d;
    for (int32_t pair_idx = section_start + blockIdx.x;
         pair_idx < section_start + count;
         pair_idx += gridDim.x) {

      const int32_t dest_rank =
          dispatch_meta[pair_idx].source_rank;
      const int32_t orig_token_idx =
          dispatch_meta[pair_idx].source_token_idx;
      const float weight =
          dispatch_meta[pair_idx].topk_weight;

      // Skip routing-filtered entries (weight==0).
      if (weight == 0.0f) continue;

      if (dest_rank < 0 ||
          dest_rank >= config->world_size)
        continue;

      if (!config->remote_combine_offsets[dest_rank] ||
          !config->remote_combine_recv[dest_rank] ||
          !config->remote_combine_meta[dest_rank])
        continue;

      if (threadIdx.x == 0) {
        int32_t local_off = atomicAdd(
            &config->local_combine_counters[
                dest_rank], 1);
        int32_t ss = config->combine_section_size;
        s_write_pos = (local_off < ss)
            ? config->rank * ss + local_off
            : config->max_recv;
      }
      __syncthreads();

      const int32_t write_pos = s_write_pos;
      if (write_pos >= config->max_recv) continue;

      T* dest_data = reinterpret_cast<T*>(
          config->remote_combine_recv[dest_rank]);
      // Read expert output from compact position.
      const int32_t ci =
          compact_reverse[pair_idx];
      const T* src_data =
          expert_output + ci * K;
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
    }
  }
}

// ====================================================================
// Fused prepare: barrier + stamp/zero + routing metadata
// ====================================================================
// Fuses p2p_barrier(RESET_COMBINE) + stamp_and_zero_dispatch
// + routing extraction into one kernel. Block 0 does the
// cross-GPU barrier; other blocks spin on the counter.
// Grid = kPersistentGrid, block = kBlockSize.
// expert_num_tokens is zeroed inline by block 0 after
// barrier, before signaling other blocks.
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

    // Push dispatch section counts to remote ranks.
    // Reset per-sender combine offsets + local counters.
    if (tid < ws) {
      if (config->remote_dispatch_offsets[tid])
        config->remote_dispatch_offsets[tid][rank] =
            config->local_dispatch_counters[tid];
      if (config->remote_combine_offsets[rank])
        config->remote_combine_offsets[rank][tid] = 0;
      config->local_combine_counters[tid] = 0;
      config->local_dispatch_counters[tid] = 0;
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

    // Zero expert_num_tokens for Phase 2's atomicAdd.
    for (int32_t i = tid; i < num_experts;
         i += blockDim.x) {
      expert_num_tokens[i] = 0;
    }
    __threadfence();
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
  // Section-aware: each sender owns a section of size
  // dispatch_section_size. Entry is real if its offset
  // within its section < that section's count.
  const int32_t rank = config->rank;
  const int32_t ws_p = config->world_size;
  const int32_t ss_p = config->dispatch_section_size;

  for (int32_t idx = blockIdx.x; idx < mc;
       idx += gridDim.x) {
    int32_t sec = idx / ss_p;
    int32_t off = idx % ss_p;
    int32_t sec_cnt_p = config->
        remote_dispatch_offsets[rank][sec];
    if (sec_cnt_p > ss_p) sec_cnt_p = ss_p;
    bool is_real = (sec < ws_p)
        && (off < sec_cnt_p);
    if (is_real) {
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
      // Stale: K-data zeroing removed (never consumed,
      // bounded by expert_num_tokens). Metadata sentinels
      // removed (combine_p2p is section-aware).
      // Only expert_topk_ids/weights needed by
      // moe_align_block_size downstream.
      if (threadIdx.x == 0) {
        expert_topk_ids[idx] =
            static_cast<int64_t>(num_experts);
        expert_topk_weights[idx] = 0.0f;
      }
    }
  }
}

// ====================================================================
// Scatter-add: entry-parallel via fp32 atomicAdd
// ====================================================================
// Reads combine recv/meta via IPC pointers in config.
// Each block processes a stride of entries, atomicAdds
// weighted values to fp32 accum buffer. Threads tile
// over K columns (coalesced). fp32 atomicAdd is native
// on sm_80+ (no CAS loop, no adjacent-element
// contention from paired 32-bit words).
// Accum buffer MUST be pre-zeroed by host wrapper.
// O(entries) metadata reads — each entry scanned once.
template <typename T>
__global__ void scatter_add_atomic_kernel(
    float* __restrict__ accum,
    const DispatchCombineConfig* __restrict__ config,
    int32_t K) {
  const int32_t rank = config->rank;
  const int32_t ws = config->world_size;
  const int32_t ss_c = config->combine_section_size;

  const TokenMetadata* meta =
      reinterpret_cast<const TokenMetadata*>(
          config->remote_combine_meta[rank]);
  const T* recv = reinterpret_cast<const T*>(
      config->remote_combine_recv[rank]);

  for (int32_t s = 0; s < ws; s++) {
    int32_t sec_start = s * ss_c;
    int32_t count =
        config->remote_combine_offsets[rank][s];
    if (count > ss_c) count = ss_c;
    for (int32_t idx = sec_start + blockIdx.x;
         idx < sec_start + count;
         idx += gridDim.x) {

      int32_t tok = meta[idx].source_token_idx;
      float w = meta[idx].topk_weight;
      if (w == 0.0f) continue;

      for (int32_t k = threadIdx.x; k < K;
           k += blockDim.x) {
        float val = static_cast<float>(
            recv[idx * K + k]) * w;
        atomicAdd(accum + tok * K + k, val);
      }
    }
  }
}

// Convert fp32 accumulation buffer to half-precision.
template <typename T>
__global__ void fp32_to_half_kernel(
    T* __restrict__ output,
    const float* __restrict__ input,
    int32_t N) {
  for (int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
       i < N; i += gridDim.x * blockDim.x) {
    output[i] = static_cast<T>(input[i]);
  }
}

// ====================================================================
// Fused combine + barrier + scatter-add kernel
// ====================================================================
// Replaces 3 separate kernel launches:
//   combine_p2p + p2p_barrier_reset_dispatch + scatter_add_direct
// Phase 0: Zero fp32 accum buffer (all blocks cooperate).
// Phase 1: Combine P2P writes (persistent grid loop).
// Grid-wide sync: all blocks done writing.
// Phase 2: Inline P2P barrier (RESET_DISPATCH).
// Phase 3: Scatter-add to fp32 accum (native atomicAdd).
// Host launches fp32_to_half_kernel after to convert
// accum → output. Grid = kCombineScatterGrid, block = kBlockSize.
template <typename T>
__global__ void combine_and_scatter_kernel(
    const T* __restrict__ expert_output,
    const TokenMetadata* __restrict__ dispatch_meta,
    const int32_t* __restrict__ compact_reverse,
    float* __restrict__ accum,
    const DispatchCombineConfig* __restrict__ config,
    int32_t mc, int32_t K, int32_t M) {
  const int32_t rank = config->rank;
  const int32_t ws = config->world_size;

  // Read monotonic counter bases BEFORE any phase
  // modifies them (CUDA graph replay compatible).
  __shared__ FlagType s_cd_base;
  __shared__ FlagType s_barrier_expected;
  if (threadIdx.x == 0) {
    s_cd_base = static_cast<FlagType>(
        *config->combine_done_counter);
    s_barrier_expected =
        config->self_signals->counter + 1;
  }
  __syncthreads();
  FlagType cd_base = s_cd_base;
  FlagType barrier_expected = s_barrier_expected;

  // ---- Phase 0: Zero fp32 accum buffer ----
  // Vectorized: int4 = 16 bytes = 4 floats.
  // Phase 1 doesn't touch accum, and Phase 3 follows
  // multiple system fences + barrier, so no extra sync.
  {
    int4* out4 = reinterpret_cast<int4*>(accum);
    constexpr int32_t kElemsPerI4 =
        static_cast<int32_t>(
            sizeof(int4) / sizeof(float));
    int32_t n4 = M * K / kElemsPerI4;
    int4 z4 = make_int4(0, 0, 0, 0);
    for (int32_t i =
             blockIdx.x * blockDim.x + threadIdx.x;
         i < n4; i += gridDim.x * blockDim.x) {
      out4[i] = z4;
    }
  }

  // ---- Phase 1: Combine P2P writes (persistent blocks) ----
  // Only kPersistentGrid (32) blocks do NVLink writes.
  // Less contention on local_combine_counters, and only
  // 32 blocks call __threadfence_system() (staggered).
  __shared__ int32_t s_write_pos;
  if (blockIdx.x < kPersistentGrid) {
    const int32_t ss_d = config->dispatch_section_size;
    for (int32_t s = 0; s < ws; s++) {
      int32_t section_start = s * ss_d;
      int32_t count = config->
          remote_dispatch_offsets[rank][s];
      if (count > ss_d) count = ss_d;
      for (int32_t pair_idx =
               section_start + blockIdx.x;
           pair_idx < section_start + count;
           pair_idx += kPersistentGrid) {

        const int32_t dest_rank =
            dispatch_meta[pair_idx].source_rank;
        const int32_t orig_token_idx =
            dispatch_meta[pair_idx].source_token_idx;
        const float weight =
            dispatch_meta[pair_idx].topk_weight;

        if (weight == 0.0f) continue;
        if (dest_rank < 0 || dest_rank >= ws)
          continue;
        if (!config->
                remote_combine_offsets[dest_rank] ||
            !config->
                remote_combine_recv[dest_rank] ||
            !config->
                remote_combine_meta[dest_rank])
          continue;

        if (threadIdx.x == 0) {
          int32_t local_off = atomicAdd(
              &config->local_combine_counters[
                  dest_rank], 1);
          int32_t ss = config->combine_section_size;
          s_write_pos = (local_off < ss)
              ? rank * ss + local_off
              : config->max_recv;
        }
        __syncthreads();

        const int32_t write_pos = s_write_pos;
        if (write_pos >= config->max_recv) continue;

        T* dest_data = reinterpret_cast<T*>(
            config->remote_combine_recv[dest_rank]);
        const int32_t ci =
            compact_reverse[pair_idx];
        const T* src_data =
            expert_output + ci * K;
        for (int32_t k = threadIdx.x; k < K;
             k += blockDim.x) {
          dest_data[write_pos * K + k] =
              src_data[k];
        }

        if (threadIdx.x == 0) {
          TokenMetadata* dest_meta =
              reinterpret_cast<TokenMetadata*>(
                  config->remote_combine_meta[
                      dest_rank]);
          dest_meta[write_pos].source_rank = rank;
          dest_meta[write_pos].source_token_idx =
              orig_token_idx;
          dest_meta[write_pos].expert_id =
              dispatch_meta[pair_idx].expert_id;
          dest_meta[write_pos].topk_weight = weight;
        }
      }
    }

    // Staggered fence: blocks finish at different
    // times, avoiding NVLink ack congestion.
    __syncthreads();
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
      atomicAdd(config->combine_done_counter,
                static_cast<FlagType>(1));
    }
  }

  // ---- Phase 2: Single barrier ----
  // Block 0 waits for kPersistentGrid blocks, does
  // P2P barrier. All other blocks wait for counter.
  if (blockIdx.x == 0) {
    if (threadIdx.x == 0) {
      FlagType target = cd_base + kPersistentGrid;
      while (dc_ld_flag_acquire(
                 config->combine_done_counter)
              < target)
        ;
    }
    __syncthreads();

    const int32_t tid = threadIdx.x;
    if (tid < ws) {
      if (config->remote_combine_offsets[tid])
        config->remote_combine_offsets[tid][rank] =
            config->local_combine_counters[tid];
      if (config->remote_dispatch_offsets[rank])
        config->remote_dispatch_offsets[rank][tid]
            = 0;
      config->local_dispatch_counters[tid] = 0;
      config->local_combine_counters[tid] = 0;
    }

    __threadfence_system();

    if (tid < ws) {
      dc_st_flag_release(
          &config->peer_signals[tid]->flags[rank],
          barrier_expected);
      while (dc_ld_flag_acquire(
          &config->self_signals->flags[tid])
              != barrier_expected)
        ;
    }

    __syncthreads();

    if (tid == 0) {
      dc_st_flag_release(
          &config->self_signals->counter,
          barrier_expected);
    }
  } else {
    // Blocks 1-511: wait for barrier completion.
    // Blocks >= kPersistentGrid: device fence ensures
    // Phase 0 accum zeroing is L2-visible for Phase 3.
    if (blockIdx.x >= kPersistentGrid) {
      __threadfence();
    }
    if (threadIdx.x == 0) {
      while (dc_ld_flag_acquire(
          &config->self_signals->counter)
              != barrier_expected)
        ;
    }
    __syncthreads();
  }

  // ---- Phase 3: Scatter-add to fp32 accum ----
  // Native fp32 atomicAdd: no CAS loop, no adjacent-
  // element contention from packed 32-bit words.
  // Section-aware iteration: only visit real entries.
  {
    const int32_t ss_c = config->combine_section_size;
    const TokenMetadata* cmeta =
        reinterpret_cast<const TokenMetadata*>(
            config->remote_combine_meta[rank]);
    const T* crecv = reinterpret_cast<const T*>(
        config->remote_combine_recv[rank]);

    for (int32_t s = 0; s < ws; s++) {
      int32_t section_start = s * ss_c;
      int32_t count = config->
          remote_combine_offsets[rank][s];
      if (count > ss_c) count = ss_c;
      for (int32_t idx = section_start + blockIdx.x;
           idx < section_start + count;
           idx += gridDim.x) {

        int32_t token_idx =
            cmeta[idx].source_token_idx;
        float wt = cmeta[idx].topk_weight;
        if (wt == 0.0f) continue;

        for (int32_t k = threadIdx.x; k < K;
             k += blockDim.x) {
          float val = static_cast<float>(
              crecv[idx * K + k]) * wt;
          atomicAdd(
              accum + token_idx * K + k, val);
        }
      }
    }
  }
}

// ====================================================================
// Split-phase dispatch + route kernels (profiling mode)
// ====================================================================
// When VLLM_DC_SPLIT_KERNELS=1, dispatch_and_route is split into
// 6 separate kernels for per-phase latency measurement.
// Each kernel launch is an implicit grid barrier, eliminating
// the need for phase_a_done_counter, routing_ready_flag, and
// combine_done_counter used by the fused kernel.

// ---- Phase A: Broadcast dispatch + expert count accumulation ----
// Grid = kPersistentGrid, block = kBlockSize.
// Shared mem: (NL + 2*ws + 3*64) * 4 bytes.
template <typename T>
__global__ void dar_phase_a_kernel(
    const T* __restrict__ input,
    const int32_t* __restrict__ topk_ids,
    const float* __restrict__ topk_weights,
    const DispatchCombineConfig* __restrict__ config,
    int32_t M, int32_t K, int32_t topk) {
  const int32_t rank = config->rank;
  const int32_t ws = config->world_size;
  const int32_t max_rep = config->max_replicas;
  const int32_t epr = config->physical_experts_per_rank;
  const int32_t NL = config->num_logical_experts;

  extern __shared__ int32_t shared[];
  constexpr int32_t kMaxEntries = 64;
  int32_t* s_expert_counts = shared;
  int32_t* s_grp_dest  = shared + NL;
  int32_t* s_grp_count = shared + NL + ws;
  int32_t* s_ent_lid   = shared + NL + 2 * ws;
  float*   s_ent_wt    = reinterpret_cast<float*>(
      s_ent_lid + kMaxEntries);
  int32_t* s_ent_grp   = reinterpret_cast<int32_t*>(
      s_ent_wt + kMaxEntries);

  __shared__ int32_t s_num_groups;
  __shared__ int32_t s_total_entries;
  __shared__ int32_t s_grp_base[64];

  // Zero shared expert counts.
  for (int32_t e = threadIdx.x; e < NL;
       e += blockDim.x) {
    s_expert_counts[e] = 0;
  }
  if (threadIdx.x == 0) {
    s_num_groups = 0;
    s_total_entries = 0;
  }
  __syncthreads();

  // Per-token loop.
  for (int32_t t = blockIdx.x; t < M; t += gridDim.x) {
    // Step 1a: Parallel expansion (threads 0..topk-1).
    if (threadIdx.x < topk) {
      int32_t slot = threadIdx.x;
      int32_t lid = topk_ids[t * topk + slot];
      if (lid >= 0 && lid < NL) {
        atomicAdd(&s_expert_counts[lid], 1);
        float wt = topk_weights[t * topk + slot];
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
          int32_t ei =
              atomicAdd(&s_total_entries, 1);
          if (ei < kMaxEntries) {
            s_ent_lid[ei] = lid;
            s_ent_wt[ei] = wt;
            s_ent_grp[ei] = dr;
          }
        }
      }
    }
    __syncthreads();

    // Step 1b: Thread 0 groups by dest_rank.
    if (threadIdx.x == 0) {
      int32_t ne = s_total_entries;
      if (ne > kMaxEntries) ne = kMaxEntries;
      for (int32_t i = 0; i < ne; i++) {
        int32_t dr = s_ent_grp[i];
        int32_t g = -1;
        for (int32_t j = 0; j < s_num_groups; j++) {
          if (s_grp_dest[j] == dr) {
            g = j; break;
          }
        }
        if (g == -1) {
          g = s_num_groups++;
          s_grp_dest[g] = dr;
          s_grp_count[g] = 0;
        }
        s_ent_grp[i] = g;
        s_grp_count[g]++;
      }
    }
    __syncthreads();

    // Step 2: Claim write positions (local atomicAdd).
    if (threadIdx.x < s_num_groups) {
      int32_t dr = s_grp_dest[threadIdx.x];
      int32_t local_off = atomicAdd(
          &config->local_dispatch_counters[dr],
          s_grp_count[threadIdx.x]);
      int32_t ss = config->dispatch_section_size;
      s_grp_base[threadIdx.x] = (local_off < ss)
          ? rank * ss + local_off : config->max_recv;
    }
    __syncthreads();

    // Step 3: Write data + metadata per group.
    for (int32_t g = 0; g < s_num_groups; g++) {
      int32_t dr = s_grp_dest[g];
      int32_t base = s_grp_base[g];
      int32_t n = s_grp_count[g];
      if (base >= config->max_recv) continue;
      if (base + n > config->max_recv)
        n = config->max_recv - base;

      T* dest = reinterpret_cast<T*>(
          config->remote_dispatch_recv[dr]);
      const T* src = input + t * K;
      for (int32_t k = threadIdx.x; k < K;
           k += blockDim.x) {
        dest[base * K + k] = src[k];
      }

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
    if (threadIdx.x == 0) {
      s_num_groups = 0;
      s_total_entries = 0;
    }
    __syncthreads();
  }

  // Flush expert counts to local device buffer.
  for (int32_t e = threadIdx.x; e < NL;
       e += blockDim.x) {
    int32_t count = s_expert_counts[e];
    if (count > 0) {
      atomicAdd(&config->local_expert_counts[e],
                count);
    }
  }

  // Make all P2P writes + local_expert_counts visible.
  __syncthreads();
  __threadfence_system();
}

// ---- Push allgather counts + P2P barrier (RESET_COMBINE) ----
// Grid = 1, block = kBlockSize.
__global__ void dar_push_and_barrier_kernel(
    const DispatchCombineConfig* __restrict__ config) {
  const int32_t rank = config->rank;
  const int32_t ws = config->world_size;
  const int32_t NL = config->num_logical_experts;
  const int32_t tid = threadIdx.x;

  // Allgather push: write local counts to all peers.
  for (int32_t e = tid; e < NL; e += blockDim.x) {
    int32_t count = config->local_expert_counts[e];
    for (int32_t r = 0; r < ws; r++) {
      reinterpret_cast<int32_t*>(
          config->remote_expert_counts[r])
              [rank * NL + e] = count;
    }
  }

  // Push dispatch section counts.
  if (tid < ws) {
    int32_t dr = tid;
    config->remote_dispatch_offsets[dr][rank] =
        config->local_dispatch_counters[dr];
  }
  __syncthreads();

  // Reset combine offsets + local counters.
  if (tid < ws) {
    config->remote_combine_offsets[rank][tid] = 0;
    config->local_combine_counters[tid] = 0;
  }

  __threadfence_system();

  // P2P flag exchange.
  FlagType expected =
      config->self_signals->counter + 1;

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
}

// ---- Phase C: Deterministic router ----
// Grid = 1, block = kBlockSize.
// Shared mem: (3*NL + NL*max_rep + ws) * 4 bytes.
__global__ void dar_phase_c_kernel(
    const DispatchCombineConfig* __restrict__ config) {
  const int32_t rank = config->rank;
  const int32_t ws = config->world_size;
  const int32_t NL = config->num_logical_experts;
  const int32_t max_rep = config->max_replicas;
  const int32_t epr = config->physical_experts_per_rank;

  extern __shared__ int32_t shared[];
  int32_t* s_expert_sum = shared;
  int32_t* s_replica_count = shared + NL;
  int32_t* s_l2p_map = shared + 2 * NL;
  int32_t* routing_sel =
      shared + 2 * NL + NL * max_rep;
  int32_t* rank_active = routing_sel + NL;

  // Parallel preload from global to smem.
  int32_t* ec_buf =
      reinterpret_cast<int32_t*>(
          config->remote_expert_counts[rank]);
  for (int32_t e = threadIdx.x; e < NL;
       e += blockDim.x) {
    int32_t sum = 0;
    for (int32_t s = 0; s < ws; s++) {
      sum += ec_buf[s * NL + e];
    }
    s_expert_sum[e] = sum;
  }
  for (int32_t e = threadIdx.x; e < NL;
       e += blockDim.x) {
    s_replica_count[e] = static_cast<int32_t>(
        config->logical_replica_count[e]);
  }
  for (int32_t i = threadIdx.x;
       i < NL * max_rep;
       i += blockDim.x) {
    s_l2p_map[i] =
        config->logical_to_physical_map[i];
  }
  for (int32_t e = threadIdx.x; e < NL;
       e += blockDim.x) {
    routing_sel[e] = -1;
  }
  for (int32_t r = threadIdx.x; r < ws;
       r += blockDim.x) {
    rank_active[r] = 0;
  }
  __syncthreads();

  // Sequential routing (thread 0).
  if (threadIdx.x == 0) {
    for (int32_t e = 0; e < NL; e++) {
      const int32_t count = s_expert_sum[e];
      if (count == 0) continue;

      int32_t rc = s_replica_count[e];
      if (rc <= 0) continue;
      if (rc > max_rep) rc = max_rep;

      if (rc == 1) {
        const int32_t phys =
            s_l2p_map[e * max_rep];
        routing_sel[e] = phys;
        rank_active[phys / epr] += count;
        continue;
      }

      int32_t best_phys = -1;
      int32_t best_rank = -1;
      int32_t best_cost = INT_MAX;
      for (int32_t i = 0; i < rc; i++) {
        const int32_t phys =
            s_l2p_map[e * max_rep + i];
        const int32_t r = phys / epr;
        const int32_t c = rank_active[r];
        if (c < best_cost ||
            (c == best_cost && r < best_rank)) {
          best_cost = c;
          best_rank = r;
          best_phys = phys;
        }
      }
      routing_sel[e] = best_phys;
      rank_active[best_rank] += count;
    }
  }
  __syncthreads();

  // Write routing_selection to global memory.
  // No trailing fence needed: kernel launch boundary
  // to D1/D2 on the same stream provides device-wide
  // visibility of all global memory writes.
  for (int32_t e = threadIdx.x; e < NL;
       e += blockDim.x) {
    config->routing_selection[e] = routing_sel[e];
  }
}

// ---- Phase D1: Zero expert_num_tokens for D2 ----
// D2 uses atomicAdd on expert_num_tokens, so it must
// start at zero. Folded here to avoid host-side
// cudaMemsetAsync overhead. Kernel-to-kernel ordering
// on the same stream guarantees D2 sees the zeros.
__global__ void dar_phase_d1_kernel(
    int32_t* __restrict__ expert_num_tokens,
    int32_t num_physical_experts) {
  for (int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
       i < num_physical_experts;
       i += gridDim.x * blockDim.x) {
    expert_num_tokens[i] = 0;
  }
}

// ---- Phase D2: Single-pass fill + routing filter ----
// Grid = kPersistentGrid, block = kBlockSize.
// For each entry: write sentinel defaults, then check
// if real and overwrite with routing results. Single
// pass avoids cross-block race between fill and routing
// when section boundaries don't align with grid stride.
// expert_num_tokens is zeroed by D1 (split path) or
// inline by Phase C / block 0 barrier (fused paths).
__global__ void dar_phase_d2_kernel(
    int64_t* __restrict__ expert_topk_ids,
    float* __restrict__ expert_topk_weights,
    int32_t* __restrict__ expert_num_tokens,
    int32_t* __restrict__ data_remap,
    const DispatchCombineConfig* __restrict__ config,
    int32_t mc,
    int32_t num_physical_experts) {
  const int32_t rank = config->rank;
  const int32_t ws = config->world_size;
  const int32_t NL = config->num_logical_experts;
  const int32_t epr = config->physical_experts_per_rank;
  const int32_t ss = config->dispatch_section_size;

  const TokenMetadata* meta_r =
      reinterpret_cast<const TokenMetadata*>(
          config->remote_dispatch_meta[rank]);
  TokenMetadata* meta_w =
      reinterpret_cast<TokenMetadata*>(
          config->remote_dispatch_meta[rank]);

  const int64_t sentinel =
      static_cast<int64_t>(num_physical_experts);

  for (int32_t base = blockIdx.x * blockDim.x;
       base < mc;
       base += gridDim.x * blockDim.x) {
    int32_t idx = base + threadIdx.x;
    if (idx < mc) {
      // Write sentinel defaults for ALL entries.
      // Stale entries keep these; real entries
      // overwrite below (same thread, no race).
      expert_topk_ids[idx] = sentinel;
      expert_topk_weights[idx] = 0.0f;
      data_remap[idx] = idx;

      // Section-aware real check. Clamp count to
      // section size: raw counter may exceed ss due
      // to overflow counting in dispatch atomicAdd.
      int32_t sec = idx / ss;
      int32_t off = idx % ss;
      int32_t sec_count = config->
          remote_dispatch_offsets[rank][sec];
      if (sec_count > ss) sec_count = ss;
      if (sec < ws && off < sec_count) {
        // Backward scan for group leader.
        int32_t section_start = sec * ss;
        int32_t leader = idx;
        if (idx > section_start) {
          int32_t sr = meta_r[idx].source_rank;
          int32_t st =
              meta_r[idx].source_token_idx;
          int32_t chk = idx - 1;
          while (chk >= section_start
                 && meta_r[chk].source_rank == sr
                 && meta_r[chk].source_token_idx
                     == st) {
            leader = chk;
            chk--;
          }
        }
        data_remap[idx] = leader;

        // Routing filter.
        const int32_t logical_id =
            meta_r[idx].expert_id;
        if (logical_id < 0
            || logical_id >= NL) {
          meta_w[idx].topk_weight = 0.0f;
        } else {
          const int32_t sel =
              config->routing_selection[
                  logical_id];
          if (sel < 0
              || sel >= num_physical_experts) {
            meta_w[idx].topk_weight = 0.0f;
          } else if (sel / epr == rank) {
            // KEEP: local replica.
            expert_topk_ids[idx] =
                static_cast<int64_t>(sel);
            expert_topk_weights[idx] =
                meta_r[idx].topk_weight;
            atomicAdd(
                &expert_num_tokens[sel], 1);
          } else {
            // FILTER: not our replica.
            meta_w[idx].topk_weight = 0.0f;
          }
        }
      }
    }
  }
}

// ---- Section compaction ----
// Gathers valid entries from scattered per-sender sections
// into contiguous positions. Builds compact_data_remap
// (compact_idx → original leader index for dedup + gather),
// compact_expert_topk_ids, compact_expert_topk_weights,
// and compact_reverse (original_idx → compact_idx for
// combine kernel to read expert output).
// Grid = kPersistentGrid, block = kBlockSize.
__global__ void dar_compact_kernel(
    const int64_t* __restrict__ expert_topk_ids,
    const float* __restrict__ expert_topk_weights,
    const int32_t* __restrict__ data_remap,
    int64_t* __restrict__ compact_expert_topk_ids,
    float* __restrict__ compact_expert_topk_weights,
    int32_t* __restrict__ compact_data_remap,
    int32_t* __restrict__ compact_reverse,
    const DispatchCombineConfig* __restrict__ config,
    int32_t mc_compact,
    int32_t num_physical_experts) {
  const int32_t rank = config->rank;
  const int32_t ws = config->world_size;
  const int32_t ss = config->dispatch_section_size;
  const int64_t sentinel =
      static_cast<int64_t>(num_physical_experts);

  // Compute per-section prefix sums (thread 0 only).
  __shared__ int32_t s_compact_offset[kMaxRanks];
  __shared__ int32_t s_section_count[kMaxRanks];
  __shared__ int32_t s_num_valid;

  if (threadIdx.x == 0) {
    int32_t running = 0;
    for (int32_t s = 0; s < ws; s++) {
      int32_t count = config->
          remote_dispatch_offsets[rank][s];
      if (count > ss) count = ss;
      s_section_count[s] = count;
      s_compact_offset[s] = running;
      running += count;
    }
    s_num_valid = running;
  }
  __syncthreads();

  const int32_t num_valid = s_num_valid;

  // Build compact mappings for valid entries.
  for (int32_t s = 0; s < ws; s++) {
    const int32_t section_start = s * ss;
    const int32_t count = s_section_count[s];
    const int32_t compact_base = s_compact_offset[s];
    for (int32_t i =
             blockIdx.x * blockDim.x + threadIdx.x;
         i < count;
         i += gridDim.x * blockDim.x) {
      const int32_t original_idx = section_start + i;
      const int32_t compact_idx = compact_base + i;

      // Remap data_remap: leader is in same section,
      // so leader_compact = compact_base + (leader -
      // section_start).
      const int32_t leader_original =
          data_remap[original_idx];
      compact_data_remap[compact_idx] =
          compact_base
          + (leader_original - section_start);

      // Copy expert_topk_ids and weights.
      compact_expert_topk_ids[compact_idx] =
          expert_topk_ids[original_idx];
      compact_expert_topk_weights[compact_idx] =
          expert_topk_weights[original_idx];

      // Reverse mapping for combine kernel.
      compact_reverse[original_idx] = compact_idx;
    }
  }

  // Pad entries [num_valid, mc_compact) with sentinel.
  for (int32_t i = num_valid
           + blockIdx.x * blockDim.x + threadIdx.x;
       i < mc_compact;
       i += gridDim.x * blockDim.x) {
    compact_expert_topk_ids[i] = sentinel;
    compact_expert_topk_weights[i] = 0.0f;
    compact_data_remap[i] = 0;  // harmless index
  }
}

// ---- Phase E: Zero counts for next invocation ----
// Grid = kPersistentGrid, block = kBlockSize.
__global__ void dar_phase_e_kernel(
    const DispatchCombineConfig* __restrict__ config) {
  const int32_t rank = config->rank;
  const int32_t ws = config->world_size;
  const int32_t NL = config->num_logical_experts;

  int32_t* remote_ec =
      reinterpret_cast<int32_t*>(
          config->remote_expert_counts[rank]);
  int32_t* local_ec =
      config->local_expert_counts;
  for (int32_t e = blockIdx.x * blockDim.x
           + threadIdx.x;
       e < NL;
       e += gridDim.x * blockDim.x) {
    remote_ec[rank * NL + e] = 0;
    local_ec[e] = 0;
  }
  if (blockIdx.x == 0 && threadIdx.x < ws) {
    config->local_dispatch_counters[threadIdx.x] = 0;
  }
}

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
  // Pass 1/2: s_expert_counts[NL] + s_dr_count[ws]
  //           + s_ent_lid[64] + s_ent_wt[64]
  //           + s_ent_dr[64]
  // Phase C:  s_expert_sum[NL] + s_replica_count[NL]
  //           + s_l2p_map[NL*max_rep]
  //           + routing_selection_smem[NL]
  //           + rank_active_counts[ws]
  // Phases don't overlap, so same memory is reused.
  extern __shared__ int32_t shared[];

  // Read monotonic counter base values BEFORE any phase
  // (for CUDA graph replay compatibility).
  FlagType rf_expected = 0;
  FlagType pa_base = 0;
  FlagType prf_expected = 0;
  FlagType barrier_expected = 0;
  if (threadIdx.x == 0) {
    rf_expected =
        static_cast<FlagType>(
            *config->routing_ready_flag) + 1;
    pa_base =
        static_cast<FlagType>(
            *config->phase_a_done_counter);
    prf_expected =
        static_cast<FlagType>(
            *config->positions_ready_flag) + 1;
    barrier_expected =
        config->self_signals->counter + 1;
  }
  // Broadcast to all threads via shared mem.
  __shared__ FlagType s_rf_expected;
  __shared__ FlagType s_pa_base;
  __shared__ FlagType s_prf_expected;
  __shared__ FlagType s_barrier_expected;
  if (threadIdx.x == 0) {
    s_rf_expected = rf_expected;
    s_pa_base = pa_base;
    s_prf_expected = prf_expected;
    s_barrier_expected = barrier_expected;
  }
  __syncthreads();
  rf_expected = s_rf_expected;
  pa_base = s_pa_base;
  prf_expected = s_prf_expected;
  barrier_expected = s_barrier_expected;

  // ============================================================
  // PASS 1: Entry counting (no NVLink writes)
  // ============================================================
  // Count entries per dest_rank, accumulate expert counts.
  // No serial grouping, no device atomicAdd for positions.
  int32_t* s_expert_counts = shared;       // [NL]
  int32_t* s_dr_count = shared + NL;       // [ws]

  for (int32_t e = threadIdx.x; e < NL;
       e += blockDim.x) {
    s_expert_counts[e] = 0;
  }
  if (threadIdx.x < ws) {
    s_dr_count[threadIdx.x] = 0;
  }
  __syncthreads();

  for (int32_t t = blockIdx.x; t < M; t += gridDim.x) {
    if (threadIdx.x < topk) {
      int32_t slot = threadIdx.x;
      int32_t lid = topk_ids[t * topk + slot];
      if (lid >= 0 && lid < NL) {
        atomicAdd(&s_expert_counts[lid], 1);
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
          atomicAdd(&s_dr_count[dr], 1);
        }
      }
    }
    // No __syncthreads between tokens: smem atomics
    // accumulate across the entire token loop.
  }

  // Flush expert counts to local device buffer.
  __syncthreads();
  for (int32_t e = threadIdx.x; e < NL;
       e += blockDim.x) {
    int32_t count = s_expert_counts[e];
    if (count > 0) {
      atomicAdd(&config->local_expert_counts[e],
                count);
    }
  }
  // Write per-block per-rank counts to device buffer.
  if (threadIdx.x < ws) {
    config->block_dispatch_counts[
        blockIdx.x * kMaxRanks + threadIdx.x] =
            s_dr_count[threadIdx.x];
  }

  // ============================================================
  // AGGREGATION: Grid sync + block 0 prefix sum + count push
  // ============================================================
  __syncthreads();
  __threadfence();  // device fence (no NVLink writes yet)
  if (threadIdx.x == 0) {
    atomicAdd(config->phase_a_done_counter,
              static_cast<FlagType>(1));
  }

  if (blockIdx.x == 0) {
    // Wait for all blocks to finish Pass 1.
    if (threadIdx.x == 0) {
      FlagType target = pa_base + gridDim.x;
      while (dc_ld_flag_acquire(
                 config->phase_a_done_counter)
              < target)
        ;
    }
    __syncthreads();

    // Prefix sum: per-block starting positions.
    // Thread tid handles dest_rank tid.
    if (threadIdx.x < ws) {
      int32_t dr = threadIdx.x;
      int32_t ss = config->dispatch_section_size;
      int32_t running = 0;
      for (int32_t b = 0; b < gridDim.x; b++) {
        int32_t c = config->block_dispatch_counts[
            b * kMaxRanks + dr];
        // Section overflow: clamp.
        int32_t pos = (running < ss)
            ? rank * ss + running
            : config->max_recv;
        config->block_start_positions[
            b * kMaxRanks + dr] = pos;
        running += c;
      }
      // Push total dispatch count to remote rank
      // (fire-and-forget NVLink write).
      config->remote_dispatch_offsets[dr][rank] =
          running;
    }

    // Push expert counts to all remote ranks
    // (fire-and-forget NVLink writes).
    for (int32_t e = threadIdx.x; e < NL;
         e += blockDim.x) {
      int32_t count =
          config->local_expert_counts[e];
      for (int32_t r = 0; r < ws; r++) {
        reinterpret_cast<int32_t*>(
            config->remote_expert_counts[r])
                [rank * NL + e] = count;
      }
    }

    // Fence positions (device memory) and signal ready.
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
      dc_st_flag_release(
          config->positions_ready_flag,
          prf_expected);
    }
  } else {
    // Blocks 1-31: wait for positions to be ready.
    if (threadIdx.x == 0) {
      while (dc_ld_flag_acquire(
                 config->positions_ready_flag)
              != prf_expected)
        ;
    }
    __syncthreads();
    __threadfence();
  }

  // ============================================================
  // PASS 2: Fire-and-forget NVLink writes
  // ============================================================
  // Re-expand entries using pre-computed positions.
  // Data written ONCE per unique (token, dest_rank).
  // No serial grouping, no device atomicAdd.
  {
    constexpr int32_t kMaxEntries = 64;
    // Reuse shared[] for per-token entry tracking.
    int32_t* s_ent_lid = shared;                  // [kMaxEntries]
    float*   s_ent_wt  = reinterpret_cast<float*>(
        s_ent_lid + kMaxEntries);                 // [kMaxEntries]
    int32_t* s_ent_dr  = reinterpret_cast<int32_t*>(
        s_ent_wt + kMaxEntries);                  // [kMaxEntries]

    __shared__ int32_t s_total_entries;
    __shared__ int32_t s_pos[kMaxRanks];
    __shared__ int32_t s_dr_count_t[kMaxRanks];
    __shared__ int32_t s_first_pos[kMaxRanks];

    // Init per-rank position counters from aggregation.
    if (threadIdx.x < ws) {
      s_pos[threadIdx.x] =
          config->block_start_positions[
              blockIdx.x * kMaxRanks + threadIdx.x];
    }
    if (threadIdx.x == 0) {
      s_total_entries = 0;
    }
    __syncthreads();

    for (int32_t t = blockIdx.x; t < M;
         t += gridDim.x) {
      // Snapshot positions before expansion.
      if (threadIdx.x < ws) {
        s_first_pos[threadIdx.x] =
            s_pos[threadIdx.x];
        s_dr_count_t[threadIdx.x] = 0;
      }
      if (threadIdx.x == 0) {
        s_total_entries = 0;
      }
      __syncthreads();

      // Expand entries (L2-cached re-reads).
      if (threadIdx.x < topk) {
        int32_t slot = threadIdx.x;
        int32_t lid = topk_ids[t * topk + slot];
        if (lid >= 0 && lid < NL) {
          float wt = topk_weights[t * topk + slot];
          int32_t rc = static_cast<int32_t>(
              config->logical_replica_count[lid]);
          if (rc > max_rep) rc = max_rep;
          for (int32_t rep = 0; rep < rc; rep++) {
            int32_t phys =
                config->logical_to_physical_map[
                    lid * max_rep + rep];
            int32_t dr = phys / epr;
            if (dr < 0 || dr >= ws) continue;
            if (!config->remote_dispatch_offsets[dr]
                || !config->remote_dispatch_recv[dr]
                || !config->remote_dispatch_meta[dr])
              continue;
            int32_t ei =
                atomicAdd(&s_total_entries, 1);
            if (ei < kMaxEntries) {
              s_ent_lid[ei] = lid;
              s_ent_wt[ei] = wt;
              s_ent_dr[ei] = dr;
            }
            atomicAdd(&s_pos[dr], 1);
            atomicAdd(&s_dr_count_t[dr], 1);
          }
        }
      }
      __syncthreads();

      // Write data ONCE + metadata per dest_rank.
      int32_t ne = s_total_entries;
      if (ne > kMaxEntries) ne = kMaxEntries;
      for (int32_t dr = 0; dr < ws; dr++) {
        int32_t n = s_dr_count_t[dr];
        if (n == 0) continue;
        int32_t fpos = s_first_pos[dr];
        int32_t ss = config->dispatch_section_size;
        if (fpos >= rank * ss + ss) continue;

        // All threads: write token data at fpos
        // (fire-and-forget NVLink store).
        T* dest = reinterpret_cast<T*>(
            config->remote_dispatch_recv[dr]);
        const T* src = input + t * K;
        for (int32_t k = threadIdx.x; k < K;
             k += blockDim.x) {
          dest[fpos * K + k] = src[k];
        }

        // Thread 0: write metadata per entry.
        if (threadIdx.x == 0) {
          TokenMetadata* meta =
              reinterpret_cast<TokenMetadata*>(
                  config->remote_dispatch_meta[dr]);
          int32_t mi = 0;
          for (int32_t ei = 0;
               ei < ne && mi < n; ei++) {
            if (s_ent_dr[ei] != dr) continue;
            int32_t pos = fpos + mi;
            if (pos >= rank * ss + ss) break;
            meta[pos].source_rank = rank;
            meta[pos].source_token_idx = t;
            meta[pos].expert_id = s_ent_lid[ei];
            meta[pos].topk_weight = s_ent_wt[ei];
            mi++;
          }
        }
      }
      __syncthreads();
    }
  }

  // ============================================================
  // SINGLE BARRIER: fence + grid sync + P2P barrier
  // ============================================================
  // Covers ALL NVLink writes: data, metadata, expert
  // counts, dispatch offsets. Staggered fence: blocks
  // finish Pass 2 at different times.
  __syncthreads();
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    atomicAdd(config->phase_a_done_counter,
              static_cast<FlagType>(1));
  }

  if (blockIdx.x == 0) {
    // Wait for all blocks to finish Pass 2 + fence.
    if (threadIdx.x == 0) {
      FlagType target = pa_base + 2 * gridDim.x;
      while (dc_ld_flag_acquire(
                 config->phase_a_done_counter)
              < target)
        ;
    }
    __syncthreads();

    const int32_t tid = threadIdx.x;

    // Reset combine state for next layer.
    if (tid < ws) {
      config->remote_combine_offsets[rank][tid] = 0;
      config->local_combine_counters[tid] = 0;
    }

    // P2P barrier: dc_st_flag_release orders each
    // thread's prior reset writes via release semantics.
    // All NVLink data writes already system-visible
    // via __threadfence_system() + grid sync above.
    if (tid < ws) {
      dc_st_flag_release(
          &config->peer_signals[tid]->flags[rank],
          barrier_expected);
      while (dc_ld_flag_acquire(
          &config->self_signals->flags[tid])
              != barrier_expected)
        ;
    }

    __syncthreads();

    if (tid == 0) {
      dc_st_flag_release(
          &config->self_signals->counter,
          barrier_expected);
    }
  } else {
    if (threadIdx.x == 0) {
      while (dc_ld_flag_acquire(
          &config->self_signals->counter)
              != barrier_expected)
        ;
    }
    __syncthreads();
  }

  // Sum per-sender dispatch counts (available after
  // barrier). Each sender pushed its section count
  // into remote_dispatch_offsets[rank][sender].
  int32_t actual = 0;
  {
    int32_t ss = config->dispatch_section_size;
    for (int32_t s = 0; s < ws; s++) {
      int32_t c = config->remote_dispatch_offsets[rank][s];
      if (c > ss) c = ss;
      actual += c;
    }
    if (actual > config->max_recv)
      actual = config->max_recv;
  }

  // ---- Phase C + D1 (block 0 only) ----
  // Block 0: parallel preload + deterministic router +
  // zero expert_num_tokens. Other blocks spin-wait on
  // routing_ready_flag.
  if (blockIdx.x == 0) {
    // Reuse shared[] for Phase C preload layout.
    int32_t* s_expert_sum = shared;           // [NL]
    int32_t* s_replica_count = shared + NL;   // [NL]
    int32_t* s_l2p_map = shared + 2 * NL;    // [NL*mr]
    int32_t* routing_sel =
        shared + 2 * NL + NL * max_rep;       // [NL]
    int32_t* rank_active = routing_sel + NL;   // [ws]

    // All threads: parallel preload from global to smem.
    // Sum expert counts across allgather sections.
    int32_t* ec_buf =
        reinterpret_cast<int32_t*>(
            config->remote_expert_counts[rank]);
    for (int32_t e = threadIdx.x; e < NL;
         e += blockDim.x) {
      int32_t sum = 0;
      for (int32_t s = 0; s < ws; s++) {
        sum += ec_buf[s * NL + e];
      }
      s_expert_sum[e] = sum;
    }
    for (int32_t e = threadIdx.x; e < NL;
         e += blockDim.x) {
      s_replica_count[e] = static_cast<int32_t>(
          config->logical_replica_count[e]);
    }
    for (int32_t i = threadIdx.x;
         i < NL * max_rep;
         i += blockDim.x) {
      s_l2p_map[i] =
          config->logical_to_physical_map[i];
    }
    for (int32_t e = threadIdx.x; e < NL;
         e += blockDim.x) {
      routing_sel[e] = -1;
    }
    for (int32_t r = threadIdx.x; r < ws;
         r += blockDim.x) {
      rank_active[r] = 0;
    }
    __syncthreads();

    // Sequential routing (thread 0 only).
    // All reads from shared memory (~1 cycle each).
    if (threadIdx.x == 0) {
      for (int32_t e = 0; e < NL; e++) {
        const int32_t count = s_expert_sum[e];
        if (count == 0) continue;

        int32_t rc = s_replica_count[e];
        if (rc <= 0) continue;
        if (rc > max_rep) rc = max_rep;

        if (rc == 1) {
          const int32_t phys =
              s_l2p_map[e * max_rep];
          routing_sel[e] = phys;
          rank_active[phys / epr] += count;
          continue;
        }

        // Multiple replicas: pick minimum-loaded rank
        // (ties: lower rank for determinism).
        int32_t best_phys = -1;
        int32_t best_rank = -1;
        int32_t best_cost = INT_MAX;
        for (int32_t i = 0; i < rc; i++) {
          const int32_t phys =
              s_l2p_map[e * max_rep + i];
          const int32_t r = phys / epr;
          const int32_t c = rank_active[r];
          if (c < best_cost ||
              (c == best_cost && r < best_rank)) {
            best_cost = c;
            best_rank = r;
            best_phys = phys;
          }
        }
        routing_sel[e] = best_phys;
        rank_active[best_rank] += count;
      }
    }
    __syncthreads();

    // Write routing_selection to global memory.
    for (int32_t e = threadIdx.x; e < NL;
         e += blockDim.x) {
      config->routing_selection[e] = routing_sel[e];
    }

    // Zero expert_num_tokens for Phase D2's atomicAdd.
    for (int32_t i = threadIdx.x;
         i < num_physical_experts;
         i += blockDim.x) {
      expert_num_tokens[i] = 0;
    }
    __threadfence();
    __syncthreads();

    // Signal routing complete.
    if (threadIdx.x == 0) {
      dc_st_flag_release(
          config->routing_ready_flag, rf_expected);
    }
  } else {
    // Wait for routing to complete (blocks 1-31).
    if (threadIdx.x == 0) {
      while (dc_ld_flag_acquire(
              config->routing_ready_flag)
              != rf_expected)
        ;
    }
    __syncthreads();
    __threadfence();
  }

  // ---- Phase D2: Single-pass fill + routing filter ----
  // For each entry: write sentinel defaults, then check
  // if real and overwrite. Single pass ensures no cross-
  // block race between fill and routing.
  {
    const TokenMetadata* meta_r =
        reinterpret_cast<const TokenMetadata*>(
            config->remote_dispatch_meta[rank]);
    TokenMetadata* meta_w =
        reinterpret_cast<TokenMetadata*>(
            config->remote_dispatch_meta[rank]);
    const int32_t ss_d2 =
        config->dispatch_section_size;
    const int64_t d2_sentinel =
        static_cast<int64_t>(num_physical_experts);

    for (int32_t base = blockIdx.x * blockDim.x;
         base < mc;
         base += gridDim.x * blockDim.x) {
      int32_t idx = base + threadIdx.x;
      if (idx < mc) {
        // Write sentinel defaults for ALL entries.
        expert_topk_ids[idx] = d2_sentinel;
        expert_topk_weights[idx] = 0.0f;
        data_remap[idx] = idx;

        // Section-aware real check. Clamp count to
        // section size: raw counter may exceed ss_d2.
        int32_t sec = idx / ss_d2;
        int32_t off = idx % ss_d2;
        int32_t sec_cnt = config->
            remote_dispatch_offsets[rank][sec];
        if (sec_cnt > ss_d2) sec_cnt = ss_d2;
        if (sec < ws && off < sec_cnt) {
          // Backward scan for group leader.
          int32_t section_start = sec * ss_d2;
          int32_t leader = idx;
          if (idx > section_start) {
            int32_t sr = meta_r[idx].source_rank;
            int32_t st =
                meta_r[idx].source_token_idx;
            int32_t chk = idx - 1;
            while (chk >= section_start
                   && meta_r[chk].source_rank == sr
                   && meta_r[chk].source_token_idx
                       == st) {
              leader = chk;
              chk--;
            }
          }
          data_remap[idx] = leader;

          // Routing filter.
          const int32_t logical_id =
              meta_r[idx].expert_id;
          if (logical_id < 0
              || logical_id >= NL) {
            meta_w[idx].topk_weight = 0.0f;
          } else {
            const int32_t sel =
                config->routing_selection[
                    logical_id];
            if (sel < 0
                || sel >= num_physical_experts) {
              meta_w[idx].topk_weight = 0.0f;
            } else if (sel / epr == rank) {
              // KEEP: local replica.
              expert_topk_ids[idx] =
                  static_cast<int64_t>(sel);
              expert_topk_weights[idx] =
                  meta_r[idx].topk_weight;
              atomicAdd(
                  &expert_num_tokens[sel], 1);
            } else {
              // FILTER: not our replica.
              meta_w[idx].topk_weight = 0.0f;
            }
          }
        }
      }
    }
  }

  // ---- Phase E: Zero counts for next invocation ----
  // Zero this rank's allgather section + local counts
  // + local dispatch counters. Other ranks' sections
  // are zeroed by their owners.
  // The next layer's combine barrier (RESET_DISPATCH)
  // includes __threadfence_system() which ensures this
  // zeroing is visible to all peers before they start
  // the next dispatch_and_route.
  {
    int32_t* remote_ec =
        reinterpret_cast<int32_t*>(
            config->remote_expert_counts[rank]);
    int32_t* local_ec =
        config->local_expert_counts;
    for (int32_t e = blockIdx.x * blockDim.x
             + threadIdx.x;
         e < NL;
         e += gridDim.x * blockDim.x) {
      remote_ec[rank * NL + e] = 0;
      local_ec[e] = 0;
    }
    // Zero local dispatch counters for next invocation.
    if (blockIdx.x == 0 && threadIdx.x < ws) {
      config->local_dispatch_counters[threadIdx.x] = 0;
    }
  }
}

}  // namespace dispatch_combine
}  // namespace vllm
