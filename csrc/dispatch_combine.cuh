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
constexpr int kPersistentGrid = 108;

// Max unique (dest_rank, token) groups per tile in CAS
// Phase 1 accumulation. Dynamic shared memory sized to
// kCasMaxUnique * K * sizeof(float). If nu exceeds this,
// tiled accumulation processes groups in batches of
// kCasMaxUnique (ceil(nu/kCasMaxUnique) tiles).
// A100: 164KB smem max. 11 × 3584 × 4 = 154KB + ~4KB
// static = ~158KB < 164KB.
constexpr int kCasMaxUnique = 11;

// Decode path uses fewer accumulators for lower smem
// (5 × 3584 × 4 = 70KB → 2 blocks/SM on A100).
constexpr int kCasMaxUniqueDecode = 5;

// M threshold: decode path (direct HBM scan, small smem)
// vs prefill path (cooperative loading, large smem).
constexpr int kCasDecodeThreshold = 256;

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
  // once per invocation (after fused scan+write + fence).
  // Block 0 spins until counter == base + gridDim.x.
  // Monotonic for CUDA graph replay.
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

  // Grid-wide sync for fused scatter-add → fp32_to_half.
  // All blocks increment after Phase 3 (scatter-add);
  // all blocks spin until counter reaches target before
  // Phase 4 conversion. Monotonic for CUDA graph replay.
  FlagType* scatter_done_counter;

  // ---- Fine-grained profiling support ----
  // When non-null, block 0 writes globaltimer timestamps
  // at each step boundary. Layout:
  //   [0..kDarNumSteps-1] for dispatch_and_route_kernel
  //   [kDarNumSteps..kDarNumSteps+kCasNumSteps-1] for
  //     combine_and_scatter_kernel
  // Host reads back and accumulates averages.
  int64_t* profiling_timestamps;  // null = disabled
};

// Number of timestamp slots per kernel.
constexpr int kDarNumSteps = 19;
constexpr int kCasNumSteps = 18;
constexpr int kTotalProfileSlots =
    kDarNumSteps + kCasNumSteps;

// Step names (host-side, for printing).
inline const char* dar_step_name(int i) {
  static const char* names[] = {
    "dar:read_counters",     // 0
    "dar:scan_write",        // 1
    "dar:scan_expand",       // 2  Step 1: topk reads
    "dar:scan_group",        // 3  (fused into Step 1)
    "dar:scan_claim",        // 4  Step 3: atomicAdd
    "dar:scan_nvlink",       // 5  Step 4: NVLink write
    "dar:expert_flush",      // 6
    "dar:threadfence_sys",   // 7
    "dar:grid_sync",         // 8
    "dar:expert_push",       // 9
    "dar:fence2",            // 10 fence#2 drain
    "dar:p2p_wait",          // 11 P2P flag exchange
    "dar:phase_c_preload",   // 12 smem preload
    "dar:phase_c_route",     // 13 routing start
    "dar:route_pass1",       // 14 parallel rc==1
    "dar:route_pass2",       // 15 sequential rc>1
    "dar:route_writeback",   // 16 write+zero+fence
    "dar:phase_d2_filter",   // 17
    "dar:end",               // 18
  };
  return (i < kDarNumSteps) ? names[i] : "dar:?";
}

inline const char* cas_step_name(int i) {
  static const char* names[] = {
    "cas:read_counters",     // 0
    "cas:zero_accum",        // 1
    "cas:coop_load",         // 2  cooperative HBM load
    "cas:sw_scan",           // 3  thread-0 scan+sort done
    "cas:sw_zero",           // 4  accum zeroed (fast path)
    "cas:sw_accum",          // 5  accumulation (fast path)
    "cas:tile_zero",         // 6  tiled: zero done
    "cas:tile_accum",        // 7  tiled: accumulate done
    "cas:tile_nvlink",       // 8  tiled: NVLink write done
    "cas:tile_done",         // 9  all tiles done
    "cas:grid_sync",         // 10 grid-wide sync
    "cas:offset_push",       // 11 NVLink stores
    "cas:fence2",            // 12 fence#2 drain
    "cas:p2p_wait",          // 13 P2P flag exchange
    "cas:scatter_add",       // 14 scatter-add
    "cas:convert",           // 15 fp32→half conversion
    "cas:end",               // 16
    "cas:?",                 // 17 unused
  };
  return (i < kCasNumSteps) ? names[i] : "cas:?";
}

// Helper: read globaltimer (synchronized across SMs).
static __device__ __forceinline__ int64_t
dc_globaltimer() {
  int64_t ts;
  asm volatile("mov.u64 %0, %%globaltimer;"
               : "=l"(ts));
  return ts;
}

// Helper: record timestamp if profiling enabled.
// Only block 0, thread 0 writes to avoid contention.
#define DC_TIMESTAMP(config, slot)                   \
  do {                                               \
    if (blockIdx.x == 0 && threadIdx.x == 0 &&       \
        (config)->profiling_timestamps) {             \
      (config)->profiling_timestamps[(slot)] =        \
          dc_globaltimer();                           \
    }                                                 \
  } while (0)

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
// Dispatch P2P kernel (dedup: data sent once per group)
// ====================================================================
// Persistent-grid per-token dispatch: groups topk entries by
// dest_rank, claims contiguous positions per group, writes
// activation data ONCE per (token, dest_rank) group over
// NVLink. Metadata written per entry. Same pattern as
// dispatch_and_route_kernel Phase A but without routing.
// Grid = kPersistentGrid, block = kBlockSize.
// Requires dynamic shared memory: ws*4 + 3*kMaxEntries*4.
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
  const int32_t ws = config->world_size;
  const int32_t epr = config->experts_per_rank;

  constexpr int32_t kMaxEntries = 64;
  extern __shared__ int32_t shared[];
  int32_t* s_grp_count = shared;              // [ws]
  int32_t* s_ent_eid   = shared + ws;         // [kME]
  float*   s_ent_wt    = reinterpret_cast<float*>(
      s_ent_eid + kMaxEntries);                // [kME]
  int32_t* s_ent_grp   = reinterpret_cast<int32_t*>(
      s_ent_wt + kMaxEntries);                 // [kME]
  __shared__ int32_t s_total_entries;
  __shared__ int32_t s_grp_base[kMaxRanks];

  // Init per-token state.
  if (threadIdx.x < ws)
    s_grp_count[threadIdx.x] = 0;
  if (threadIdx.x == 0) s_total_entries = 0;
  __syncthreads();

  for (int32_t t = blockIdx.x; t < M;
       t += gridDim.x) {
    // Step 1: threads 0..topk-1 expand entries,
    // group by dest_rank.
    if (threadIdx.x < topk) {
      int32_t eid =
          topk_ids[t * topk + threadIdx.x];
      if (eid >= 0) {
        int32_t dr = eid / epr;
        if (dr >= 0 && dr < ws
            && config->remote_dispatch_recv[dr]
            && config->remote_dispatch_meta[dr]
            && config->
                   remote_dispatch_offsets[dr]) {
          float wt = topk_weights[
              t * topk + threadIdx.x];
          int32_t ei =
              atomicAdd(&s_total_entries, 1);
          if (ei < kMaxEntries) {
            s_ent_eid[ei] = eid;
            s_ent_wt[ei] = wt;
            s_ent_grp[ei] = dr;
            atomicAdd(&s_grp_count[dr], 1);
          }
        }
      }
    }
    __syncthreads();

    // Step 2: Claim contiguous positions per group.
    if (threadIdx.x < ws
        && s_grp_count[threadIdx.x] > 0) {
      int32_t local_off = atomicAdd(
          &config->local_dispatch_counters[
              threadIdx.x],
          s_grp_count[threadIdx.x]);
      int32_t ss = config->dispatch_section_size;
      s_grp_base[threadIdx.x] =
          (local_off < ss)
          ? rank * ss + local_off
          : config->max_recv;
    }
    __syncthreads();

    // Step 3: Data copy ONCE per group + metadata.
    // int4 = 16 bytes → 8x fewer stores than bf16.
    const int32_t K4 = K *
        static_cast<int32_t>(sizeof(T)) /
        static_cast<int32_t>(sizeof(int4));
    for (int32_t g = 0; g < ws; g++) {
      if (s_grp_count[g] == 0) continue;
      int32_t base = s_grp_base[g];
      int32_t n = s_grp_count[g];
      if (base >= config->max_recv) continue;
      if (base + n > config->max_recv)
        n = config->max_recv - base;

      // Vectorized data copy (int4).
      // Data position: deterministic from (rank, t).
      // Each token written once per dest (dedup).
      int32_t data_pos =
          rank * config->max_num_tokens_per_rank
          + t;
      const int4* src4 =
          reinterpret_cast<const int4*>(
              input + t * K);
      int4* dest4 = reinterpret_cast<int4*>(
          reinterpret_cast<T*>(
              config->remote_dispatch_recv[g])
          + data_pos * K);
      for (int32_t i = threadIdx.x; i < K4;
           i += blockDim.x) {
        dest4[i] = src4[i];
      }

      // Thread 0: write metadata per entry.
      if (threadIdx.x == 0) {
        TokenMetadata* meta =
            reinterpret_cast<TokenMetadata*>(
                config->remote_dispatch_meta[g]);
        int32_t ne2 = s_total_entries;
        if (ne2 > kMaxEntries) ne2 = kMaxEntries;
        int32_t mi = 0;
        for (int32_t ei = 0;
             ei < ne2 && mi < n; ei++) {
          if (s_ent_grp[ei] != g) continue;
          meta[base + mi].source_rank = rank;
          meta[base + mi].source_token_idx = t;
          meta[base + mi].expert_id =
              s_ent_eid[ei];
          meta[base + mi].topk_weight =
              s_ent_wt[ei];
          mi++;
        }
      }
    }

    // Reset per-token state for next iteration.
    if (threadIdx.x < ws)
      s_grp_count[threadIdx.x] = 0;
    if (threadIdx.x == 0) s_total_entries = 0;
    __syncthreads();
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
        int32_t max_c = ss * config->world_size;
        s_write_pos = (local_off < ss)
            ? config->rank * ss + local_off
            : max_c;
      }
      __syncthreads();

      const int32_t write_pos = s_write_pos;
      if (write_pos >= config->combine_section_size
          * config->world_size) continue;

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
// + routing extraction + data_remap computation into one
// kernel. Block 0 does the cross-GPU barrier; other blocks
// spin on the counter.
// Grid = kPersistentGrid, block = kBlockSize.
// expert_num_tokens is zeroed inline by block 0 after
// barrier, before signaling other blocks.
// data_remap: maps each entry to its group leader (first
// entry with same source_rank + source_token_idx within
// the section). Enables dispatch_p2p dedup: data is only
// at the leader position, other entries share it.
template <typename T>
__global__ void prepare_dispatch_recv_kernel(
    T* __restrict__ dispatch_recv,
    int64_t* __restrict__ expert_topk_ids,
    float* __restrict__ expert_topk_weights,
    int32_t* __restrict__ expert_num_tokens,
    int32_t* __restrict__ data_remap,
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

  // Phase 2: Stamp/zero + routing (thread-parallel).
  // Section-aware: each sender owns a section of size
  // dispatch_section_size. Entry is real if its offset
  // within its section < that section's count.
  // All threads participate (thread-stride) for 256x
  // throughput vs old block-stride/thread-0-only.
  const int32_t rank = config->rank;
  const int32_t ws_p = config->world_size;
  const int32_t ss_p = config->dispatch_section_size;
  const TokenMetadata* meta_p2 =
      reinterpret_cast<const TokenMetadata*>(
          config->remote_dispatch_meta[rank]);

  for (int32_t idx = blockIdx.x * blockDim.x
           + threadIdx.x;
       idx < mc;
       idx += gridDim.x * blockDim.x) {
    int32_t sec = idx / ss_p;
    int32_t off = idx % ss_p;
    int32_t sec_cnt_p = config->
        remote_dispatch_offsets[rank][sec];
    if (sec_cnt_p > ss_p) sec_cnt_p = ss_p;
    if (sec < ws_p && off < sec_cnt_p) {
      int32_t eid = meta_p2[idx].expert_id;
      expert_topk_ids[idx] =
          static_cast<int64_t>(eid);
      expert_topk_weights[idx] = 1.0f;
      if (eid >= 0 && eid < num_experts) {
        atomicAdd(&expert_num_tokens[eid], 1);
      }
    } else {
      expert_topk_ids[idx] =
          static_cast<int64_t>(num_experts);
      expert_topk_weights[idx] = 0.0f;
    }
  }

  // Phase 3: Compute data_remap (compact data position).
  // Data is at sender_rank * max_num_tokens_per_rank +
  // source_token_idx (deterministic, no scan needed).
  for (int32_t idx = blockIdx.x * blockDim.x
           + threadIdx.x;
       idx < mc;
       idx += gridDim.x * blockDim.x) {
    int32_t sec = idx / ss_p;
    int32_t off = idx % ss_p;
    int32_t sec_cnt_dr = config->
        remote_dispatch_offsets[rank][sec];
    if (sec_cnt_dr > ss_p) sec_cnt_dr = ss_p;
    if (sec < ws_p && off < sec_cnt_dr) {
      data_remap[idx] =
          meta_p2[idx].source_rank
          * config->max_num_tokens_per_rank
          + meta_p2[idx].source_token_idx;
    } else {
      data_remap[idx] = 0;  // Harmless for stale
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
    T* __restrict__ output,
    const DispatchCombineConfig* __restrict__ config,
    int32_t mc, int32_t K, int32_t M) {
  const int32_t rank = config->rank;
  const int32_t ws = config->world_size;

  DC_TIMESTAMP(config, kDarNumSteps + 0);  // cas:read_counters

  // Read monotonic counter bases BEFORE any phase
  // modifies them (CUDA graph replay compatible).
  __shared__ FlagType s_cd_base;
  __shared__ FlagType s_sd_base;
  __shared__ FlagType s_barrier_expected;
  if (threadIdx.x == 0) {
    s_cd_base = static_cast<FlagType>(
        *config->combine_done_counter);
    s_sd_base = static_cast<FlagType>(
        *config->scatter_done_counter);
    s_barrier_expected =
        config->self_signals->counter + 1;
  }
  __syncthreads();
  FlagType cd_base = s_cd_base;
  FlagType sd_base = s_sd_base;
  FlagType barrier_expected = s_barrier_expected;

  // ---- Phase 0: Zero fp32 accum buffer ----
  DC_TIMESTAMP(config, kDarNumSteps + 1);  // cas:zero_accum

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

  // ---- Phase 1: Hybrid one-pass combine P2P writes ----
  // Cooperative HBM loading + thread-0 dedup/sort +
  // tiled accumulation with sorted entry ranges.
  DC_TIMESTAMP(config, kDarNumSteps + 2);  // cas:coop_load

  // Dynamic shared memory: fp32 accumulator [K].
  extern __shared__ char dyn_shared_raw[];
  float* s_accum = reinterpret_cast<float*>(
      dyn_shared_raw);

  constexpr int32_t kMaxBatch = 128;
  __shared__ int32_t s_batch_ci[kMaxBatch];
  __shared__ float   s_batch_weight[kMaxBatch];
  __shared__ int32_t s_batch_uid[kMaxBatch];
  __shared__ int32_t s_unique_dr[kMaxBatch];
  __shared__ int32_t s_unique_token[kMaxBatch];
  __shared__ int32_t s_unique_wpos[kMaxBatch];
  // Per-uid start indices for sorted entry access
  // in tiled path. s_uid_start[u] = first entry
  // index for uid u in sorted batch arrays.
  __shared__ int32_t s_uid_start[kMaxBatch + 1];
  __shared__ int32_t s_num_valid;
  __shared__ int32_t s_num_unique;

  // Opt 2: Cooperative loading buffers.
  // Pre-load dispatch_meta + compact_reverse into smem
  // so thread-0 scan reads smem (0-cycle) not HBM.
  __shared__ TokenMetadata s_entries[kMaxBatch];
  __shared__ int32_t s_cr[kMaxBatch];
  // Per-section load ranges computed by thread 0.
  __shared__ int32_t s_load_sec_start[kMaxRanks];
  __shared__ int32_t s_load_sec_count[kMaxRanks];
  __shared__ int32_t s_load_sec_pair_base[kMaxRanks];
  __shared__ int32_t s_total_load;

  if (blockIdx.x < kPersistentGrid) {
    const int32_t ss_d =
        config->dispatch_section_size;
    const int32_t ss_c =
        config->combine_section_size;

    if (M <= kCasDecodeThreshold) {
    // ============================================================
    // DECODE PATH: Direct HBM scan, kCasMaxUniqueDecode=5.
    // Lower smem (70KB) → 2 blocks/SM occupancy on A100.
    // ============================================================

    // Thread-0 reads dispatch_meta directly from HBM
    // (no cooperative loading — overhead not worth it
    // for ~1-2 entries per block at decode batch sizes).
    if (threadIdx.x == 0) {
      s_num_valid = 0;
      s_num_unique = 0;
      int32_t nv = 0;
      int32_t nu = 0;
      for (int32_t s = 0; s < ws; s++) {
        int32_t section_start = s * ss_d;
        int32_t count = config->
            remote_dispatch_offsets[rank][s];
        if (count > ss_d) count = ss_d;
        int32_t chunk =
            (count + kPersistentGrid - 1)
            / kPersistentGrid;
        int32_t my_start = blockIdx.x * chunk;
        int32_t my_end = my_start + chunk;
        if (my_end > count) my_end = count;

        for (int32_t i = my_start;
             i < my_end; i++) {
          int32_t pair_idx = section_start + i;
          float weight =
              dispatch_meta[pair_idx].topk_weight;
          if (weight == 0.0f) continue;
          int32_t dr =
              dispatch_meta[pair_idx].source_rank;
          if (dr < 0 || dr >= ws) continue;
          if (!config->
                  remote_combine_offsets[dr] ||
              !config->
                  remote_combine_recv[dr] ||
              !config->
                  remote_combine_meta[dr])
            continue;

          int32_t token =
              dispatch_meta[pair_idx]
                  .source_token_idx;
          int32_t ci =
              compact_reverse[pair_idx];

          // Find or create unique (dr, token).
          int32_t g = -1;
          for (int32_t j = 0; j < nu; j++) {
            if (s_unique_token[j] == token &&
                s_unique_dr[j] == dr) {
              g = j;
              break;
            }
          }
          if (g == -1) {
            if (nu >= kMaxBatch) continue;
            g = nu++;
            s_unique_token[g] = token;
            s_unique_dr[g] = dr;
          }

          if (nv < kMaxBatch) {
            s_batch_ci[nv] = ci;
            s_batch_weight[nv] = weight;
            s_batch_uid[nv] = g;
          }
          nv++;
        }
      }
      s_num_valid =
          (nv < kMaxBatch) ? nv : kMaxBatch;
      s_num_unique = nu;

      // Claim combine buffer positions.
      int32_t cdr_count[kMaxRanks] = {};
      for (int32_t g = 0; g < nu; g++)
        cdr_count[s_unique_dr[g]]++;

      int32_t cdr_start[kMaxRanks];
      for (int32_t d = 0; d < ws; d++) {
        cdr_start[d] = (cdr_count[d] > 0)
            ? atomicAdd(
                  &config->
                      local_combine_counters[d],
                  cdr_count[d])
            : 0;
      }

      int32_t pos_arr[kMaxRanks];
      for (int32_t d = 0; d < ws; d++)
        pos_arr[d] = cdr_start[d];
      int32_t max_c = ss_c * ws;
      for (int32_t g = 0; g < nu; g++) {
        int32_t dr = s_unique_dr[g];
        int32_t local_off = pos_arr[dr]++;
        s_unique_wpos[g] = (local_off < ss_c)
            ? rank * ss_c + local_off
            : max_c;
      }
    }
    __syncthreads();

    DC_TIMESTAMP(config, kDarNumSteps + 3);
    // cas:sw_scan — thread-0 HBM scan done

    int32_t nu = s_num_unique;
    int32_t nv = s_num_valid;

    if (nu <= kCasMaxUniqueDecode) {
      // Fast path: nu accumulators in smem.
      for (int32_t k = threadIdx.x; k < nu * K;
           k += blockDim.x)
        s_accum[k] = 0.0f;

      DC_TIMESTAMP(config, kDarNumSteps + 4);

      for (int32_t i = 0; i < nv; i++) {
        int32_t u = s_batch_uid[i];
        const T* src =
            expert_output + s_batch_ci[i] * K;
        float w = s_batch_weight[i];
        for (int32_t k = threadIdx.x; k < K;
             k += blockDim.x)
          s_accum[u * K + k] +=
              static_cast<float>(src[k]) * w;
      }

      DC_TIMESTAMP(config, kDarNumSteps + 5);

      constexpr int32_t kVecD =
          sizeof(int4) / sizeof(T);
      const int32_t K4d = K / kVecD;
      int32_t max_c_wr = ss_c * ws;
      for (int32_t u = 0; u < nu; u++) {
        int32_t write_pos = s_unique_wpos[u];
        if (write_pos >= max_c_wr) continue;
        int32_t dr = s_unique_dr[u];
        T* dest = reinterpret_cast<T*>(
            config->remote_combine_recv[dr]);
        int4* dest4 = reinterpret_cast<int4*>(
            dest + write_pos * K);
        for (int32_t i = threadIdx.x; i < K4d;
             i += blockDim.x) {
          T packed[kVecD];
          #pragma unroll
          for (int32_t j = 0; j < kVecD; j++)
            packed[j] = static_cast<T>(
                s_accum[u * K + i * kVecD + j]);
          dest4[i] =
              *reinterpret_cast<const int4*>(
                  packed);
        }
        if (threadIdx.x == 0) {
          TokenMetadata* meta =
              reinterpret_cast<TokenMetadata*>(
                  config->
                      remote_combine_meta[dr]);
          meta[write_pos].source_rank = rank;
          meta[write_pos].source_token_idx =
              s_unique_token[u];
          meta[write_pos].expert_id = 0;
          meta[write_pos].topk_weight = 1.0f;
        }
      }

      DC_TIMESTAMP(config, kDarNumSteps + 6);
      DC_TIMESTAMP(config, kDarNumSteps + 7);
      DC_TIMESTAMP(config, kDarNumSteps + 8);
    } else {
      // Sequential fallback (rare for decode).
      DC_TIMESTAMP(config, kDarNumSteps + 4);
      DC_TIMESTAMP(config, kDarNumSteps + 5);
      int32_t max_c_wr = ss_c * ws;
      for (int32_t u = 0; u < nu; u++) {
        for (int32_t k = threadIdx.x; k < K;
             k += blockDim.x)
          s_accum[k] = 0.0f;
        __syncthreads();
        for (int32_t i = 0; i < nv; i++) {
          if (s_batch_uid[i] != u) continue;
          const T* src =
              expert_output + s_batch_ci[i] * K;
          float w = s_batch_weight[i];
          for (int32_t k = threadIdx.x; k < K;
               k += blockDim.x)
            s_accum[k] +=
                static_cast<float>(src[k]) * w;
        }
        __syncthreads();
        int32_t write_pos = s_unique_wpos[u];
        if (write_pos < max_c_wr) {
          int32_t dr = s_unique_dr[u];
          T* dest = reinterpret_cast<T*>(
              config->remote_combine_recv[dr]);
          constexpr int32_t kVecS =
              sizeof(int4) / sizeof(T);
          const int32_t K4s = K / kVecS;
          int4* dest4 = reinterpret_cast<int4*>(
              dest + write_pos * K);
          for (int32_t i = threadIdx.x; i < K4s;
               i += blockDim.x) {
            T packed[kVecS];
            #pragma unroll
            for (int32_t j = 0; j < kVecS; j++)
              packed[j] = static_cast<T>(
                  s_accum[i * kVecS + j]);
            dest4[i] =
                *reinterpret_cast<const int4*>(
                    packed);
          }
          if (threadIdx.x == 0) {
            TokenMetadata* meta =
                reinterpret_cast<TokenMetadata*>(
                    config->
                        remote_combine_meta[dr]);
            meta[write_pos].source_rank = rank;
            meta[write_pos].source_token_idx =
                s_unique_token[u];
            meta[write_pos].expert_id = 0;
            meta[write_pos].topk_weight = 1.0f;
          }
        }
        __syncthreads();
      }
      DC_TIMESTAMP(config, kDarNumSteps + 6);
      DC_TIMESTAMP(config, kDarNumSteps + 7);
      DC_TIMESTAMP(config, kDarNumSteps + 8);
    }

    } else {
    // ============================================================
    // PREFILL PATH: Cooperative HBM loading, kCasMaxUnique=11.
    // Higher smem (154KB) but better throughput for large M.
    // ============================================================

    // Step 1: Thread 0 computes per-section chunk
    // ranges and flat load count.
    if (threadIdx.x == 0) {
      s_num_valid = 0;
      s_num_unique = 0;
      int32_t total = 0;
      for (int32_t s = 0; s < ws; s++) {
        int32_t count = config->
            remote_dispatch_offsets[rank][s];
        if (count > ss_d) count = ss_d;
        int32_t chunk =
            (count + kPersistentGrid - 1)
            / kPersistentGrid;
        int32_t my_start = blockIdx.x * chunk;
        int32_t my_end = my_start + chunk;
        if (my_end > count) my_end = count;
        int32_t cnt = (my_end > my_start)
            ? (my_end - my_start) : 0;
        s_load_sec_start[s] = my_start;
        s_load_sec_count[s] = cnt;
        s_load_sec_pair_base[s] = s * ss_d;
        total += cnt;
      }
      s_total_load =
          (total < kMaxBatch) ? total : kMaxBatch;
    }
    __syncthreads();

    // Step 2: All threads cooperatively load
    // dispatch_meta + compact_reverse into smem.
    // Coalesced reads within each section.
    {
      int32_t flat_off = 0;
      for (int32_t s = 0; s < ws; s++) {
        int32_t cnt = s_load_sec_count[s];
        int32_t pair_base =
            s_load_sec_pair_base[s]
            + s_load_sec_start[s];
        for (int32_t i = threadIdx.x; i < cnt;
             i += blockDim.x) {
          int32_t dest = flat_off + i;
          if (dest < kMaxBatch) {
            int32_t pi = pair_base + i;
            s_entries[dest] = dispatch_meta[pi];
            s_cr[dest] = compact_reverse[pi];
          }
        }
        flat_off += cnt;
      }
    }
    __syncthreads();

    // Step 3: Thread-0 scan on smem data (no HBM).
    // Dedup into unique (dr, token) groups, fill
    // batch arrays, claim combine positions, then
    // counting-sort entries by uid for tiled path.
    if (threadIdx.x == 0) {
      int32_t total_load = s_total_load;
      int32_t nv = 0;
      int32_t nu = 0;
      for (int32_t i = 0; i < total_load; i++) {
        float weight = s_entries[i].topk_weight;
        if (weight == 0.0f) continue;
        int32_t dr = s_entries[i].source_rank;
        if (dr < 0 || dr >= ws) continue;
        if (!config->
                remote_combine_offsets[dr] ||
            !config->
                remote_combine_recv[dr] ||
            !config->
                remote_combine_meta[dr])
          continue;

        int32_t token =
            s_entries[i].source_token_idx;
        int32_t ci = s_cr[i];

        // Find or create unique (dr, token).
        int32_t g = -1;
        for (int32_t j = 0; j < nu; j++) {
          if (s_unique_token[j] == token &&
              s_unique_dr[j] == dr) {
            g = j;
            break;
          }
        }
        if (g == -1) {
          if (nu >= kMaxBatch) continue;
          g = nu++;
          s_unique_token[g] = token;
          s_unique_dr[g] = dr;
        }

        if (nv < kMaxBatch) {
          s_batch_ci[nv] = ci;
          s_batch_weight[nv] = weight;
          s_batch_uid[nv] = g;
        }
        nv++;
      }
      int32_t nv_clamped =
          (nv < kMaxBatch) ? nv : kMaxBatch;
      s_num_valid = nv_clamped;
      s_num_unique = nu;

      // Claim combine buffer positions.
      int32_t cdr_count[kMaxRanks] = {};
      for (int32_t g = 0; g < nu; g++)
        cdr_count[s_unique_dr[g]]++;

      int32_t cdr_start[kMaxRanks];
      for (int32_t d = 0; d < ws; d++) {
        cdr_start[d] = (cdr_count[d] > 0)
            ? atomicAdd(
                  &config->
                      local_combine_counters[d],
                  cdr_count[d])
            : 0;
      }

      int32_t pos_arr[kMaxRanks];
      for (int32_t d = 0; d < ws; d++)
        pos_arr[d] = cdr_start[d];
      int32_t max_c = ss_c * ws;
      for (int32_t g = 0; g < nu; g++) {
        int32_t dr = s_unique_dr[g];
        int32_t local_off = pos_arr[dr]++;
        s_unique_wpos[g] = (local_off < ss_c)
            ? rank * ss_c + local_off
            : max_c;
      }

      // Counting-sort entries by uid so
      // each tile iterates only its own entries.
      // Reuse s_entries/s_cr smem (no longer needed
      // after coop load scan) for sort temporaries
      // to avoid local memory spill to HBM.
      int32_t* uid_count =
          reinterpret_cast<int32_t*>(&s_entries[0]);
      for (int32_t i = 0; i < nu; i++)
        uid_count[i] = 0;
      for (int32_t i = 0; i < nv_clamped; i++)
        uid_count[s_batch_uid[i]]++;

      s_uid_start[0] = 0;
      for (int32_t u = 0; u < nu; u++)
        s_uid_start[u + 1] =
            s_uid_start[u] + uid_count[u];

      // Scatter into sorted order using smem temps.
      // Layout in s_entries/s_cr region (2560B total):
      //   uid_count[128] = s_entries[0..511] (done)
      //   spos reuses uid_count (overwritten below)
      //   tmp_ci[128] = s_entries[512..1023]
      //   tmp_w[128]  = s_entries[1024..1535]
      //   tmp_uid[128]= s_entries[1536..2047]
      int32_t* spos = uid_count;  // reuse uid_count
      for (int32_t u = 0; u < nu; u++)
        spos[u] = s_uid_start[u];

      int32_t* tmp_ci = uid_count + kMaxBatch;
      float* tmp_w =
          reinterpret_cast<float*>(
              tmp_ci + kMaxBatch);
      int32_t* tmp_uid =
          reinterpret_cast<int32_t*>(
              tmp_w + kMaxBatch);
      for (int32_t i = 0; i < nv_clamped; i++) {
        int32_t u = s_batch_uid[i];
        int32_t p = spos[u]++;
        tmp_ci[p] = s_batch_ci[i];
        tmp_w[p] = s_batch_weight[i];
        tmp_uid[p] = u;
      }
      for (int32_t i = 0; i < nv_clamped; i++) {
        s_batch_ci[i] = tmp_ci[i];
        s_batch_weight[i] = tmp_w[i];
        s_batch_uid[i] = tmp_uid[i];
      }
    }
    __syncthreads();

    DC_TIMESTAMP(config, kDarNumSteps + 3);
    // cas:sw_scan — coop load + scan + sort done

    // Single-pass local reduction + NVLink write.
    int32_t nu = s_num_unique;
    int32_t nv = s_num_valid;

    if (nu <= kCasMaxUnique) {
      // Fast path: single-pass with nu accumulators
      // in shared memory (nu × K floats).
      {
        int4 z4 = make_int4(0, 0, 0, 0);
        int4* acc4 = reinterpret_cast<int4*>(s_accum);
        int32_t n4 = nu * K / 4;  // 4 floats per int4
        for (int32_t i = threadIdx.x; i < n4;
             i += blockDim.x)
          acc4[i] = z4;
      }

      DC_TIMESTAMP(config, kDarNumSteps + 4);
      // cas:sw_zero — accumulators zeroed

      // Accumulate all entries in one pass.
      for (int32_t i = 0; i < nv; i++) {
        int32_t u = s_batch_uid[i];
        const T* src =
            expert_output + s_batch_ci[i] * K;
        float w = s_batch_weight[i];
        for (int32_t k = threadIdx.x; k < K;
             k += blockDim.x)
          s_accum[u * K + k] +=
              static_cast<float>(src[k]) * w;
      }

      DC_TIMESTAMP(config, kDarNumSteps + 5);
      // cas:sw_accum — HBM reads + accumulation done

      // Write all reduced vectors to NVLink (int4).
      constexpr int32_t kVec =
          sizeof(int4) / sizeof(T);
      const int32_t K4 = K / kVec;
      int32_t max_c_wr = ss_c * ws;
      for (int32_t u = 0; u < nu; u++) {
        int32_t write_pos = s_unique_wpos[u];
        if (write_pos >= max_c_wr) continue;
        int32_t dr = s_unique_dr[u];
        T* dest = reinterpret_cast<T*>(
            config->remote_combine_recv[dr]);
        int4* dest4 = reinterpret_cast<int4*>(
            dest + write_pos * K);
        for (int32_t i = threadIdx.x; i < K4;
             i += blockDim.x) {
          T packed[kVec];
          #pragma unroll
          for (int32_t j = 0; j < kVec; j++)
            packed[j] = static_cast<T>(
                s_accum[u * K + i * kVec + j]);
          dest4[i] =
              *reinterpret_cast<const int4*>(
                  packed);
        }
        if (threadIdx.x == 0) {
          TokenMetadata* meta =
              reinterpret_cast<TokenMetadata*>(
                  config->
                      remote_combine_meta[dr]);
          meta[write_pos].source_rank = rank;
          meta[write_pos].source_token_idx =
              s_unique_token[u];
          meta[write_pos].expert_id = 0;
          meta[write_pos].topk_weight = 1.0f;
        }
      }

      // Stamp tiled-path slots so stale values don't
      // produce garbage deltas in profiling output.
      DC_TIMESTAMP(config, kDarNumSteps + 6);
      DC_TIMESTAMP(config, kDarNumSteps + 7);
      DC_TIMESTAMP(config, kDarNumSteps + 8);
    } else {
      // Tiled accumulation with sorted entries.
      // Each tile iterates ONLY its uid range
      // (no wasted scanning of all entries).
      // Stamp fast-path slots so stale values don't
      // produce garbage deltas in profiling output.
      DC_TIMESTAMP(config, kDarNumSteps + 4);
      DC_TIMESTAMP(config, kDarNumSteps + 5);
      int32_t max_c_wr = ss_c * ws;
      for (int32_t tile = 0; tile < nu;
           tile += kCasMaxUnique) {
        int32_t tile_end = tile + kCasMaxUnique;
        if (tile_end > nu) tile_end = nu;
        int32_t tile_sz = tile_end - tile;

        // Zero tile accumulators (vectorized int4).
        {
          int4 z4 = make_int4(0, 0, 0, 0);
          int4* acc4 = reinterpret_cast<int4*>(
              s_accum);
          int32_t n4 = tile_sz * K / 4;
          for (int32_t i = threadIdx.x; i < n4;
               i += blockDim.x)
            acc4[i] = z4;
        }
        __syncthreads();

        DC_TIMESTAMP(config, kDarNumSteps + 6);
        // cas:tile_zero

        // Accumulate only entries belonging to
        // this tile (sorted by uid).
        int32_t entry_start = s_uid_start[tile];
        int32_t entry_end =
            s_uid_start[tile_end];
        for (int32_t i = entry_start;
             i < entry_end; i++) {
          int32_t u = s_batch_uid[i];
          const T* src =
              expert_output + s_batch_ci[i] * K;
          float w = s_batch_weight[i];
          for (int32_t k = threadIdx.x; k < K;
               k += blockDim.x)
            s_accum[(u - tile) * K + k] +=
                static_cast<float>(src[k]) * w;
        }
        __syncthreads();

        DC_TIMESTAMP(config, kDarNumSteps + 7);
        // cas:tile_accum

        // Write tile results to NVLink (int4).
        constexpr int32_t kVecT =
            sizeof(int4) / sizeof(T);
        const int32_t K4t = K / kVecT;
        for (int32_t u = tile; u < tile_end; u++) {
          int32_t write_pos = s_unique_wpos[u];
          if (write_pos >= max_c_wr) continue;
          int32_t dr = s_unique_dr[u];
          T* dest = reinterpret_cast<T*>(
              config->remote_combine_recv[dr]);
          int4* dest4 = reinterpret_cast<int4*>(
              dest + write_pos * K);
          for (int32_t i = threadIdx.x; i < K4t;
               i += blockDim.x) {
            T packed[kVecT];
            #pragma unroll
            for (int32_t j = 0; j < kVecT; j++)
              packed[j] = static_cast<T>(
                  s_accum[(u - tile) * K
                      + i * kVecT + j]);
            dest4[i] =
                *reinterpret_cast<const int4*>(
                    packed);
          }
          if (threadIdx.x == 0) {
            TokenMetadata* meta =
                reinterpret_cast<TokenMetadata*>(
                    config->
                        remote_combine_meta[dr]);
            meta[write_pos].source_rank = rank;
            meta[write_pos].source_token_idx =
                s_unique_token[u];
            meta[write_pos].expert_id = 0;
            meta[write_pos].topk_weight = 1.0f;
          }
        }

        DC_TIMESTAMP(config, kDarNumSteps + 8);
        // cas:tile_nvlink
      }
    }

    } // end prefill path

    DC_TIMESTAMP(config, kDarNumSteps + 9);
    // cas:tile_done — all tiles / fast path done

    // fence#1 removed: NVLink stores drain in background
    // during grid_sync + offset_push. Each block fences
    // once before P2P (block 0 after offset_push, others
    // in else branch).
    __syncthreads();
    if (threadIdx.x == 0) {
      atomicAdd(config->combine_done_counter,
                static_cast<FlagType>(1));
    }
  }

  // ---- Grid sync + P2P barrier ----
  DC_TIMESTAMP(config, kDarNumSteps + 10);
  // cas:grid_sync

  // Block 0 waits for kPersistentGrid (one increment
  // per block), does P2P barrier. Others wait for
  // barrier counter.
  if (blockIdx.x == 0) {
    if (threadIdx.x == 0) {
      FlagType target =
          cd_base + kPersistentGrid;
      while (dc_ld_flag_acquire(
                 config->combine_done_counter)
              < target)
        __nanosleep(200);
    }
    __syncthreads();

    DC_TIMESTAMP(config, kDarNumSteps + 11);
    // cas:offset_push

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

    DC_TIMESTAMP(config, kDarNumSteps + 12);
    // cas:fence2

    __threadfence_system();

    DC_TIMESTAMP(config, kDarNumSteps + 13);
    // cas:p2p_wait

    // Opt 4: Add __nanosleep to P2P spin loop
    // (was tight spin, now yields to reduce NVLink
    // polling pressure).
    if (tid < ws) {
      dc_st_flag_release(
          &config->peer_signals[tid]->flags[rank],
          barrier_expected);
      while (dc_ld_flag_acquire(
          &config->self_signals->flags[tid])
              != barrier_expected)
        __nanosleep(100);
    }

    __syncthreads();

    if (tid == 0) {
      dc_st_flag_release(
          &config->self_signals->counter,
          barrier_expected);
    }
  } else {
    // Blocks 1..(gridDim-1): fence scan_write stores,
    // then wait for P2P completion. With persistent grid,
    // this fence completes before block 0's fence+P2P.
    __threadfence_system();

    if (blockIdx.x >= kPersistentGrid) {
      __threadfence();
    }
    if (threadIdx.x == 0) {
      while (dc_ld_flag_acquire(
          &config->self_signals->counter)
              != barrier_expected)
        __nanosleep(200);
    }
    __syncthreads();
  }

  // ---- Phase 3: Scatter-add to fp32 accum ----
  DC_TIMESTAMP(config, kDarNumSteps + 14);  // cas:scatter_add
  // (timestamp after barrier, before scatter-add)

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

  // ---- Grid-wide sync: all scatter-add atomics done ----
  // Each block does threadfence (ensures its atomicAdds
  // are visible globally) then increments counter.
  // All blocks spin until all have checked in.
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    atomicAdd(config->scatter_done_counter,
              static_cast<FlagType>(1));
  }
  if (threadIdx.x == 0) {
    FlagType target = sd_base + gridDim.x;
    while (dc_ld_flag_acquire(
               config->scatter_done_counter)
            < target)
      __nanosleep(200);
  }
  __syncthreads();

  DC_TIMESTAMP(config, kDarNumSteps + 15);  // cas:convert

  // ---- Phase 4: fp32 → output dtype conversion ----
  // Replaces separate fp32_to_half_kernel launch.
  {
    const int32_t N = M * K;
    for (int32_t i =
             blockIdx.x * blockDim.x + threadIdx.x;
         i < N; i += gridDim.x * blockDim.x) {
      output[i] = static_cast<T>(accum[i]);
    }
  }

  DC_TIMESTAMP(config, kDarNumSteps + 16);  // cas:end
}

// ====================================================================
// ---- Section compaction ----
// Gathers valid entries from scattered per-sender sections
// into contiguous positions. Builds compact_data_remap
// (compact_idx → compact data position for gather),
// compact_expert_topk_ids, compact_expert_topk_weights,
// and compact_reverse (original_idx → compact_idx for
// combine kernel to read expert output).
// Grid = kPersistentGrid, block = kBlockSize.
template <typename T>
__global__ void dar_compact_kernel(
    const int64_t* __restrict__ expert_topk_ids,
    const float* __restrict__ expert_topk_weights,
    const int32_t* __restrict__ data_remap,
    int64_t* __restrict__ compact_expert_topk_ids,
    float* __restrict__ compact_expert_topk_weights,
    int32_t* __restrict__ compact_data_remap,
    int32_t* __restrict__ compact_reverse,
    const T* __restrict__ dispatch_recv,
    T* __restrict__ expert_x,
    const DispatchCombineConfig* __restrict__ config,
    int32_t mc_compact,
    int32_t num_physical_experts,
    int32_t K) {
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
  // Each block cooperatively handles one entry at a time:
  // thread 0 writes metadata, all threads copy token data
  // via int4 vectorized loads/stores.
  constexpr int32_t kElemsPerI4 =
      static_cast<int32_t>(sizeof(int4) / sizeof(T));
  const int32_t K4 = K / kElemsPerI4;

  // Block-strided loop over flattened valid entries.
  // Use num_valid (sum of all section counts) as total.
  for (int32_t flat_idx = blockIdx.x;
       flat_idx < num_valid;
       flat_idx += gridDim.x) {
    // Map flat_idx to (section, offset_in_section)
    // using prefix sums. Linear scan over sections
    // (ws <= kMaxRanks=64, cheap in registers).
    int32_t s = 0;
    int32_t remaining = flat_idx;
    while (s < ws - 1
           && remaining >= s_section_count[s]) {
      remaining -= s_section_count[s];
      s++;
    }
    const int32_t original_idx = s * ss + remaining;
    const int32_t compact_idx =
        s_compact_offset[s] + remaining;

    // Thread 0 writes metadata.
    if (threadIdx.x == 0) {
      compact_data_remap[compact_idx] =
          data_remap[original_idx];
      compact_expert_topk_ids[compact_idx] =
          expert_topk_ids[original_idx];
      compact_expert_topk_weights[compact_idx] =
          expert_topk_weights[original_idx];
      compact_reverse[original_idx] = compact_idx;
    }

    // All threads cooperatively copy token data.
    const int32_t data_pos =
        data_remap[original_idx];
    const int4* src =
        reinterpret_cast<const int4*>(
            dispatch_recv
            + static_cast<int64_t>(data_pos) * K);
    int4* dst = reinterpret_cast<int4*>(
        expert_x
        + static_cast<int64_t>(compact_idx) * K);
    for (int32_t k = threadIdx.x; k < K4;
         k += blockDim.x) {
      dst[k] = src[k];
    }
    __syncthreads();
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
    int32_t num_physical_experts,
    int32_t routing_mode) {
  const int32_t rank = config->rank;
  const int32_t ws = config->world_size;
  const int32_t max_rep = config->max_replicas;
  const int32_t epr = config->physical_experts_per_rank;
  const int32_t NL = config->num_logical_experts;

  // Shared memory layout (reused across phases):
  // Scan_write: s_expert_counts[NL] + s_grp_count[ws]
  //             + s_ent_lid[64] + s_ent_wt[64]
  //             + s_ent_grp[64] + s_replica_count[NL]
  //             + s_l2p_map[NL*max_rep]
  // Phase C:    s_expert_sum[NL] + s_replica_count[NL]
  //             + s_l2p_map[NL*max_rep]
  //             + routing_selection_smem[NL]
  //             + rank_active_counts[ws]
  // Phases don't overlap, so same memory is reused.
  extern __shared__ int32_t shared[];

  DC_TIMESTAMP(config, 0);  // dar:read_counters

  // Read monotonic counter base values BEFORE any phase
  // (for CUDA graph replay compatibility).
  FlagType rf_expected = 0;
  FlagType pa_base = 0;
  FlagType barrier_expected = 0;
  if (threadIdx.x == 0) {
    rf_expected =
        static_cast<FlagType>(
            *config->routing_ready_flag) + 1;
    pa_base =
        static_cast<FlagType>(
            *config->phase_a_done_counter);
    barrier_expected =
        config->self_signals->counter + 1;
  }
  // Broadcast to all threads via shared mem.
  __shared__ FlagType s_rf_expected;
  __shared__ FlagType s_pa_base;
  __shared__ FlagType s_barrier_expected;
  if (threadIdx.x == 0) {
    s_rf_expected = rf_expected;
    s_pa_base = pa_base;
    s_barrier_expected = barrier_expected;
  }
  __syncthreads();
  rf_expected = s_rf_expected;
  pa_base = s_pa_base;
  barrier_expected = s_barrier_expected;

  // ============================================================
  // FUSED SCAN+WRITE: per-token expansion + position claiming
  //   + NVLink write. Eliminates two-pass overhead (prefix sum,
  //   positions_ready signal/wait).
  // ============================================================
  DC_TIMESTAMP(config, 1);  // dar:scan_write

  constexpr int32_t kMaxEntries = 64;
  int32_t* s_expert_counts = shared;           // [NL]
  int32_t* s_grp_count = shared + NL;          // [ws]
  int32_t* s_ent_lid   = shared + NL + ws;     // [kME]
  float*   s_ent_wt    = reinterpret_cast<float*>(
      s_ent_lid + kMaxEntries);                 // [kME]
  int32_t* s_ent_grp   = reinterpret_cast<int32_t*>(
      s_ent_wt + kMaxEntries);                  // [kME]
  // Preloaded config arrays (fit in existing alloc).
  int32_t* s_replica_count =
      s_ent_grp + kMaxEntries;                  // [NL]
  int32_t* s_l2p_map =
      s_replica_count + NL;                     // [NL*mr]
  // Mode-3 only: block-local per-rank load counter for
  // tie-breaking the greedy minimize-fanout router. Lives
  // in shared mem (per-block, per-invocation — invisible
  // to CUDA-graph capture/replay).
  int32_t* s_block_rank_load =
      s_l2p_map + NL * max_rep;                 // [ws]

  __shared__ int32_t s_total_entries;
  __shared__ int32_t s_grp_base[kMaxRanks];

  // Zero expert counts + preload config arrays.
  for (int32_t e = threadIdx.x; e < NL;
       e += blockDim.x) {
    s_expert_counts[e] = 0;
    s_replica_count[e] =
        config->logical_replica_count[e];
  }
  for (int32_t i = threadIdx.x;
       i < NL * max_rep; i += blockDim.x) {
    s_l2p_map[i] =
        config->logical_to_physical_map[i];
  }
  if (threadIdx.x < ws) {
    s_grp_count[threadIdx.x] = 0;
    s_block_rank_load[threadIdx.x] = 0;
  }
  if (threadIdx.x == 0) s_total_entries = 0;
  __syncthreads();

  // Per-token loop: expand → claim → write.
  // Groups pre-allocated by dest_rank (g == dr).
  for (int32_t t = blockIdx.x; t < M;
       t += gridDim.x) {
    // Step 1: Parallel expansion (threads 0..topk-1)
    // + direct group tagging (fused with old Step 2).
    // Mode 3 takes a separate single-thread path: the
    // greedy minimize-rank-fanout decision is sequential
    // across topk picks of the same token (later picks see
    // the chosen-ranks set built by earlier picks).
    if (routing_mode == 3) {
      if (threadIdx.x == 0) {
        constexpr int32_t kMaxTopkLocal = 16;
        int32_t chosen_phys[kMaxTopkLocal];
        uint64_t chosen_ranks_mask = 0;
        const int32_t topk_eff =
            (topk < kMaxTopkLocal) ? topk : kMaxTopkLocal;

        // Pass 1: rc==1 forced (no choice). Establishes
        // the initial chosen_ranks_mask before Pass 2's
        // multi-replica greedy picks.
        for (int32_t k = 0; k < topk_eff; k++) {
          int32_t lid = topk_ids[t * topk + k];
          if (lid < 0 || lid >= NL) {
            chosen_phys[k] = -1;
            continue;
          }
          int32_t rc = s_replica_count[lid];
          if (rc > max_rep) rc = max_rep;
          if (rc <= 0) {
            chosen_phys[k] = -1;
          } else if (rc == 1) {
            int32_t phys = s_l2p_map[lid * max_rep];
            chosen_phys[k] = phys;
            int32_t r = phys / epr;
            if (r >= 0 && r < ws) {
              chosen_ranks_mask |= 1ULL << r;
            }
          } else {
            chosen_phys[k] = -2;  // defer to Pass 2
          }
        }

        // Pass 2: multi-replica picks. Primary: prefer a
        // replica whose rank is already in chosen_ranks
        // (collapse fanout). Secondary: lower block-local
        // load. Final: smaller rank id (determinism).
        for (int32_t k = 0; k < topk_eff; k++) {
          if (chosen_phys[k] != -2) continue;
          int32_t lid = topk_ids[t * topk + k];
          int32_t rc = s_replica_count[lid];
          if (rc > max_rep) rc = max_rep;
          int32_t best_phys = -1;
          int32_t best_rank = INT_MAX;
          int32_t best_in_set = -1;
          int32_t best_load = INT_MAX;
          for (int32_t rep = 0; rep < rc; rep++) {
            int32_t phys = s_l2p_map[lid * max_rep + rep];
            int32_t r = phys / epr;
            if (r < 0 || r >= ws) continue;
            int32_t in_set = static_cast<int32_t>(
                (chosen_ranks_mask >> r) & 1ULL);
            int32_t load = s_block_rank_load[r];
            bool better =
                (in_set > best_in_set)
                || (in_set == best_in_set
                    && load < best_load)
                || (in_set == best_in_set
                    && load == best_load
                    && r < best_rank);
            if (better) {
              best_phys = phys;
              best_rank = r;
              best_in_set = in_set;
              best_load = load;
            }
          }
          chosen_phys[k] = best_phys;
          if (best_rank >= 0 && best_rank < ws) {
            chosen_ranks_mask |= 1ULL << best_rank;
          }
        }

        // Push entries to per-block staging arrays. For
        // mode 3 we store the PHYSICAL expert id in
        // s_ent_lid (Phase D2 reads it back as physical).
        for (int32_t k = 0; k < topk_eff; k++) {
          int32_t phys = chosen_phys[k];
          if (phys < 0) continue;
          int32_t lid = topk_ids[t * topk + k];
          if (lid < 0 || lid >= NL) continue;
          float wt = topk_weights[t * topk + k];
          int32_t dr = phys / epr;
          if (dr < 0 || dr >= ws) continue;
          if (!config->remote_dispatch_offsets[dr]
              || !config->remote_dispatch_recv[dr]
              || !config->remote_dispatch_meta[dr])
            continue;
          // Logical-id count for global stats (still
          // matches modes 0/1/2 semantics).
          atomicAdd(&s_expert_counts[lid], 1);
          int32_t ei = atomicAdd(&s_total_entries, 1);
          if (ei < kMaxEntries) {
            s_ent_lid[ei] = phys;  // physical for mode 3
            s_ent_wt[ei] = wt;
            s_ent_grp[ei] = dr;
            atomicAdd(&s_grp_count[dr], 1);
            // Single-thread; plain ++ is fine.
            s_block_rank_load[dr] += 1;
          }
        }
      }
    } else if (threadIdx.x < topk) {
      int32_t slot = threadIdx.x;
      int32_t lid = topk_ids[t * topk + slot];
      if (lid >= 0 && lid < NL) {
        atomicAdd(&s_expert_counts[lid], 1);
        float wt = topk_weights[t * topk + slot];
        int32_t rc = s_replica_count[lid];
        if (rc > max_rep) rc = max_rep;
        if (routing_mode == 2 && rc > 1) {
          // Mode 2: per-token round-robin replica
          // selection. Only send to the selected
          // replica, not all replicas. Deterministic
          // for Phase D2 reconstruction via
          // source_token_idx % rc.
          int32_t sel_rep = t % rc;
          int32_t phys =
              s_l2p_map[lid * max_rep + sel_rep];
          int32_t dr = phys / epr;
          if (dr >= 0 && dr < ws
              && config->
                  remote_dispatch_offsets[dr]
              && config->
                  remote_dispatch_recv[dr]
              && config->
                  remote_dispatch_meta[dr]) {
            int32_t ei =
                atomicAdd(&s_total_entries, 1);
            if (ei < kMaxEntries) {
              s_ent_lid[ei] = lid;
              s_ent_wt[ei] = wt;
              s_ent_grp[ei] = dr;
              atomicAdd(&s_grp_count[dr], 1);
            }
          }
        } else {
          for (int32_t rep = 0; rep < rc; rep++) {
            int32_t phys =
                s_l2p_map[lid * max_rep + rep];
            int32_t dr = phys / epr;
            if (dr < 0 || dr >= ws) continue;
            if (!config->
                    remote_dispatch_offsets[dr] ||
                !config->
                    remote_dispatch_recv[dr] ||
                !config->
                    remote_dispatch_meta[dr])
              continue;
            int32_t ei =
                atomicAdd(&s_total_entries, 1);
            if (ei < kMaxEntries) {
              s_ent_lid[ei] = lid;
              s_ent_wt[ei] = wt;
              s_ent_grp[ei] = dr;
              atomicAdd(&s_grp_count[dr], 1);
            }
          }
        }
      }
    }
    __syncthreads();

    DC_TIMESTAMP(config, 2);  // dar:scan_expand
    DC_TIMESTAMP(config, 3);  // dar:scan_group (fused)

    // Step 3: Claim write positions via atomicAdd.
    if (threadIdx.x < ws &&
        s_grp_count[threadIdx.x] > 0) {
      int32_t local_off = atomicAdd(
          &config->local_dispatch_counters[
              threadIdx.x],
          s_grp_count[threadIdx.x]);
      int32_t ss = config->dispatch_section_size;
      s_grp_base[threadIdx.x] = (local_off < ss)
          ? rank * ss + local_off
          : config->max_recv;
    }
    __syncthreads();

    DC_TIMESTAMP(config, 4);  // dar:scan_claim

    // Step 4: Vectorized data + metadata per group.
    // int4 = 16 bytes → 8x fewer stores than bf16.
    const int32_t K4 = K *
        static_cast<int32_t>(sizeof(T)) /
        static_cast<int32_t>(sizeof(int4));
    for (int32_t g = 0; g < ws; g++) {
      if (s_grp_count[g] == 0) continue;
      int32_t base = s_grp_base[g];
      int32_t n = s_grp_count[g];
      if (base >= config->max_recv) continue;
      if (base + n > config->max_recv)
        n = config->max_recv - base;

      // All threads: vectorized data copy (int4).
      // Data position: deterministic from (rank, t).
      // Each token written once per dest (dedup).
      int32_t data_pos =
          rank * config->max_num_tokens_per_rank
          + t;
      const int4* src4 =
          reinterpret_cast<const int4*>(
              input + t * K);
      int4* dest4 = reinterpret_cast<int4*>(
          reinterpret_cast<T*>(
              config->remote_dispatch_recv[g])
          + data_pos * K);
      for (int32_t i = threadIdx.x; i < K4;
           i += blockDim.x) {
        dest4[i] = src4[i];
      }

      // Thread 0: write metadata per entry.
      if (threadIdx.x == 0) {
        TokenMetadata* meta =
            reinterpret_cast<TokenMetadata*>(
                config->remote_dispatch_meta[g]);
        int32_t ne2 = s_total_entries;
        if (ne2 > kMaxEntries) ne2 = kMaxEntries;
        int32_t mi = 0;
        for (int32_t ei = 0;
             ei < ne2 && mi < n; ei++) {
          if (s_ent_grp[ei] != g) continue;
          meta[base + mi].source_rank = rank;
          meta[base + mi].source_token_idx = t;
          meta[base + mi].expert_id =
              s_ent_lid[ei];
          meta[base + mi].topk_weight =
              s_ent_wt[ei];
          mi++;
        }
      }
    }

    // Reset per-token state for next iteration.
    if (threadIdx.x < ws)
      s_grp_count[threadIdx.x] = 0;
    if (threadIdx.x == 0) s_total_entries = 0;
    __syncthreads();

    DC_TIMESTAMP(config, 5);  // dar:scan_nvlink
  }

  // Flush expert counts to device buffer.
  DC_TIMESTAMP(config, 6);  // dar:expert_flush
  for (int32_t e = threadIdx.x; e < NL;
       e += blockDim.x) {
    int32_t count = s_expert_counts[e];
    if (count > 0) {
      atomicAdd(&config->local_expert_counts[e],
                count);
    }
  }

  // ============================================================
  // DEFERRED FENCE: grid sync + expert push + single fence + P2P
  // ============================================================
  // fence#1 removed: NVLink stores from scan_write drain in the
  // background while grid_sync + expert_push execute. Each block
  // calls __threadfence_system() once before P2P, draining all
  // its pending stores in a single pass. With the persistent
  // grid (108 blocks on 108 SMs), blocks 1-107 start their
  // fence before block 0 (no expert_push to do), so all stores
  // are globally visible before block 0's P2P flag exchange.
  __syncthreads();
  DC_TIMESTAMP(config, 7);  // dar:threadfence_sys (removed)
  if (threadIdx.x == 0) {
    atomicAdd(config->phase_a_done_counter,
              static_cast<FlagType>(1));
  }

  if (blockIdx.x == 0) {
    // Wait for all blocks to finish scan+write.
    if (threadIdx.x == 0) {
      FlagType target = pa_base + gridDim.x;
      while (dc_ld_flag_acquire(
                 config->phase_a_done_counter)
              < target)
        __nanosleep(200);
    }
    __syncthreads();

    DC_TIMESTAMP(config, 8);  // dar:grid_sync

    // Push dispatch offsets on threads NL..NL+ws-1
    // (warp 4), concurrent with expert count push on
    // warps 0-3. Avoids intra-warp NVLink serialization.
    if (threadIdx.x >= NL &&
        threadIdx.x < NL + ws) {
      int32_t dr = threadIdx.x - NL;
      config->remote_dispatch_offsets[dr][rank] =
          config->local_dispatch_counters[dr];
    }

    // Push expert counts to all remote ranks
    // (fire-and-forget NVLink writes, warps 0-3).
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

    DC_TIMESTAMP(config, 9);  // dar:expert_push

    // Reset combine state for next layer.
    const int32_t tid = threadIdx.x;
    if (tid < ws) {
      config->remote_combine_offsets[rank][tid]
          = 0;
      config->local_combine_counters[tid] = 0;
    }

    // Single fence: drains scan_write + expert_push +
    // offset_push + combine_reset stores in one pass.
    // Scan_write stores have been draining in background
    // during grid_sync + expert_push (~1.5 us head start).
    __threadfence_system();

    DC_TIMESTAMP(config, 10);  // dar:fence2

    // P2P barrier exchange.
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
    // Blocks 1-107: fence scan_write stores, then wait
    // for P2P completion. With persistent grid, this
    // fence completes before block 0's fence+P2P.
    __threadfence_system();

    if (threadIdx.x == 0) {
      while (dc_ld_flag_acquire(
          &config->self_signals->counter)
              != barrier_expected)
        __nanosleep(200);
    }
    __syncthreads();
  }

  DC_TIMESTAMP(config, 11);  // dar:p2p_wait

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
  DC_TIMESTAMP(config, 12);  // dar:phase_c_preload

  if (blockIdx.x == 0) {
    if (routing_mode == 2 || routing_mode == 3) {
      // Modes 2 and 3: routing decided in Phase A per
      // token (mode 2: t % rc; mode 3: greedy minimize
      // fanout). No routing_selection table needed —
      // just zero expert_num_tokens and signal readiness.
      DC_TIMESTAMP(config, 13);  // dar:phase_c_route
      DC_TIMESTAMP(config, 14);  // dar:route_pass1
      DC_TIMESTAMP(config, 15);  // dar:route_pass2

      for (int32_t i = threadIdx.x;
           i < num_physical_experts;
           i += blockDim.x) {
        expert_num_tokens[i] = 0;
      }
      __threadfence();
      __syncthreads();

      if (threadIdx.x == 0) {
        dc_st_flag_release(
            config->routing_ready_flag, rf_expected);
      }
      DC_TIMESTAMP(config, 16);  // dar:route_writeback
    } else {
    // Reuse shared[] for Phase C preload layout.
    // routing_mode=0 (minimize experts):
    //   routing_sel[NL], rank_active[ws],
    //   s_multi_experts[NL], s_num_multi[1]
    // routing_mode=1 (balance tokens):
    //   section_routing[ws*NL], s_section_counts[ws*NL],
    //   rank_active[ws], s_multi_experts[NL],
    //   s_num_multi[1]
    int32_t* s_expert_sum = shared;           // [NL]
    int32_t* s_replica_count = shared + NL;   // [NL]
    int32_t* s_l2p_map = shared + 2 * NL;    // [NL*mr]
    int32_t* base_ptr = shared + 2 * NL + NL * max_rep;

    // Mode-dependent layout after base_ptr.
    int32_t* routing_sel = nullptr;       // mode 0 only
    int32_t* section_routing = nullptr;   // mode 1 only
    int32_t* s_section_counts = nullptr;  // mode 1 only
    int32_t* rank_active = nullptr;       // [ws]
    int32_t* s_multi_experts = nullptr;   // [NL]
    int32_t* s_num_multi = nullptr;       // [1]

    int32_t* s_l2p_rank = nullptr;        // [NL*mr] mode 0 only
    int32_t* s_l2p_slot = nullptr;        // [NL*mr] mode 0 only
    if (routing_mode == 0) {
      routing_sel = base_ptr;             // [NL]
      rank_active = routing_sel + NL;     // [ws]
      s_multi_experts = rank_active + ws; // [NL]
      s_num_multi = s_multi_experts + NL; // [1]
      s_l2p_rank = s_num_multi + 1;       // [NL*mr]
      s_l2p_slot = s_l2p_rank + NL * max_rep;
    } else {
      section_routing = base_ptr;                 // [ws*NL]
      s_section_counts = section_routing + ws*NL; // [ws*NL]
      rank_active = s_section_counts + ws * NL;   // [ws]
      s_multi_experts = rank_active + ws;         // [NL]
      s_num_multi = s_multi_experts + NL;         // [1]
    }

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
    // Mode-specific init.
    if (routing_mode == 0) {
      // Precompute rank/slot for every replica so Pass 2's serial
      // greedy never divides. Same values, computed in parallel.
      for (int32_t i = threadIdx.x;
           i < NL * max_rep;
           i += blockDim.x) {
        const int32_t phys = s_l2p_map[i];
        const int32_t r = phys / epr;
        s_l2p_rank[i] = r;
        s_l2p_slot[i] = phys - r * epr;
      }
      for (int32_t e = threadIdx.x; e < NL;
           e += blockDim.x) {
        routing_sel[e] = -1;
      }
    } else {
      // Preload per-section per-expert counts.
      for (int32_t i = threadIdx.x;
           i < ws * NL; i += blockDim.x) {
        s_section_counts[i] = ec_buf[i];
        section_routing[i] = -1;
      }
    }
    for (int32_t r = threadIdx.x; r < ws;
         r += blockDim.x) {
      rank_active[r] = 0;
    }
    if (threadIdx.x == 0) s_num_multi[0] = 0;
    __syncthreads();

    DC_TIMESTAMP(config, 13);  // dar:phase_c_route

    // ---- Two-pass routing ----
    // Pass 1 (parallel): Route single-replica experts.
    // routing_mode=0: rank_active += 1 (min experts)
    // routing_mode=1: rank_active += count (balance tok)
    for (int32_t e = threadIdx.x; e < NL;
         e += blockDim.x) {
      const int32_t count = s_expert_sum[e];
      if (count == 0) continue;
      int32_t rc = s_replica_count[e];
      if (rc <= 0) continue;
      if (rc > max_rep) rc = max_rep;
      if (rc == 1) {
        const int32_t phys =
            s_l2p_map[e * max_rep];
        if (routing_mode == 0) {
          routing_sel[e] = phys;
          atomicAdd(&rank_active[phys / epr], 1);
        } else {
          for (int32_t s = 0; s < ws; s++)
            section_routing[s * NL + e] = phys;
          atomicAdd(&rank_active[phys / epr], count);
        }
      } else {
        // rc > 1: defer to Pass 2 via compact list.
        int32_t idx = atomicAdd(s_num_multi, 1);
        s_multi_experts[idx] = e;
      }
    }
    __syncthreads();

    DC_TIMESTAMP(config, 14);  // dar:route_pass1

    // Pass 2 (sequential): Route multi-replica experts.
    // routing_mode=0: pick ONE replica per expert,
    //   rank_active += 1 (minimize activated experts).
    // routing_mode=1: section-level splitting — assign
    //   each section's tokens independently to the
    //   least-loaded replica, rank_active += section_cnt.
    if (threadIdx.x == 0) {
      const int32_t nm = s_num_multi[0];
      // Insertion sort compact list by expert index.
      for (int32_t i = 1; i < nm; i++) {
        int32_t key = s_multi_experts[i];
        int32_t j = i - 1;
        while (j >= 0 && s_multi_experts[j] > key) {
          s_multi_experts[j + 1] =
              s_multi_experts[j];
          j--;
        }
        s_multi_experts[j + 1] = key;
      }
      if (routing_mode == 0) {
        // Greedy: one replica per expert.
        // Tie-break on equal rank-active load: prefer the
        // replica with the SMALLEST LOCAL SLOT INDEX
        // (= phys % epr). Under the grouped two-phase
        // placement, originals live at slots 0..S/R-1 and
        // redundants at slots S/R..phy_per_rank-1, so
        // "smaller slot" biases the pick toward the
        // original replica and keeps each rank's active
        // expert set contiguous in the low-slot range.
        //
        // Note: comparing `phys` itself wouldn't do this —
        // phys = rank*epr + slot, so phys ordering is
        // dominated by rank and is equivalent to the old
        // "smaller rank" tie-break (a no-op). The slot
        // extraction below is what actually changes mode-0's
        // behavior.
        for (int32_t idx = 0; idx < nm; idx++) {
          const int32_t e = s_multi_experts[idx];
          int32_t rc = s_replica_count[e];
          if (rc > max_rep) rc = max_rep;
          int32_t best_phys = -1;
          int32_t best_rank = -1;
          int32_t best_cost = INT_MAX;
          int32_t best_slot = INT_MAX;
          for (int32_t i = 0; i < rc; i++) {
            const int32_t li = e * max_rep + i;
            const int32_t phys = s_l2p_map[li];
            const int32_t r = s_l2p_rank[li];
            const int32_t c = rank_active[r];
            const int32_t slot = s_l2p_slot[li];
            const bool better =
                (c < best_cost)
                || (c == best_cost && slot < best_slot)
                || (c == best_cost && slot == best_slot
                    && r < best_rank);
            if (better) {
              best_cost = c;
              best_rank = r;
              best_phys = phys;
              best_slot = slot;
            }
          }
          routing_sel[e] = best_phys;
          rank_active[best_rank] += 1;
        }
      } else if (routing_mode == 1) {
        // Expert-level LPT: process multi-replica experts
        // in descending order of total token count. Large
        // experts are assigned while rank_active is most
        // symmetric, improving balance vs. arbitrary
        // expert-index order. Within each expert, sections
        // are assigned greedily as before (greedy naturally
        // splits load across replicas).
        //
        // Local state: done[NL] = 128 bytes, trivially
        // L1-cached. No spill to global local memory.
        // Cost: O(nm^2 * ws) shared-mem reads;
        // nm<=NL(128), ws<=8 → max ~131K reads (typical
        // ~4.6K for 24 redundant experts). << 1 µs.
        // kMaxNL covers the max num_logical_experts (128).
        // Only zero the first nm entries actually used.
        constexpr int32_t kMaxNL = 128;
        bool done[kMaxNL];
        for (int32_t k = 0; k < nm; k++) done[k] = false;

        for (int32_t ii = 0; ii < nm; ii++) {
          // Selection sort: find unprocessed expert with
          // highest total token count across all sections.
          int32_t best_jj = 0;
          int32_t best_tot = -1;
          for (int32_t jj = 0; jj < nm; jj++) {
            if (done[jj]) continue;
            const int32_t e = s_multi_experts[jj];
            int32_t tot = 0;
            for (int32_t s = 0; s < ws; s++)
              tot += s_section_counts[s * NL + e];
            if (tot > best_tot) {
              best_tot = tot;
              best_jj = jj;
            }
          }
          done[best_jj] = true;
          const int32_t e = s_multi_experts[best_jj];

          // Assign each section of this expert greedily
          // to the least-loaded replica.
          for (int32_t s = 0; s < ws; s++) {
            const int32_t cnt_s =
                s_section_counts[s * NL + e];
            int32_t rc = s_replica_count[e];
            if (rc > max_rep) rc = max_rep;
            int32_t best_phys = -1;
            int32_t best_rank = -1;
            int32_t best_cost = INT_MAX;
            for (int32_t i2 = 0; i2 < rc; i2++) {
              const int32_t phys =
                  s_l2p_map[e * max_rep + i2];
              const int32_t r = phys / epr;
              const int32_t c = rank_active[r];
              if (c < best_cost ||
                  (c == best_cost && r < best_rank)) {
                best_cost = c;
                best_rank = r;
                best_phys = phys;
              }
            }
            section_routing[s * NL + e] = best_phys;
            rank_active[best_rank] += cnt_s;
          }
        }
      } else {
        // routing_mode == 2: even round-robin across
        // replicas. Section s assigned to replica
        // (s % rc). Simple, deterministic, no load
        // tracking — mirrors EPLB's even splitting.
        for (int32_t idx = 0; idx < nm; idx++) {
          const int32_t e = s_multi_experts[idx];
          int32_t rc = s_replica_count[e];
          if (rc > max_rep) rc = max_rep;
          for (int32_t s = 0; s < ws; s++) {
            const int32_t phys =
                s_l2p_map[e * max_rep + (s % rc)];
            section_routing[s * NL + e] = phys;
            const int32_t cnt_s =
                s_section_counts[s * NL + e];
            rank_active[phys / epr] += cnt_s;
          }
        }
      }
    }
    __syncthreads();

    DC_TIMESTAMP(config, 15);  // dar:route_pass2

    // Write routing decisions to global memory.
    if (routing_mode == 0) {
      for (int32_t e = threadIdx.x; e < NL;
           e += blockDim.x) {
        config->routing_selection[e] = routing_sel[e];
      }
    } else {
      for (int32_t i = threadIdx.x;
           i < ws * NL; i += blockDim.x) {
        config->routing_selection[i] =
            section_routing[i];
      }
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
    DC_TIMESTAMP(config, 16);  // dar:route_writeback
    } // end else (routing_mode != 2)
  } else {
    // Wait for routing to complete (blocks 1-31).
    if (threadIdx.x == 0) {
      while (dc_ld_flag_acquire(
              config->routing_ready_flag)
              != rf_expected)
        __nanosleep(200);
    }
    __syncthreads();
    __threadfence();
  }

  // ---- Phase D2: Single-pass fill + routing filter ----
  DC_TIMESTAMP(config, 17);  // dar:phase_d2_filter

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
        data_remap[idx] = 0;  // Harmless for stale

        // Section-aware real check. Clamp count to
        // section size: raw counter may exceed ss_d2.
        int32_t sec = idx / ss_d2;
        int32_t off = idx % ss_d2;
        int32_t sec_cnt = config->
            remote_dispatch_offsets[rank][sec];
        if (sec_cnt > ss_d2) sec_cnt = ss_d2;
        if (sec < ws && off < sec_cnt) {
          // Compact data position (deterministic).
          data_remap[idx] =
              meta_r[idx].source_rank
              * config->max_num_tokens_per_rank
              + meta_r[idx].source_token_idx;

          // Routing filter.
          if (routing_mode == 3) {
            // Mode 3: source rank stored the chosen
            // PHYSICAL expert id directly in expert_id
            // (not logical). It only pushed entries to
            // the chosen rank, so the rank check below
            // is a defensive sanity guard.
            int32_t sel = meta_r[idx].expert_id;
            if (sel < 0
                || sel >= num_physical_experts) {
              meta_w[idx].topk_weight = 0.0f;
            } else if (sel / epr == rank) {
              expert_topk_ids[idx] =
                  static_cast<int64_t>(sel);
              expert_topk_weights[idx] =
                  meta_r[idx].topk_weight;
              atomicAdd(
                  &expert_num_tokens[sel], 1);
            } else {
              meta_w[idx].topk_weight = 0.0f;
            }
          } else {
          const int32_t logical_id =
              meta_r[idx].expert_id;
          if (logical_id < 0
              || logical_id >= NL) {
            meta_w[idx].topk_weight = 0.0f;
          } else if (routing_mode == 2) {
            // Mode 2: reconstruct Phase A's per-token
            // round-robin decision. No routing_selection
            // lookup needed.
            int32_t rc2 = static_cast<int32_t>(
                config->logical_replica_count[
                    logical_id]);
            if (rc2 > max_rep) rc2 = max_rep;
            int32_t sel;
            if (rc2 <= 1) {
              sel = config->logical_to_physical_map[
                  logical_id * max_rep];
            } else {
              int32_t sel_rep =
                  meta_r[idx].source_token_idx % rc2;
              sel = config->logical_to_physical_map[
                  logical_id * max_rep + sel_rep];
            }
            if (sel >= 0
                && sel < num_physical_experts
                && sel / epr == rank) {
              // KEEP: selected replica on our rank.
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
          } else {
            // routing_mode=0: one selection per expert.
            // routing_mode=1: per-section selection
            //   (section_routing[sec * NL + expert]).
            const int32_t sel = (routing_mode == 0)
                ? config->routing_selection[logical_id]
                : config->routing_selection[
                    sec * NL + logical_id];
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
          }  // end else (modes 0/1/2)
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

  DC_TIMESTAMP(config, 18);  // dar:end
}

}  // namespace dispatch_combine
}  // namespace vllm
