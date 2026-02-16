#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace vllm {
namespace dispatch_combine {

#define DC_CUDACHECK(cmd)                                               \
  do {                                                                  \
    cudaError_t e = cmd;                                                \
    if (e != cudaSuccess) {                                             \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__,    \
             cudaGetErrorString(e));                                    \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  } while (0)

// Maximum number of EP ranks supported.
constexpr int kMaxRanks = 64;

// Metadata for each dispatched token-expert pair.
// Packed into 16 bytes for efficient P2P transfer.
struct __align__(16) TokenMetadata {
  int32_t source_rank;       // Originating rank
  int32_t source_token_idx;  // Token index on originating rank
  int32_t expert_id;         // Global expert ID
  float topk_weight;         // Router weight for this token-expert pair
};

// Signal structure for P2P synchronization between ranks.
// Each rank writes to its signal slot after finishing P2P writes.
// Other ranks poll their local signal memory to know data is ready.
struct __align__(128) DispatchCombineSignals {
  // dispatch_counter[dst_rank] = number of token-expert pairs written
  // by this rank to dst_rank's recv buffer.
  volatile int32_t dispatch_counter[kMaxRanks];
  // combine_counter[dst_rank] = number of results written
  // by this rank to dst_rank's combine recv buffer.
  volatile int32_t combine_counter[kMaxRanks];
  // Padding to cache line boundary.
  int32_t _pad[kMaxRanks * 2 - kMaxRanks * 2];
};

// Per-rank buffer configuration passed to CUDA kernels.
struct DispatchCombineConfig {
  // Pointers to each rank's dispatch recv buffer (via IPC).
  // remote_dispatch_recv[r] points to rank r's dispatch recv buffer.
  void* remote_dispatch_recv[kMaxRanks];
  // Pointers to each rank's dispatch metadata buffer (via IPC).
  void* remote_dispatch_meta[kMaxRanks];
  // Pointers to each rank's dispatch write-offset counter (via IPC).
  // Atomic counters on remote GPUs for claiming write slots.
  int32_t* remote_dispatch_offsets[kMaxRanks];
  // Pointers to each rank's combine recv buffer (via IPC).
  void* remote_combine_recv[kMaxRanks];
  // Pointers to each rank's combine metadata buffer (via IPC).
  void* remote_combine_meta[kMaxRanks];
  // Pointers to each rank's combine write-offset counter (via IPC).
  int32_t* remote_combine_offsets[kMaxRanks];

  int32_t rank;
  int32_t world_size;
  int32_t experts_per_rank;
  int32_t hidden_dim;
  int32_t max_num_tokens_per_rank;
};

// ====================================================================
// Dispatch P2P kernel
// ====================================================================
// For each (token, expert_slot) pair on the local rank:
//   1. Determine destination rank = expert_id / experts_per_rank
//   2. Atomically claim a write slot on the dest rank's recv buffer
//   3. Copy the token's hidden state to the remote buffer via NVLink P2P
//   4. Write metadata (source_rank, source_token_idx, expert_id, weight)
//
// Template parameter T is the hidden state dtype (e.g., __nv_bfloat16).
template <typename T>
__global__ void dispatch_p2p_kernel(
    const T* __restrict__ input,          // (M, K) local hidden states
    const int32_t* __restrict__ topk_ids, // (M, topk) expert assignments
    const float* __restrict__ topk_weights, // (M, topk) router weights
    const DispatchCombineConfig* __restrict__ config,
    int32_t M,   // number of local tokens
    int32_t K,   // hidden dimension
    int32_t topk // experts per token
) {
  const int32_t rank = config->rank;
  const int32_t experts_per_rank = config->experts_per_rank;

  // Each thread block handles one (token, expert_slot) pair.
  const int32_t pair_idx = blockIdx.x;
  if (pair_idx >= M * topk) return;

  const int32_t token_idx = pair_idx / topk;
  const int32_t expert_slot = pair_idx % topk;

  const int32_t expert_id = topk_ids[token_idx * topk + expert_slot];
  const float weight = topk_weights[token_idx * topk + expert_slot];
  const int32_t dest_rank = expert_id / experts_per_rank;

  // Atomically claim a write slot on the destination rank's recv buffer.
  int32_t write_pos = atomicAdd(
      config->remote_dispatch_offsets[dest_rank], 1);

  // P2P write: copy hidden state to remote dispatch recv buffer.
  T* dest_data = reinterpret_cast<T*>(
      config->remote_dispatch_recv[dest_rank]);
  const T* src_data = input + token_idx * K;
  // Each thread in the block copies a portion of the hidden dim.
  for (int32_t k = threadIdx.x; k < K; k += blockDim.x) {
    dest_data[write_pos * K + k] = src_data[k];
  }

  // P2P write: write metadata to remote metadata buffer.
  if (threadIdx.x == 0) {
    TokenMetadata* dest_meta = reinterpret_cast<TokenMetadata*>(
        config->remote_dispatch_meta[dest_rank]);
    dest_meta[write_pos].source_rank = rank;
    dest_meta[write_pos].source_token_idx = token_idx;
    dest_meta[write_pos].expert_id = expert_id;
    dest_meta[write_pos].topk_weight = weight;
  }

  // Memory fence to ensure all P2P writes are visible system-wide.
  __threadfence_system();
}

// ====================================================================
// Combine P2P kernel
// ====================================================================
// For each received token-expert result:
//   1. Look up the originating rank from dispatch metadata
//   2. Atomically claim a write slot on that rank's combine recv buffer
//   3. Copy the expert output to the remote buffer
//   4. Write metadata (token_idx, weight)
template <typename T>
__global__ void combine_p2p_kernel(
    const T* __restrict__ expert_output,       // (M_recv, K)
    const TokenMetadata* __restrict__ dispatch_meta, // (M_recv,)
    const DispatchCombineConfig* __restrict__ config,
    int32_t M_recv, // number of received token-expert pairs
    int32_t K       // hidden dimension
) {
  const int32_t pair_idx = blockIdx.x;
  if (pair_idx >= M_recv) return;

  // Read dispatch metadata to find where to send the result.
  const int32_t dest_rank = dispatch_meta[pair_idx].source_rank;
  const int32_t orig_token_idx = dispatch_meta[pair_idx].source_token_idx;
  const float weight = dispatch_meta[pair_idx].topk_weight;

  // Atomically claim a write slot on dest rank's combine recv buffer.
  int32_t write_pos = atomicAdd(
      config->remote_combine_offsets[dest_rank], 1);

  // P2P write: copy expert output to remote combine recv buffer.
  T* dest_data = reinterpret_cast<T*>(
      config->remote_combine_recv[dest_rank]);
  const T* src_data = expert_output + pair_idx * K;
  for (int32_t k = threadIdx.x; k < K; k += blockDim.x) {
    dest_data[write_pos * K + k] = src_data[k];
  }

  // P2P write: write combine metadata.
  if (threadIdx.x == 0) {
    TokenMetadata* dest_meta = reinterpret_cast<TokenMetadata*>(
        config->remote_combine_meta[dest_rank]);
    dest_meta[write_pos].source_rank = config->rank;
    dest_meta[write_pos].source_token_idx = orig_token_idx;
    dest_meta[write_pos].expert_id = dispatch_meta[pair_idx].expert_id;
    dest_meta[write_pos].topk_weight = weight;
  }

  __threadfence_system();
}

// ====================================================================
// Scatter-add weighted kernel
// ====================================================================
// After combine recv: for each received result, atomically add
// weight * result to the output at the original token position.
template <typename T>
__global__ void scatter_add_weighted_kernel(
    float* __restrict__ output,                      // (M_orig, K) float32
    const T* __restrict__ combine_recv,              // (N_recv, K)
    const TokenMetadata* __restrict__ combine_meta,  // (N_recv,)
    int32_t N_recv, // number of received combine entries
    int32_t K       // hidden dimension
) {
  const int32_t entry_idx = blockIdx.x;
  if (entry_idx >= N_recv) return;

  const int32_t token_idx = combine_meta[entry_idx].source_token_idx;
  const float weight = combine_meta[entry_idx].topk_weight;

  // Weighted scatter-add: output[token_idx] += weight * combine_recv[entry]
  for (int32_t k = threadIdx.x; k < K; k += blockDim.x) {
    float val = static_cast<float>(combine_recv[entry_idx * K + k]);
    val *= weight;
    atomicAdd(output + token_idx * K + k, val);
  }
}

// ====================================================================
// Host-callable wrappers
// ====================================================================
void dispatch_p2p(
    torch::Tensor input,       // (M, K) bfloat16/float16
    torch::Tensor topk_ids,    // (M, topk) int32
    torch::Tensor topk_weights,// (M, topk) float32
    torch::Tensor config_tensor, // DispatchCombineConfig as raw bytes
    int M, int K, int topk);

void combine_p2p(
    torch::Tensor expert_output,  // (M_recv, K) bfloat16/float16
    torch::Tensor dispatch_meta,  // (M_recv,) TokenMetadata as raw bytes
    torch::Tensor config_tensor,  // DispatchCombineConfig as raw bytes
    int M_recv, int K);

void scatter_add_weighted(
    torch::Tensor output,       // (M_orig, K) float32
    torch::Tensor combine_recv, // (N_recv, K) bfloat16/float16
    torch::Tensor combine_meta, // (N_recv,) TokenMetadata as raw bytes
    int N_recv, int K);

}  // namespace dispatch_combine
}  // namespace vllm
