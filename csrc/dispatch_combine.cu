// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include "dispatch_combine.cuh"

namespace vllm {
namespace dispatch_combine {

// Thread block size for P2P copy kernels.
constexpr int kBlockSize = 256;

void dispatch_p2p(
    torch::Tensor input,
    torch::Tensor topk_ids,
    torch::Tensor topk_weights,
    torch::Tensor config_tensor,
    int64_t M, int64_t K, int64_t topk) {

  if (M == 0) return;

  const auto stream = at::cuda::getCurrentCUDAStream();
  const DispatchCombineConfig* config =
      reinterpret_cast<const DispatchCombineConfig*>(
          config_tensor.data_ptr());

  const int32_t M32 = static_cast<int32_t>(M);
  const int32_t K32 = static_cast<int32_t>(K);
  const int32_t topk32 = static_cast<int32_t>(topk);
  int num_pairs = M32 * topk32;
  dim3 grid(num_pairs);
  dim3 block(kBlockSize);

  AT_DISPATCH_SWITCH(
      input.scalar_type(), "dispatch_p2p",
      AT_DISPATCH_CASE(at::ScalarType::BFloat16,
        [&] {
          dispatch_p2p_kernel<__nv_bfloat16>
              <<<grid, block, 0, stream>>>(
              reinterpret_cast<const __nv_bfloat16*>(
                  input.data_ptr()),
              topk_ids.data_ptr<int32_t>(),
              topk_weights.data_ptr<float>(),
              config, M32, K32, topk32);
        })
      AT_DISPATCH_CASE(at::ScalarType::Half,
        [&] {
          dispatch_p2p_kernel<__half>
              <<<grid, block, 0, stream>>>(
              reinterpret_cast<const __half*>(
                  input.data_ptr()),
              topk_ids.data_ptr<int32_t>(),
              topk_weights.data_ptr<float>(),
              config, M32, K32, topk32);
        })
  );
}

void combine_p2p(
    torch::Tensor expert_output,
    torch::Tensor dispatch_meta,
    torch::Tensor config_tensor,
    int64_t max_recv, int64_t K) {

  if (max_recv == 0) return;

  const auto stream = at::cuda::getCurrentCUDAStream();
  const DispatchCombineConfig* config =
      reinterpret_cast<const DispatchCombineConfig*>(
          config_tensor.data_ptr());
  const TokenMetadata* meta =
      reinterpret_cast<const TokenMetadata*>(
          dispatch_meta.data_ptr());

  const int32_t K32 = static_cast<int32_t>(K);
  int32_t grid_sz = static_cast<int32_t>(max_recv);
  if (grid_sz > kPersistentGrid)
    grid_sz = kPersistentGrid;
  if (grid_sz < 1) grid_sz = 1;
  dim3 grid(grid_sz);
  dim3 block(kBlockSize);

  AT_DISPATCH_SWITCH(
      expert_output.scalar_type(), "combine_p2p",
      AT_DISPATCH_CASE(at::ScalarType::BFloat16,
        [&] {
          combine_p2p_kernel<__nv_bfloat16>
              <<<grid, block, 0, stream>>>(
              reinterpret_cast<const __nv_bfloat16*>(
                  expert_output.data_ptr()),
              meta, config, K32);
        })
      AT_DISPATCH_CASE(at::ScalarType::Half,
        [&] {
          combine_p2p_kernel<__half>
              <<<grid, block, 0, stream>>>(
              reinterpret_cast<const __half*>(
                  expert_output.data_ptr()),
              meta, config, K32);
        })
  );
}

// ====================================================================
// P2P flag-based barriers (replace NCCL AllReduce)
// ====================================================================

void p2p_barrier(torch::Tensor config_tensor) {
  const auto stream = at::cuda::getCurrentCUDAStream();
  const DispatchCombineConfig* config =
      reinterpret_cast<const DispatchCombineConfig*>(
          config_tensor.data_ptr());
  p2p_barrier_kernel<BarrierMode::PURE>
      <<<1, kMaxRanks, 0, stream>>>(config);
}

void p2p_barrier_reset_dispatch(
    torch::Tensor config_tensor) {
  const auto stream = at::cuda::getCurrentCUDAStream();
  const DispatchCombineConfig* config =
      reinterpret_cast<const DispatchCombineConfig*>(
          config_tensor.data_ptr());
  p2p_barrier_kernel<BarrierMode::RESET_DISPATCH>
      <<<1, kMaxRanks, 0, stream>>>(config);
}

// ====================================================================
// Fused kernels
// ====================================================================

void prepare_dispatch_recv(
    torch::Tensor dispatch_recv,
    torch::Tensor expert_topk_ids,
    torch::Tensor expert_topk_weights,
    torch::Tensor expert_num_tokens,
    torch::Tensor config_tensor,
    int64_t mc, int64_t K,
    int64_t num_experts) {

  if (mc == 0) return;

  const auto stream = at::cuda::getCurrentCUDAStream();
  const DispatchCombineConfig* config =
      reinterpret_cast<const DispatchCombineConfig*>(
          config_tensor.data_ptr());

  // Zero expert_num_tokens before kernel launch
  // (atomicAdd needs zeroed counters).
  cudaMemsetAsync(
      expert_num_tokens.data_ptr(), 0,
      num_experts * sizeof(int32_t), stream);

  const int32_t mc32 = static_cast<int32_t>(mc);
  const int32_t K32 = static_cast<int32_t>(K);
  const int32_t ne32 =
      static_cast<int32_t>(num_experts);
  // Persistent grid; kernel has inline barrier +
  // loops over entries.
  int32_t grid_sz = mc32;
  if (grid_sz > kPersistentGrid)
    grid_sz = kPersistentGrid;
  if (grid_sz < 1) grid_sz = 1;
  dim3 grid(grid_sz);
  dim3 block(kBlockSize);

  AT_DISPATCH_SWITCH(
      dispatch_recv.scalar_type(),
      "prepare_dispatch_recv",
      AT_DISPATCH_CASE(at::ScalarType::BFloat16,
        [&] {
          prepare_dispatch_recv_kernel<__nv_bfloat16>
              <<<grid, block, 0, stream>>>(
              reinterpret_cast<__nv_bfloat16*>(
                  dispatch_recv.data_ptr()),
              expert_topk_ids.data_ptr<int64_t>(),
              expert_topk_weights.data_ptr<float>(),
              expert_num_tokens.data_ptr<int32_t>(),
              config, mc32, K32, ne32);
        })
      AT_DISPATCH_CASE(at::ScalarType::Half,
        [&] {
          prepare_dispatch_recv_kernel<__half>
              <<<grid, block, 0, stream>>>(
              reinterpret_cast<__half*>(
                  dispatch_recv.data_ptr()),
              expert_topk_ids.data_ptr<int64_t>(),
              expert_topk_weights.data_ptr<float>(),
              expert_num_tokens.data_ptr<int32_t>(),
              config, mc32, K32, ne32);
        })
  );
}

void scatter_add_direct(
    torch::Tensor output,
    torch::Tensor config_tensor,
    int64_t mc, int64_t K,
    int64_t M) {

  if (mc == 0 || M == 0) return;

  const auto stream = at::cuda::getCurrentCUDAStream();
  const DispatchCombineConfig* config =
      reinterpret_cast<const DispatchCombineConfig*>(
          config_tensor.data_ptr());

  // Zero output before scatter-add (atomicAdd).
  cudaMemsetAsync(
      output.data_ptr(), 0,
      M * K * output.element_size(), stream);

  const int32_t mc32 = static_cast<int32_t>(mc);
  const int32_t K32 = static_cast<int32_t>(K);
  int32_t grid_sz = mc32;
  if (grid_sz < 1) grid_sz = 1;
  dim3 grid(grid_sz);
  dim3 block(kBlockSize);

  AT_DISPATCH_SWITCH(
      output.scalar_type(),
      "scatter_add_direct",
      AT_DISPATCH_CASE(at::ScalarType::BFloat16,
        [&] {
          scatter_add_direct_kernel<__nv_bfloat16>
              <<<grid, block, 0, stream>>>(
              reinterpret_cast<__nv_bfloat16*>(
                  output.data_ptr()),
              config, mc32, K32);
        })
      AT_DISPATCH_CASE(at::ScalarType::Half,
        [&] {
          scatter_add_direct_kernel<__half>
              <<<grid, block, 0, stream>>>(
              reinterpret_cast<__half*>(
                  output.data_ptr()),
              config, mc32, K32);
        })
  );
}

torch::Tensor wrap_cuda_ptr(
    torch::Tensor dummy,
    int64_t ptr, int64_t dim0, int64_t dim1,
    int64_t dtype_code) {
  // dtype_code: 0=bf16, 1=fp16, 2=int32
  at::ScalarType dtype;
  switch (dtype_code) {
    case 0: dtype = at::ScalarType::BFloat16; break;
    case 1: dtype = at::ScalarType::Half; break;
    case 2: dtype = at::ScalarType::Int; break;
    default:
      TORCH_CHECK(false,
          "wrap_cuda_ptr: unsupported dtype_code=",
          dtype_code);
  }
  auto options = torch::TensorOptions()
      .dtype(dtype)
      .device(dummy.device());
  return torch::from_blob(
      reinterpret_cast<void*>(ptr),
      {dim0, dim1}, options);
}

// ====================================================================
// Fused dispatch + route + filter (integrated EPLB)
// ====================================================================

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
    int64_t world_size) {

  if (M == 0) return;

  const auto stream = at::cuda::getCurrentCUDAStream();
  DispatchCombineConfig* config =
      reinterpret_cast<DispatchCombineConfig*>(
          config_tensor.data_ptr());

  const int32_t M32 = static_cast<int32_t>(M);
  const int32_t K32 = static_cast<int32_t>(K);
  const int32_t topk32 = static_cast<int32_t>(topk);
  const int32_t mc32 = static_cast<int32_t>(mc);
  const int32_t ne32 =
      static_cast<int32_t>(num_physical_experts);
  const int32_t NL =
      static_cast<int32_t>(num_logical_experts);
  const int32_t ws =
      static_cast<int32_t>(world_size);

  // Zero expert_num_tokens (atomicAdd target).
  cudaMemsetAsync(
      expert_num_tokens.data_ptr(), 0,
      num_physical_experts * sizeof(int32_t), stream);

  // NOTE: expert_counts is NOT zeroed here. It is zeroed
  // at the end of the kernel (Phase E). A host-side
  // cudaMemsetAsync would race with remote ranks'
  // Phase A allgather writes to this IPC buffer.

  // Shared memory: max of Phase A and Phase C needs.
  // Phase A: NL + 2*ws + 3*64 ints (expert counts +
  //   grouping arrays).
  // Phase C: 3*NL + NL*kMaxRep + ws ints
  //   (s_expert_sum[NL] + s_replica_count[NL]
  //    + s_l2p_map[NL*kMaxRep] + routing_sel[NL]
  //    + rank_active[ws]).
  // Phases don't overlap, so same memory is reused.
  constexpr int32_t kMaxEntries = 64;
  constexpr int32_t kMaxRep = 2;
  size_t phase_a_bytes = static_cast<size_t>(
      (NL + 2 * ws + 3 * kMaxEntries)
      * sizeof(int32_t));
  size_t phase_c_bytes = static_cast<size_t>(
      (3 * NL + NL * kMaxRep + ws)
      * sizeof(int32_t));
  size_t shared_bytes = phase_a_bytes > phase_c_bytes
      ? phase_a_bytes : phase_c_bytes;

  int32_t grid_sz = mc32;
  if (grid_sz > kPersistentGrid)
    grid_sz = kPersistentGrid;
  if (grid_sz < 1) grid_sz = 1;
  dim3 grid(grid_sz);
  dim3 block(kBlockSize);

  AT_DISPATCH_SWITCH(
      input.scalar_type(), "dispatch_and_route",
      AT_DISPATCH_CASE(at::ScalarType::BFloat16,
        [&] {
          dispatch_and_route_kernel<__nv_bfloat16>
              <<<grid, block, shared_bytes, stream>>>(
              reinterpret_cast<const __nv_bfloat16*>(
                  input.data_ptr()),
              topk_ids.data_ptr<int32_t>(),
              topk_weights.data_ptr<float>(),
              reinterpret_cast<__nv_bfloat16*>(
                  dispatch_recv.data_ptr()),
              expert_topk_ids.data_ptr<int64_t>(),
              expert_topk_weights.data_ptr<float>(),
              expert_num_tokens.data_ptr<int32_t>(),
              data_remap.data_ptr<int32_t>(),
              config, M32, K32, topk32,
              mc32, ne32);
        })
      AT_DISPATCH_CASE(at::ScalarType::Half,
        [&] {
          dispatch_and_route_kernel<__half>
              <<<grid, block, shared_bytes, stream>>>(
              reinterpret_cast<const __half*>(
                  input.data_ptr()),
              topk_ids.data_ptr<int32_t>(),
              topk_weights.data_ptr<float>(),
              reinterpret_cast<__half*>(
                  dispatch_recv.data_ptr()),
              expert_topk_ids.data_ptr<int64_t>(),
              expert_topk_weights.data_ptr<float>(),
              expert_num_tokens.data_ptr<int32_t>(),
              data_remap.data_ptr<int32_t>(),
              config, M32, K32, topk32,
              mc32, ne32);
        })
  );
}

}  // namespace dispatch_combine
}  // namespace vllm
