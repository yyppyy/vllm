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
  // Persistent grid: per-token processing with dedup.
  int32_t grid_sz = M32;
  if (grid_sz > kPersistentGrid)
    grid_sz = kPersistentGrid;
  if (grid_sz < 1) grid_sz = 1;
  dim3 grid(grid_sz);
  dim3 block(kBlockSize);
  // Dynamic shared memory: ws*4 + 3*kMaxEntries*4.
  constexpr int kMaxEntries = 64;
  int smem = kMaxRanks * 4 + 3 * kMaxEntries * 4;

  AT_DISPATCH_SWITCH(
      input.scalar_type(), "dispatch_p2p",
      AT_DISPATCH_CASE(at::ScalarType::BFloat16,
        [&] {
          dispatch_p2p_kernel<__nv_bfloat16>
              <<<grid, block, smem, stream>>>(
              reinterpret_cast<const __nv_bfloat16*>(
                  input.data_ptr()),
              topk_ids.data_ptr<int32_t>(),
              topk_weights.data_ptr<float>(),
              config, M32, K32, topk32);
        })
      AT_DISPATCH_CASE(at::ScalarType::Half,
        [&] {
          dispatch_p2p_kernel<__half>
              <<<grid, block, smem, stream>>>(
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
    torch::Tensor compact_reverse,
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
  const int32_t* cr =
      compact_reverse.data_ptr<int32_t>();

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
              meta, cr, config, K32);
        })
      AT_DISPATCH_CASE(at::ScalarType::Half,
        [&] {
          combine_p2p_kernel<__half>
              <<<grid, block, 0, stream>>>(
              reinterpret_cast<const __half*>(
                  expert_output.data_ptr()),
              meta, cr, config, K32);
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
    torch::Tensor data_remap,
    torch::Tensor config_tensor,
    int64_t mc, int64_t K,
    int64_t num_experts) {

  if (mc == 0) return;

  const auto stream = at::cuda::getCurrentCUDAStream();
  const DispatchCombineConfig* config =
      reinterpret_cast<const DispatchCombineConfig*>(
          config_tensor.data_ptr());

  // expert_num_tokens zeroed inline by kernel (block 0
  // after barrier, before signaling other blocks).

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
              data_remap.data_ptr<int32_t>(),
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
              data_remap.data_ptr<int32_t>(),
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

  const int32_t K32 = static_cast<int32_t>(K);
  const int32_t N = static_cast<int32_t>(M * K);

  // This kernel has no grid-wide sync — use enough
  // blocks to saturate all SMs for HBM latency hiding.
  // 512 blocks / 108 SMs ≈ 4-5 blocks/SM = 32-40
  // warps/SM, close to the ~42 needed for peak BW.
  constexpr int32_t kScatterGrid = 512;
  int32_t grid_sz = static_cast<int32_t>(mc);
  if (grid_sz > kScatterGrid)
    grid_sz = kScatterGrid;
  if (grid_sz < 1) grid_sz = 1;

  // fp32 accumulation buffer — native fp32 atomicAdd
  // avoids bf16 CAS loops and adjacent-element
  // contention from paired 32-bit words.
  auto accum = torch::zeros(
      {M, K}, output.options().dtype(at::kFloat));

  AT_DISPATCH_SWITCH(
      output.scalar_type(),
      "scatter_add_direct",
      AT_DISPATCH_CASE(at::ScalarType::BFloat16,
        [&] {
          scatter_add_atomic_kernel<__nv_bfloat16>
              <<<grid_sz, kBlockSize, 0, stream>>>(
              accum.data_ptr<float>(),
              config, K32);
          fp32_to_half_kernel<__nv_bfloat16>
              <<<(N + kBlockSize - 1) / kBlockSize,
                 kBlockSize, 0, stream>>>(
              reinterpret_cast<__nv_bfloat16*>(
                  output.data_ptr()),
              accum.data_ptr<float>(), N);
        })
      AT_DISPATCH_CASE(at::ScalarType::Half,
        [&] {
          scatter_add_atomic_kernel<__half>
              <<<grid_sz, kBlockSize, 0, stream>>>(
              accum.data_ptr<float>(),
              config, K32);
          fp32_to_half_kernel<__half>
              <<<(N + kBlockSize - 1) / kBlockSize,
                 kBlockSize, 0, stream>>>(
              reinterpret_cast<__half*>(
                  output.data_ptr()),
              accum.data_ptr<float>(), N);
        })
  );
}

void combine_and_scatter(
    torch::Tensor expert_output,
    torch::Tensor dispatch_meta,
    torch::Tensor compact_reverse,
    torch::Tensor output,
    torch::Tensor config_tensor,
    int64_t mc, int64_t K,
    int64_t M) {

  if (mc == 0 || M == 0) return;

  const auto stream = at::cuda::getCurrentCUDAStream();
  const DispatchCombineConfig* config =
      reinterpret_cast<const DispatchCombineConfig*>(
          config_tensor.data_ptr());
  const TokenMetadata* meta =
      reinterpret_cast<const TokenMetadata*>(
          dispatch_meta.data_ptr());
  const int32_t* cr =
      compact_reverse.data_ptr<int32_t>();

  const int32_t mc32 = static_cast<int32_t>(mc);
  const int32_t K32 = static_cast<int32_t>(K);
  const int32_t M32 = static_cast<int32_t>(M);
  const int32_t N = M32 * K32;

  // No grid-wide sync needed for scatter-add phase —
  // use enough blocks to saturate all SMs.
  constexpr int32_t kCombineScatterGrid = kPersistentGrid;
  int32_t grid_sz = mc32;
  if (grid_sz > kCombineScatterGrid)
    grid_sz = kCombineScatterGrid;
  if (grid_sz < 1) grid_sz = 1;
  dim3 grid(grid_sz);
  dim3 block(kBlockSize);

  // fp32 accum buffer — zeroed inline by kernel Phase 0.
  // Native fp32 atomicAdd avoids bf16 CAS loops and
  // adjacent-element contention. Converted to output
  // dtype by fp32_to_half_kernel after the fused kernel.
  auto accum = torch::empty(
      {M, K}, output.options().dtype(at::kFloat));

  AT_DISPATCH_SWITCH(
      output.scalar_type(),
      "combine_and_scatter",
      AT_DISPATCH_CASE(at::ScalarType::BFloat16,
        [&] {
          combine_and_scatter_kernel<__nv_bfloat16>
              <<<grid, block,
                 kCasMaxUnique * K32 * sizeof(float),
                 stream>>>(
              reinterpret_cast<const __nv_bfloat16*>(
                  expert_output.data_ptr()),
              meta, cr,
              accum.data_ptr<float>(),
              config, mc32, K32, M32);
          fp32_to_half_kernel<__nv_bfloat16>
              <<<(N + kBlockSize - 1) / kBlockSize,
                 kBlockSize, 0, stream>>>(
              reinterpret_cast<__nv_bfloat16*>(
                  output.data_ptr()),
              accum.data_ptr<float>(), N);
        })
      AT_DISPATCH_CASE(at::ScalarType::Half,
        [&] {
          combine_and_scatter_kernel<__half>
              <<<grid, block,
                 kCasMaxUnique * K32 * sizeof(float),
                 stream>>>(
              reinterpret_cast<const __half*>(
                  expert_output.data_ptr()),
              meta, cr,
              accum.data_ptr<float>(),
              config, mc32, K32, M32);
          fp32_to_half_kernel<__half>
              <<<(N + kBlockSize - 1) / kBlockSize,
                 kBlockSize, 0, stream>>>(
              reinterpret_cast<__half*>(
                  output.data_ptr()),
              accum.data_ptr<float>(), N);
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
    int64_t world_size,
    int64_t max_replicas) {

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
  const int32_t mr =
      static_cast<int32_t>(max_replicas);

  // expert_num_tokens zeroed inline by kernel (Phase C,
  // block 0, before routing_ready_flag signal).

  // NOTE: expert_counts is NOT zeroed here. It is zeroed
  // at the end of the kernel (Phase E). A host-side
  // cudaMemsetAsync would race with remote ranks'
  // Phase A allgather writes to this IPC buffer.

  // Shared memory: max of scan_write and Phase C needs.
  // Scan_write: NL + ws + 3*64 + NL + NL*mr ints
  //   (expert_counts + grp_count + entries
  //    + preloaded replica_count + l2p_map).
  // Phase C: 3*NL + NL*mr + ws ints
  //   (s_expert_sum[NL] + s_replica_count[NL]
  //    + s_l2p_map[NL*mr] + routing_sel[NL]
  //    + rank_active[ws]).
  // Phases don't overlap, so same memory is reused.
  constexpr int32_t kMaxEntries = 64;
  size_t phase_a_bytes = static_cast<size_t>(
      (2 * NL + ws + 3 * kMaxEntries
       + NL * mr)
      * sizeof(int32_t));
  size_t phase_c_bytes = static_cast<size_t>(
      (3 * NL + NL * mr + ws + NL + 1)
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

void dar_compact(
    torch::Tensor expert_topk_ids,
    torch::Tensor expert_topk_weights,
    torch::Tensor data_remap,
    torch::Tensor compact_expert_topk_ids,
    torch::Tensor compact_expert_topk_weights,
    torch::Tensor compact_data_remap,
    torch::Tensor compact_reverse,
    torch::Tensor config_tensor,
    int64_t mc_compact,
    int64_t num_physical_experts) {

  if (mc_compact == 0) return;

  const auto stream = at::cuda::getCurrentCUDAStream();
  const DispatchCombineConfig* config =
      reinterpret_cast<const DispatchCombineConfig*>(
          config_tensor.data_ptr());

  const int32_t mc32 =
      static_cast<int32_t>(mc_compact);
  const int32_t ne32 =
      static_cast<int32_t>(num_physical_experts);

  dim3 grid(kPersistentGrid);
  dim3 block(kBlockSize);
  dar_compact_kernel
      <<<grid, block, 0, stream>>>(
      expert_topk_ids.data_ptr<int64_t>(),
      expert_topk_weights.data_ptr<float>(),
      data_remap.data_ptr<int32_t>(),
      compact_expert_topk_ids.data_ptr<int64_t>(),
      compact_expert_topk_weights.data_ptr<float>(),
      compact_data_remap.data_ptr<int32_t>(),
      compact_reverse.data_ptr<int32_t>(),
      config, mc32, ne32);
}

}  // namespace dispatch_combine
}  // namespace vllm
