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
  // Grid = max_recv; kernel reads actual count from config.
  dim3 grid(static_cast<int32_t>(max_recv));
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

void scatter_add_weighted(
    torch::Tensor output,
    torch::Tensor combine_recv,
    torch::Tensor combine_meta,
    int64_t N_recv, int64_t K) {

  if (N_recv == 0) return;

  const auto stream = at::cuda::getCurrentCUDAStream();
  const TokenMetadata* meta =
      reinterpret_cast<const TokenMetadata*>(
          combine_meta.data_ptr());

  const int32_t N_recv32 = static_cast<int32_t>(N_recv);
  const int32_t K32 = static_cast<int32_t>(K);
  dim3 grid(N_recv32);
  dim3 block(kBlockSize);

  TORCH_CHECK(
      output.scalar_type() == at::ScalarType::Float,
      "scatter_add_weighted: output must be float32");

  AT_DISPATCH_SWITCH(
      combine_recv.scalar_type(), "scatter_add_weighted",
      AT_DISPATCH_CASE(at::ScalarType::BFloat16,
        [&] {
          scatter_add_weighted_kernel<__nv_bfloat16>
              <<<grid, block, 0, stream>>>(
              output.data_ptr<float>(),
              reinterpret_cast<const __nv_bfloat16*>(
                  combine_recv.data_ptr()),
              meta, N_recv32, K32);
        })
      AT_DISPATCH_CASE(at::ScalarType::Half,
        [&] {
          scatter_add_weighted_kernel<__half>
              <<<grid, block, 0, stream>>>(
              output.data_ptr<float>(),
              reinterpret_cast<const __half*>(
                  combine_recv.data_ptr()),
              meta, N_recv32, K32);
        })
  );
}

// ====================================================================
// GPU-side buffer operations (CUDA-graph compatible)
// ====================================================================

void reset_offsets(torch::Tensor config_tensor) {
  const auto stream = at::cuda::getCurrentCUDAStream();
  const DispatchCombineConfig* config =
      reinterpret_cast<const DispatchCombineConfig*>(
          config_tensor.data_ptr());
  reset_offsets_kernel<<<1, 1, 0, stream>>>(config);
}

void reset_combine_offset(torch::Tensor config_tensor) {
  const auto stream = at::cuda::getCurrentCUDAStream();
  const DispatchCombineConfig* config =
      reinterpret_cast<const DispatchCombineConfig*>(
          config_tensor.data_ptr());
  reset_combine_offset_kernel<<<1, 1, 0, stream>>>(
      config);
}

void copy_dispatch_recv(
    torch::Tensor output,
    torch::Tensor config_tensor,
    int64_t max_recv, int64_t K) {
  const auto stream = at::cuda::getCurrentCUDAStream();
  const DispatchCombineConfig* config =
      reinterpret_cast<const DispatchCombineConfig*>(
          config_tensor.data_ptr());
  const int32_t K32 = static_cast<int32_t>(K);
  dim3 grid(static_cast<int32_t>(max_recv));
  dim3 block(kBlockSize);

  AT_DISPATCH_SWITCH(
      output.scalar_type(), "copy_dispatch_recv",
      AT_DISPATCH_CASE(at::ScalarType::BFloat16,
        [&] {
          copy_dispatch_recv_kernel<__nv_bfloat16>
              <<<grid, block, 0, stream>>>(
              reinterpret_cast<__nv_bfloat16*>(
                  output.data_ptr()),
              config, K32);
        })
      AT_DISPATCH_CASE(at::ScalarType::Half,
        [&] {
          copy_dispatch_recv_kernel<__half>
              <<<grid, block, 0, stream>>>(
              reinterpret_cast<__half*>(
                  output.data_ptr()),
              config, K32);
        })
  );
}

void copy_dispatch_meta(
    torch::Tensor output,
    torch::Tensor config_tensor,
    int64_t max_recv) {
  const auto stream = at::cuda::getCurrentCUDAStream();
  const DispatchCombineConfig* config =
      reinterpret_cast<const DispatchCombineConfig*>(
          config_tensor.data_ptr());
  dim3 grid(static_cast<int32_t>(max_recv));
  // 1 thread per block for metadata (only 4 int32s).
  dim3 block(1);
  copy_dispatch_meta_kernel<<<grid, block, 0, stream>>>(
      output.data_ptr<int32_t>(), config);
}

void copy_combine_recv(
    torch::Tensor output,
    torch::Tensor config_tensor,
    int64_t max_recv, int64_t K) {
  const auto stream = at::cuda::getCurrentCUDAStream();
  const DispatchCombineConfig* config =
      reinterpret_cast<const DispatchCombineConfig*>(
          config_tensor.data_ptr());
  const int32_t K32 = static_cast<int32_t>(K);
  dim3 grid(static_cast<int32_t>(max_recv));
  dim3 block(kBlockSize);

  AT_DISPATCH_SWITCH(
      output.scalar_type(), "copy_combine_recv",
      AT_DISPATCH_CASE(at::ScalarType::BFloat16,
        [&] {
          copy_combine_recv_kernel<__nv_bfloat16>
              <<<grid, block, 0, stream>>>(
              reinterpret_cast<__nv_bfloat16*>(
                  output.data_ptr()),
              config, K32);
        })
      AT_DISPATCH_CASE(at::ScalarType::Half,
        [&] {
          copy_combine_recv_kernel<__half>
              <<<grid, block, 0, stream>>>(
              reinterpret_cast<__half*>(
                  output.data_ptr()),
              config, K32);
        })
  );
}

void copy_combine_meta(
    torch::Tensor output,
    torch::Tensor config_tensor,
    int64_t max_recv) {
  const auto stream = at::cuda::getCurrentCUDAStream();
  const DispatchCombineConfig* config =
      reinterpret_cast<const DispatchCombineConfig*>(
          config_tensor.data_ptr());
  dim3 grid(static_cast<int32_t>(max_recv));
  dim3 block(1);
  copy_combine_meta_kernel<<<grid, block, 0, stream>>>(
      output.data_ptr<int32_t>(), config);
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

void p2p_barrier_reset_offsets(
    torch::Tensor config_tensor) {
  const auto stream = at::cuda::getCurrentCUDAStream();
  const DispatchCombineConfig* config =
      reinterpret_cast<const DispatchCombineConfig*>(
          config_tensor.data_ptr());
  p2p_barrier_kernel<BarrierMode::RESET_DISPATCH_COMBINE>
      <<<1, kMaxRanks, 0, stream>>>(config);
}

void p2p_barrier_reset_combine_offset(
    torch::Tensor config_tensor) {
  const auto stream = at::cuda::getCurrentCUDAStream();
  const DispatchCombineConfig* config =
      reinterpret_cast<const DispatchCombineConfig*>(
          config_tensor.data_ptr());
  p2p_barrier_kernel<BarrierMode::RESET_COMBINE>
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
// Fused kernels (eliminate copy overhead)
// ====================================================================

void stamp_and_zero_dispatch(
    torch::Tensor dispatch_recv,
    torch::Tensor config_tensor,
    int64_t mc, int64_t K) {

  if (mc == 0) return;

  const auto stream = at::cuda::getCurrentCUDAStream();
  const DispatchCombineConfig* config =
      reinterpret_cast<const DispatchCombineConfig*>(
          config_tensor.data_ptr());

  const int32_t mc32 = static_cast<int32_t>(mc);
  const int32_t K32 = static_cast<int32_t>(K);
  dim3 grid(mc32);
  dim3 block(kBlockSize);

  AT_DISPATCH_SWITCH(
      dispatch_recv.scalar_type(),
      "stamp_and_zero_dispatch",
      AT_DISPATCH_CASE(at::ScalarType::BFloat16,
        [&] {
          stamp_and_zero_dispatch_kernel<__nv_bfloat16>
              <<<grid, block, 0, stream>>>(
              reinterpret_cast<__nv_bfloat16*>(
                  dispatch_recv.data_ptr()),
              config, mc32, K32);
        })
      AT_DISPATCH_CASE(at::ScalarType::Half,
        [&] {
          stamp_and_zero_dispatch_kernel<__half>
              <<<grid, block, 0, stream>>>(
              reinterpret_cast<__half*>(
                  dispatch_recv.data_ptr()),
              config, mc32, K32);
        })
  );
}

void scatter_add_v2(
    torch::Tensor output,
    torch::Tensor config_tensor,
    int64_t mc, int64_t K,
    int64_t dtype_code) {

  if (mc == 0) return;

  const auto stream = at::cuda::getCurrentCUDAStream();
  const DispatchCombineConfig* config =
      reinterpret_cast<const DispatchCombineConfig*>(
          config_tensor.data_ptr());

  TORCH_CHECK(
      output.scalar_type() == at::ScalarType::Float,
      "scatter_add_v2: output must be float32");

  const int32_t mc32 = static_cast<int32_t>(mc);
  const int32_t K32 = static_cast<int32_t>(K);
  dim3 grid(mc32);
  dim3 block(kBlockSize);

  // dtype_code: 0=bf16, 1=fp16
  if (dtype_code == 0) {
    scatter_add_v2_kernel<__nv_bfloat16>
        <<<grid, block, 0, stream>>>(
        output.data_ptr<float>(),
        config, mc32, K32);
  } else {
    scatter_add_v2_kernel<__half>
        <<<grid, block, 0, stream>>>(
        output.data_ptr<float>(),
        config, mc32, K32);
  }
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

}  // namespace dispatch_combine
}  // namespace vllm
