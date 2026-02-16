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
    torch::Tensor input,        // (M, K) bfloat16/float16
    torch::Tensor topk_ids,     // (M, topk) int32
    torch::Tensor topk_weights, // (M, topk) float32
    torch::Tensor config_tensor,// DispatchCombineConfig as raw bytes
    int M, int K, int topk) {

  if (M == 0) return;

  const auto stream = at::cuda::getCurrentCUDAStream();
  const DispatchCombineConfig* config =
      reinterpret_cast<const DispatchCombineConfig*>(
          config_tensor.data_ptr());

  int num_pairs = M * topk;
  dim3 grid(num_pairs);
  dim3 block(kBlockSize);

  AT_DISPATCH_SWITCH(
      input.scalar_type(), "dispatch_p2p",
      AT_DISPATCH_CASE(at::ScalarType::BFloat16,
        [&] {
          dispatch_p2p_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
              reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
              topk_ids.data_ptr<int32_t>(),
              topk_weights.data_ptr<float>(),
              config,
              M, K, topk);
        })
      AT_DISPATCH_CASE(at::ScalarType::Half,
        [&] {
          dispatch_p2p_kernel<__half><<<grid, block, 0, stream>>>(
              reinterpret_cast<const __half*>(input.data_ptr()),
              topk_ids.data_ptr<int32_t>(),
              topk_weights.data_ptr<float>(),
              config,
              M, K, topk);
        })
  );
}

void combine_p2p(
    torch::Tensor expert_output,  // (M_recv, K) bfloat16/float16
    torch::Tensor dispatch_meta,  // (M_recv * sizeof(TokenMetadata),)
    torch::Tensor config_tensor,  // DispatchCombineConfig as raw bytes
    int M_recv, int K) {

  if (M_recv == 0) return;

  const auto stream = at::cuda::getCurrentCUDAStream();
  const DispatchCombineConfig* config =
      reinterpret_cast<const DispatchCombineConfig*>(
          config_tensor.data_ptr());
  const TokenMetadata* meta =
      reinterpret_cast<const TokenMetadata*>(dispatch_meta.data_ptr());

  dim3 grid(M_recv);
  dim3 block(kBlockSize);

  AT_DISPATCH_SWITCH(
      expert_output.scalar_type(), "combine_p2p",
      AT_DISPATCH_CASE(at::ScalarType::BFloat16,
        [&] {
          combine_p2p_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
              reinterpret_cast<const __nv_bfloat16*>(
                  expert_output.data_ptr()),
              meta, config, M_recv, K);
        })
      AT_DISPATCH_CASE(at::ScalarType::Half,
        [&] {
          combine_p2p_kernel<__half><<<grid, block, 0, stream>>>(
              reinterpret_cast<const __half*>(expert_output.data_ptr()),
              meta, config, M_recv, K);
        })
  );
}

void scatter_add_weighted(
    torch::Tensor output,       // (M_orig, K) float32
    torch::Tensor combine_recv, // (N_recv, K) bfloat16/float16
    torch::Tensor combine_meta, // (N_recv * sizeof(TokenMetadata),)
    int N_recv, int K) {

  if (N_recv == 0) return;

  const auto stream = at::cuda::getCurrentCUDAStream();
  const TokenMetadata* meta =
      reinterpret_cast<const TokenMetadata*>(combine_meta.data_ptr());

  dim3 grid(N_recv);
  dim3 block(kBlockSize);

  // Output must be float32 for correct atomic accumulation.
  TORCH_CHECK(output.scalar_type() == at::ScalarType::Float,
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
              meta, N_recv, K);
        })
      AT_DISPATCH_CASE(at::ScalarType::Half,
        [&] {
          scatter_add_weighted_kernel<__half><<<grid, block, 0, stream>>>(
              output.data_ptr<float>(),
              reinterpret_cast<const __half*>(combine_recv.data_ptr()),
              meta, N_recv, K);
        })
  );
}

}  // namespace dispatch_combine
}  // namespace vllm
