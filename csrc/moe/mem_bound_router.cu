#include "moe_ops.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include <limits>

#ifndef USE_ROCM

namespace {

__global__ void mem_bound_router_greedy_kernel(
    const int32_t* __restrict__ logical_ids,
    int32_t* __restrict__ out_physical_ids,
    const int32_t* __restrict__ logical_to_physical,
    const int64_t* __restrict__ logical_replica_count,
    int32_t* __restrict__ physical_token_counts,
    int32_t* __restrict__ rank_active_counts,
    uint8_t* __restrict__ physical_active_flags,
    int64_t num_pairs,
    int64_t num_logical_experts,
    int64_t slots_per_logical,
    int32_t physical_experts_per_rank,
    int32_t ep_size,
    int64_t physical_capacity,
    int64_t rank_capacity) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  for (int64_t idx = 0; idx < num_pairs; ++idx) {
    int32_t logical_id = logical_ids[idx];
    if (logical_id < 0 || logical_id >= num_logical_experts) {
      out_physical_ids[idx] = -1;
      continue;
    }

    int64_t replica_count = logical_replica_count[logical_id];
    if (replica_count <= 0) {
      out_physical_ids[idx] = -1;
      continue;
    }
    if (replica_count > slots_per_logical) {
      replica_count = slots_per_logical;
    }

    int32_t best_physical = -1;
    int32_t best_rank = 0;
    int32_t best_rank_cost = std::numeric_limits<int32_t>::max();
    int32_t best_token_cost = std::numeric_limits<int32_t>::max();

    for (int64_t slot = 0; slot < replica_count; ++slot) {
      const int32_t physical_id =
          logical_to_physical[logical_id * slots_per_logical + slot];
      if (physical_id < 0 || physical_id >= physical_capacity) {
        continue;
      }

      int32_t rank = 0;
      if (physical_experts_per_rank > 0) {
        rank = physical_id / physical_experts_per_rank;
      }
      if (rank >= ep_size) {
        rank = ep_size - 1;
      }
      if (rank < 0 || rank >= rank_capacity) {
        continue;
      }

      const int32_t rank_cost = rank_active_counts[rank];
      const int32_t token_cost = physical_token_counts[physical_id];
      const int32_t new_rank_cost =
          rank_cost + (physical_active_flags[physical_id] ? 0 : 1);

      const bool is_better =
          (new_rank_cost < best_rank_cost) ||
          (new_rank_cost == best_rank_cost &&
           (token_cost < best_token_cost ||
            (token_cost == best_token_cost &&
             (best_physical < 0 || physical_id < best_physical))));

      if (is_better) {
        best_physical = physical_id;
        best_rank = rank;
        best_rank_cost = new_rank_cost;
        best_token_cost = token_cost;
      }
    }

    if (best_physical < 0) {
      // Fall back to the first available replica to avoid undefined behavior.
      for (int64_t slot = 0; slot < replica_count; ++slot) {
        const int32_t physical_id =
            logical_to_physical[logical_id * slots_per_logical + slot];
        if (physical_id < 0 || physical_id >= physical_capacity) {
          continue;
        }
        best_physical = physical_id;
        if (physical_experts_per_rank > 0) {
          best_rank = physical_id / physical_experts_per_rank;
        } else {
          best_rank = 0;
        }
        if (best_rank >= ep_size) {
          best_rank = ep_size - 1;
        }
        break;
      }
      if (best_physical < 0) {
        out_physical_ids[idx] = -1;
        continue;
      }
      best_rank_cost = rank_active_counts[best_rank] +
                       (physical_active_flags[best_physical] ? 0 : 1);
      best_token_cost = physical_token_counts[best_physical];
    }

    out_physical_ids[idx] = best_physical;
    if (!physical_active_flags[best_physical]) {
      physical_active_flags[best_physical] = 1;
      if (best_rank >= 0 && best_rank < rank_capacity) {
        rank_active_counts[best_rank] += 1;
      }
    }
    physical_token_counts[best_physical] += 1;
  }
}

}  // namespace

void mem_bound_router_greedy(torch::Tensor logical_ids,
                             torch::Tensor logical_to_physical_map,
                             torch::Tensor logical_replica_count,
                             torch::Tensor output,
                             torch::Tensor physical_token_counts,
                             torch::Tensor rank_active_counts,
                             torch::Tensor physical_active,
                             int64_t physical_experts_per_rank,
                             int64_t ep_size) {
  TORCH_CHECK(logical_ids.is_cuda(),
              "logical_ids must reside on CUDA for mem_bound_router_greedy");
  TORCH_CHECK(output.is_cuda(),
              "output tensor must reside on CUDA for mem_bound_router_greedy");
  TORCH_CHECK(logical_to_physical_map.is_cuda(),
              "logical_to_physical_map must reside on CUDA");
  TORCH_CHECK(logical_replica_count.is_cuda(),
              "logical_replica_count must reside on CUDA");
  TORCH_CHECK(physical_token_counts.is_cuda(),
              "physical_token_counts must reside on CUDA");
  TORCH_CHECK(rank_active_counts.is_cuda(),
              "rank_active_counts must reside on CUDA");
  TORCH_CHECK(physical_active.is_cuda(),
              "physical_active must reside on CUDA");

  TORCH_CHECK(
      logical_ids.dim() == 1 && output.dim() == 1,
      "mem_bound_router_greedy expects logical_ids/output to be flattened 1D");
  TORCH_CHECK(logical_ids.numel() == output.numel(),
              "logical_ids and output must have identical numel");
  TORCH_CHECK(logical_ids.scalar_type() == at::kInt,
              "logical_ids must be torch.int32");
  TORCH_CHECK(output.scalar_type() == at::kInt,
              "output must be torch.int32");
  TORCH_CHECK(logical_to_physical_map.scalar_type() == at::kInt,
              "logical_to_physical_map must be torch.int32");
  TORCH_CHECK(logical_replica_count.scalar_type() == at::kLong,
              "logical_replica_count must be torch.int64");
  TORCH_CHECK(physical_token_counts.scalar_type() == at::kInt,
              "physical_token_counts must be torch.int32");
  TORCH_CHECK(rank_active_counts.scalar_type() == at::kInt,
              "rank_active_counts must be torch.int32");
  TORCH_CHECK(physical_active.scalar_type() == at::kByte,
              "physical_active must be torch.uint8");

  if (logical_ids.numel() == 0) {
    return;
  }

  at::cuda::CUDAGuard device_guard(logical_ids.device());
  const auto stream = at::cuda::getCurrentCUDAStream();

  physical_token_counts.zero_();
  rank_active_counts.zero_();
  physical_active.zero_();

  const int64_t num_pairs = logical_ids.numel();
  const int64_t num_logical_experts = logical_to_physical_map.size(0);
  const int64_t slots_per_logical = logical_to_physical_map.size(1);
  const int64_t physical_capacity = physical_token_counts.numel();
  const int64_t rank_capacity = rank_active_counts.numel();

  TORCH_CHECK(physical_capacity > 0,
              "physical_token_counts workspace must have positive size");
  TORCH_CHECK(rank_capacity > 0,
              "rank_active_counts workspace must have positive size");
  TORCH_CHECK(slots_per_logical > 0,
              "logical_to_physical_map must have slot dimension > 0");

  if (physical_experts_per_rank <= 0) {
    physical_experts_per_rank = std::max<int64_t>(1, physical_capacity);
  }
  if (ep_size <= 0) {
    ep_size = 1;
  }

  const dim3 grid(1);
  const dim3 block(1);
  mem_bound_router_greedy_kernel<<<grid, block, 0, stream>>>(
      logical_ids.data_ptr<int32_t>(),
      output.data_ptr<int32_t>(),
      logical_to_physical_map.data_ptr<int32_t>(),
      logical_replica_count.data_ptr<int64_t>(),
      physical_token_counts.data_ptr<int32_t>(),
      rank_active_counts.data_ptr<int32_t>(),
      physical_active.data_ptr<uint8_t>(),
      num_pairs,
      num_logical_experts,
      slots_per_logical,
      static_cast<int32_t>(physical_experts_per_rank),
      static_cast<int32_t>(ep_size),
      physical_capacity,
      rank_capacity);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#else

void mem_bound_router_greedy(torch::Tensor,
                             torch::Tensor,
                             torch::Tensor,
                             torch::Tensor,
                             torch::Tensor,
                             torch::Tensor,
                             torch::Tensor,
                             int64_t,
                             int64_t) {
  TORCH_CHECK(false,
              "mem_bound_router_greedy is not supported on ROCm builds");
}

#endif  // USE_ROCM
