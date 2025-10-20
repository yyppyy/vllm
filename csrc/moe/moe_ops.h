#pragma once

#include <torch/all.h>

void topk_softmax(torch::Tensor& topk_weights, torch::Tensor& topk_indices,
                  torch::Tensor& token_expert_indices,
                  torch::Tensor& gating_output);

void moe_sum(torch::Tensor& input, torch::Tensor& output);

void moe_align_block_size(torch::Tensor topk_ids, int64_t num_experts,
                          int64_t block_size, torch::Tensor sorted_token_ids,
                          torch::Tensor experts_ids,
                          torch::Tensor num_tokens_post_pad);
#ifndef USE_ROCM
torch::Tensor moe_wna16_gemm(torch::Tensor input, torch::Tensor output,
                             torch::Tensor b_qweight, torch::Tensor b_scales,
                             std::optional<torch::Tensor> b_qzeros,
                             std::optional<torch::Tensor> topk_weights,
                             torch::Tensor sorted_token_ids,
                             torch::Tensor expert_ids,
                             torch::Tensor num_tokens_post_pad, int64_t top_k,
                             int64_t BLOCK_SIZE_M, int64_t BLOCK_SIZE_N,
                             int64_t BLOCK_SIZE_K, int64_t bit);

std::tuple<torch::Tensor, torch::Tensor> grouped_topk(
    torch::Tensor const& scores, torch::Tensor const& scores_with_bias,
    int64_t n_group, int64_t topk_group, int64_t topk, bool renormalize,
    double routed_scaling_factor);
#endif

bool moe_permute_unpermute_supported();

void shuffle_rows(const torch::Tensor& input_tensor,
                  const torch::Tensor& dst2src_map,
                  torch::Tensor& output_tensor);


// Greedy: smallest-choice-first (device-only)
void greedy_smallest_choice_first_cuda(
    const at::Tensor& rank_offsets,   // int32 [n+1]
    const at::Tensor& rank_indices,   // int32 [nnz]
    at::Tensor& chosen_rank,          // int32 [n]
    int32_t P);

// Exact: binary search on L + capacity-bounded matching (device-only CSR)
// Returns optimal L as a Tensor scalar (int32) for graph-friendliness.
at::Tensor exact_min_max_activations_cuda(
    const at::Tensor& rank_offsets,
    const at::Tensor& rank_indices,
    at::Tensor& chosen_rank,
    int32_t P);

// Pick a physical replica that resides on chosen rank for each active expert
void select_replica_on_rank_cuda(
    const at::Tensor& logical_to_physical_map,  // int64 [E, Rmax], -1 padded
    const at::Tensor& logical_replica_count,    // int32 [E]
    const at::Tensor& active_experts,           // int64 [n]
    const at::Tensor& chosen_rank,              // int32 [n]
    at::Tensor& chosen_replica,                 // int64 [n]
    int32_t P);

// Map logical expert ids -> chosen physical ids, device-only (dense LUT)
void map_tokens_to_chosen_replica_cuda(
    const at::Tensor& topk_ids_logical,         // int64 [T,K]
    const at::Tensor& active_experts,           // int64 [n]
    const at::Tensor& chosen_replica,           // int64 [n]
    at::Tensor& out_physical_ids);              // int64 [T,K]