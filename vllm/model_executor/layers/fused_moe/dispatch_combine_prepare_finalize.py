# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PrepareAndFinalize for dispatch_combine all2all backend.

Routing-aware dispatch/combine with Standard activation format output,
using custom CUDA P2P kernels for low-latency GPU-to-GPU communication.

All runtime operations are GPU-side stream operations for CUDA graph
compatibility. Synchronization uses NCCL all-reduce barriers on the
EP device group instead of host-side torch.cuda.synchronize() or
dist.barrier(cpu_group).
"""
from typing import Callable, Optional

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceContiguous,
    TopKWeightAndReduceDelegate)
from vllm.model_executor.layers.fused_moe.utils import (
    moe_kernel_quantize_input)

logger = init_logger(__name__)


class DispatchCombinePrepareAndFinalize(
        mk.FusedMoEPrepareAndFinalize):
    """Routing-aware dispatch/combine with Standard format output.

    Uses custom CUDA P2P kernels for direct GPU-to-GPU token
    transfer. All operations are GPU-side for CUDA graph compat.

    Data flow:
    1. Dispatch (prepare): route tokens via P2P -> quantize
    2. Expert execution: TritonExperts on received tokens
    3. Combine (finalize): weight+reduce -> P2P -> scatter-add
    """

    def __init__(
        self,
        p2p_manager,
        max_num_tokens: int,
        num_experts: int,
        num_local_experts: int,
        experts_per_token: int,
        rank: int,
        world_size: int,
        rank_expert_offset: int,
    ):
        super().__init__()
        self.p2p_manager = p2p_manager
        self.max_num_tokens = max_num_tokens
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.experts_per_token = experts_per_token
        self.rank_ = rank
        self.world_size_ = world_size
        self.rank_expert_offset = rank_expert_offset
        self.experts_per_rank = num_experts // world_size
        self.max_recv = p2p_manager.max_recv
        # Tight bound on tokens needing computation.
        # Total expert assignments = max_num_tokens * topk;
        # copy kernels pack entries contiguously so
        # slicing to [:max_compute] is safe.
        self.max_compute = (
            max_num_tokens * experts_per_token)

        # Update config tensor with experts_per_rank.
        self.p2p_manager.update_experts_per_rank(
            self.experts_per_rank)

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> Optional[int]:
        return None

    def topk_indices_dtype(self) -> Optional[torch.dtype]:
        return torch.int64

    def num_dispatchers(self) -> int:
        return self.world_size_

    def supports_async(self) -> bool:
        return True

    def prepare_async(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
    ) -> mk.ReceiverType:
        M, K = a1.shape
        topk = topk_ids.size(1)

        if apply_router_weight_on_input:
            assert topk == 1, (
                "apply_router_weight_on_input only "
                "implemented for topk=1")
            a1 = a1 * topk_weights.to(a1.dtype)

        mgr = self.p2p_manager

        # Step 1: Reset offsets + P2P barrier (merged).
        # Ensures all ranks finish resetting before any
        # rank launches dispatch.
        mgr.gpu_p2p_barrier_reset_offsets()

        # Step 2: Launch dispatch P2P kernel.
        topk_ids_i32 = topk_ids.to(torch.int32)
        topk_weights_f32 = topk_weights.to(torch.float32)

        torch.ops._C_dispatch_combine.dispatch_p2p(
            a1,
            topk_ids_i32,
            topk_weights_f32,
            mgr.config_tensor,
            M, K, topk,
        )

        # Step 3: P2P barrier - all P2P writes complete.
        mgr.gpu_p2p_barrier()

        # Step 4: Copy from IPC buffers (GPU kernels).
        # These kernels read actual count from offset
        # counters and zero entries beyond actual count.
        mgr.gpu_copy_dispatch_recv()
        mgr.gpu_copy_dispatch_meta()

        return lambda: self._receiver(
            a1, K, num_experts, quant_config, expert_map)

    def _receiver(
        self,
        a1_orig: torch.Tensor,
        K: int,
        num_experts: int,
        quant_config: FusedMoEQuantConfig,
        expert_map: Optional[torch.Tensor],
    ) -> mk.PrepareResultType:
        mgr = self.p2p_manager
        mc = self.max_compute

        # Slice to max_compute (= max_num_tokens * topk).
        # Copy kernels pack real entries at 0..actual_count-1
        # and actual_count <= max_compute always holds.
        # This reduces num_tokens for fused_moe_kernel grid,
        # act_and_mul grid, and intermediate buffer sizes.
        expert_x = mgr.dispatch_recv_tensor[:mc]
        dispatch_meta = mgr.dispatch_meta_tensor[:mc]

        # Extract expert IDs from metadata column 2.
        # Padding entries have expert_id = num_experts
        # (set by copy_dispatch_meta_kernel), which
        # moe_align_block_size skips automatically.
        expert_topk_ids = dispatch_meta[:, 2].clone()

        # Shape as (max_compute, 1) for topk=1.
        expert_topk_ids = expert_topk_ids.unsqueeze(1).to(
            torch.int64)

        # Weights are all 1.0 for expert computation;
        # actual weights are applied in combine phase.
        expert_topk_weights = torch.ones(
            (mc, 1), dtype=torch.float32,
            device=expert_x.device)

        # Post-dispatch quantization.
        expert_x_scale = None
        if not quant_config.is_block_quantized:
            if expert_x.numel() != 0:
                expert_x, expert_x_scale = (
                    moe_kernel_quantize_input(
                        expert_x,
                        quant_config.a1_scale,
                        quant_dtype=quant_config.quant_dtype,
                        per_act_token_quant=False,
                        block_shape=quant_config.block_shape))
        else:
            expert_x, expert_x_scale = (
                moe_kernel_quantize_input(
                    expert_x,
                    quant_config.a1_scale,
                    quant_dtype=quant_config.quant_dtype,
                    per_act_token_quant=(
                        quant_config.per_act_token_quant),
                    block_shape=quant_config.block_shape))

        # Compute expert token counts.
        # Padding entries have expert_id = num_experts;
        # use masked scatter to exclude them (avoids OOB).
        expert_num_tokens = torch.zeros(
            num_experts, dtype=torch.int32,
            device=expert_x.device)
        flat_ids = expert_topk_ids.view(-1)
        valid_mask = (flat_ids < num_experts).to(
            torch.int32)
        safe_ids = flat_ids.clamp(0, num_experts - 1)
        expert_num_tokens.scatter_add_(
            0, safe_ids.to(torch.int64), valid_mask)

        # Slice to local experts only.
        local_expert_num_tokens = expert_num_tokens[
            self.rank_expert_offset:
            self.rank_expert_offset
            + self.num_local_experts]

        # expert_num_tokens_cpu=None for CUDA graph compat
        # (no device-to-host transfer during graph capture).
        # num_tokens_for_config = max_num_tokens so the
        # Triton autotuner picks decode-friendly tile sizes
        # instead of using max_recv (which is much larger).
        expert_tokens_meta = mk.ExpertTokensMetadata(
            expert_num_tokens=local_expert_num_tokens,
            expert_num_tokens_cpu=None,
            num_tokens_for_config=self.max_num_tokens)

        return (expert_x, expert_x_scale,
                expert_tokens_meta,
                expert_topk_ids, expert_topk_weights)

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
    ) -> mk.PrepareResultType:
        receiver = self.prepare_async(
            a1, topk_weights, topk_ids, num_experts,
            expert_map, apply_router_weight_on_input,
            quant_config)
        return receiver()

    def _finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
        do_async: bool,
    ) -> Optional[Callable]:
        K = output.shape[-1]
        mc = self.max_compute
        mgr = self.p2p_manager

        # Step 1: Apply weights + reduce on dispatched tokens.
        if fused_expert_output.numel() != 0:
            if isinstance(weight_and_reduce_impl,
                          TopKWeightAndReduceDelegate):
                weight_and_reduce_impl = (
                    TopKWeightAndReduceContiguous())
            fused_expert_output = (
                weight_and_reduce_impl.apply(
                    output=None,
                    fused_expert_output=fused_expert_output,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    apply_router_weight_on_input=(
                        apply_router_weight_on_input),
                ))

        # Step 2: Reset combine offset + P2P barrier.
        mgr.gpu_p2p_barrier_reset_combine_offset()

        # Step 3: Launch combine P2P kernel.
        # The kernel reads actual dispatch_recv count from
        # config and skips entries beyond it.
        # Grid = max_compute (not max_recv) since
        # actual_count <= max_compute always holds.
        meta_bytes = (
            mgr.dispatch_meta_tensor[:mc]
            .contiguous().view(torch.uint8))
        torch.ops._C_dispatch_combine.combine_p2p(
            fused_expert_output,
            meta_bytes,
            mgr.config_tensor,
            mc, K,
        )

        # Step 4: P2P barrier for combine completion.
        mgr.gpu_p2p_barrier()

        # Step 5: Copy combine results (GPU kernels).
        mgr.gpu_copy_combine_recv()
        mgr.gpu_copy_combine_meta()

        # Step 6: Scatter-add weighted results to output.
        # Float32 accumulator for precise atomic scatter-add.
        accum = torch.zeros(
            output.shape, dtype=torch.float32,
            device=output.device)

        combine_meta_bytes = (
            mgr.combine_meta_tensor[:mc]
            .contiguous().view(torch.uint8))
        torch.ops._C_dispatch_combine.scatter_add_weighted(
            accum,
            mgr.combine_recv_tensor[:mc],
            combine_meta_bytes,
            mc, K,
        )
        output.copy_(accum.to(output.dtype))

        if do_async:
            return lambda: None
        return None

    def finalize_async(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> Callable:
        receiver = self._finalize(
            output, fused_expert_output,
            topk_weights, topk_ids,
            apply_router_weight_on_input,
            weight_and_reduce_impl,
            do_async=True)
        assert receiver is not None
        return receiver

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        self._finalize(
            output, fused_expert_output,
            topk_weights, topk_ids,
            apply_router_weight_on_input,
            weight_and_reduce_impl,
            do_async=False)
