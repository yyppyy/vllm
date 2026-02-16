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
from vllm.distributed import get_ep_group
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

        # Update config tensor with experts_per_rank.
        self.p2p_manager.update_experts_per_rank(
            self.experts_per_rank)

        # Cache the EP group coordinator for NCCL barriers.
        self._ep_group = get_ep_group()

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

        # Step 1: Reset offset counters (GPU kernel).
        mgr.gpu_reset_offsets()

        # Step 2: NCCL barrier - all ranks must finish
        # resetting before any rank launches dispatch.
        mgr.nccl_barrier(self._ep_group)

        # Step 3: Launch dispatch P2P kernel.
        topk_ids_i32 = topk_ids.to(torch.int32)
        topk_weights_f32 = topk_weights.to(torch.float32)

        torch.ops._C_dispatch_combine.dispatch_p2p(
            a1,
            topk_ids_i32,
            topk_weights_f32,
            mgr.config_tensor,
            M, K, topk,
        )

        # Step 4: NCCL barrier - all P2P writes complete.
        mgr.nccl_barrier(self._ep_group)

        # Step 5: Copy from IPC buffers (GPU kernels).
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
        max_recv = self.max_recv

        # Use pre-allocated tensors (fixed size = max_recv).
        expert_x = mgr.dispatch_recv_tensor
        dispatch_meta = mgr.dispatch_meta_tensor

        # Extract expert IDs from metadata column 2.
        expert_topk_ids = dispatch_meta[:, 2].clone()

        # Remap -1 entries to a safe expert.
        expert_topk_ids = torch.where(
            expert_topk_ids == -1,
            (num_experts - 1
             if self.rank_expert_offset == 0 else 0),
            expert_topk_ids)

        # Shape as (max_recv, 1) for topk=1.
        expert_topk_ids = expert_topk_ids.unsqueeze(1).to(
            torch.int64)

        # Weights are all 1.0 for expert computation;
        # actual weights are applied in combine phase.
        expert_topk_weights = torch.ones(
            (max_recv, 1), dtype=torch.float32,
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
        expert_num_tokens = torch.zeros(
            num_experts, dtype=torch.int32,
            device=expert_x.device)
        flat_ids = expert_topk_ids.view(-1)
        ones = torch.ones_like(
            flat_ids, dtype=torch.int32)
        expert_num_tokens.scatter_add_(
            0, flat_ids.to(torch.int64), ones)

        # Slice to local experts only.
        local_expert_num_tokens = expert_num_tokens[
            self.rank_expert_offset:
            self.rank_expert_offset
            + self.num_local_experts]

        # expert_num_tokens_cpu=None for CUDA graph compat
        # (no device-to-host transfer during graph capture).
        expert_tokens_meta = mk.ExpertTokensMetadata(
            expert_num_tokens=local_expert_num_tokens,
            expert_num_tokens_cpu=None)

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
        max_recv = self.max_recv
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

        # Step 2: Reset combine offset (GPU kernel).
        mgr.gpu_reset_combine_offset()

        # Step 3: NCCL barrier before combine.
        mgr.nccl_barrier(self._ep_group)

        # Step 4: Launch combine P2P kernel.
        # The kernel reads actual dispatch_recv count from
        # config and skips entries beyond it.
        # dispatch_meta is the pre-allocated tensor already
        # populated by gpu_copy_dispatch_meta in prepare.
        meta_bytes = (
            mgr.dispatch_meta_tensor.contiguous().view(
                torch.uint8))
        torch.ops._C_dispatch_combine.combine_p2p(
            fused_expert_output,
            meta_bytes,
            mgr.config_tensor,
            max_recv, K,
        )

        # Step 5: NCCL barrier for combine completion.
        mgr.nccl_barrier(self._ep_group)

        # Step 6: Copy combine results (GPU kernels).
        mgr.gpu_copy_combine_recv()
        mgr.gpu_copy_combine_meta()

        # Step 7: Scatter-add weighted results to output.
        # Use pre-allocated float32 accumulator.
        accum = mgr.accum_tensor[:output.shape[0]]
        accum.zero_()

        combine_meta_bytes = (
            mgr.combine_meta_tensor.contiguous().view(
                torch.uint8))
        torch.ops._C_dispatch_combine.scatter_add_weighted(
            accum,
            mgr.combine_recv_tensor,
            combine_meta_bytes,
            max_recv, K,
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
