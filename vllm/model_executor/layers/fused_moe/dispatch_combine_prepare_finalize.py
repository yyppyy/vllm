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
import os
from typing import Callable, Optional

import torch

# When set, dispatch_and_route and combine_and_scatter
# are split into per-phase kernels for profiling with
# nsys/ncu. Default: fused (0).
_DC_SPLIT_KERNELS = (
    os.environ.get('VLLM_DC_SPLIT_KERNELS', '0')
    == '1')

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
        use_integrated_routing: bool = False,
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
        self.use_integrated_routing = (
            use_integrated_routing)
        # Set by layer.py when integrated routing is on.
        self.expert_load_view = None

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

        if self.use_integrated_routing:
            return self._prepare_integrated(
                a1, topk_weights, topk_ids,
                num_experts, expert_map,
                quant_config, M, K, topk)

        # Per-sender sections: entries are spread across
        # ws sections in the recv buffer, so mc must cover
        # the full buffer to reach all sections.
        self._mc = self.max_recv
        self._mc_full = self.max_recv

        # Step 1: Launch dispatch P2P kernel.
        # No pre-dispatch barrier needed: dispatch_offset
        # was reset by previous layer's post-combine
        # barrier (RESET_DISPATCH mode), which includes
        # threadfence_system for cross-GPU visibility.
        # First layer uses init barrier + cudaMemset.
        topk_ids_i32 = topk_ids.to(torch.int32)
        topk_weights_f32 = topk_weights.to(torch.float32)

        torch.ops._C_dispatch_combine.dispatch_p2p(
            a1,
            topk_ids_i32,
            topk_weights_f32,
            mgr.config_tensor,
            M, K, topk,
        )

        # Step 2: Fused barrier + stamp/zero + routing.
        # Inline barrier(RESET_COMBINE) syncs dispatch
        # writes and resets combine_offset to 0. Then
        # stamp/zero + routing extraction in one kernel.
        (expert_topk_ids,
         expert_topk_weights,
         expert_num_tokens) = (
            mgr.gpu_prepare_dispatch_recv(
                self._mc, num_experts))

        return lambda: self._receiver(
            a1, K, num_experts, quant_config,
            expert_map, expert_topk_ids,
            expert_topk_weights, expert_num_tokens)

    def _prepare_integrated(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: Optional[torch.Tensor],
        quant_config: FusedMoEQuantConfig,
        M: int,
        K: int,
        topk: int,
    ) -> mk.ReceiverType:
        """Integrated routing path: fused dispatch + route.

        topk_ids contain LOGICAL expert IDs. The fused
        kernel broadcasts tokens to all replica-holding
        ranks, performs push-based all-reduce of per-expert
        counts, runs deterministic routing, and filters
        tokens in a single kernel launch.
        """
        mgr = self.p2p_manager

        # Per-batch tight bound: compaction moves valid
        # entries from scattered per-sender sections into
        # contiguous positions, so mc can be tight.
        # Worst case: all ws source ranks send M*topk
        # entries, with max_rep=2 replicas each.
        self._mc = min(
            M * self.experts_per_token
            * self.world_size_ * 2,
            self.max_recv)
        # mc for combine kernel (iterates over original
        # IPC section layout, unchanged by compaction).
        self._mc_full = self.max_recv

        topk_ids_i32 = topk_ids.to(torch.int32)
        topk_weights_f32 = topk_weights.to(torch.float32)

        if mgr.expert_num_tokens_buf is None:
            mgr.init_prepare_buffers(num_experts)

        if _DC_SPLIT_KERNELS:
            # Split-phase path: 6 separate kernels
            # for per-phase latency profiling.
            mgr.gpu_dar_phase_a(
                a1, topk_ids_i32, topk_weights_f32,
                M, K, topk)
            mgr.gpu_dar_push_and_barrier()
            mgr.gpu_dar_phase_c()
            mgr.gpu_dar_phase_d1(
                self._mc_full, K, num_experts)
            mgr.gpu_dar_phase_d2(
                self._mc_full, num_experts)
            # Compact: gather valid entries from
            # scattered sections into contiguous
            # positions [0, mc_compact).
            mgr.gpu_dar_compact(
                self._mc, num_experts)
            mgr.gpu_dar_phase_e()

            expert_topk_ids = (
                mgr.compact_expert_topk_ids_buf[
                    :self._mc]
                .unsqueeze(1))
            expert_topk_weights = (
                mgr.compact_expert_topk_weights_buf[
                    :self._mc]
                .unsqueeze(1))
            expert_num_tokens = (
                mgr.expert_num_tokens_buf)
            data_remap = None  # Folded into compact
        else:
            # Fused path: single kernel launch.
            (expert_topk_ids,
             expert_topk_weights,
             expert_num_tokens,
             _data_remap) = (
                mgr.gpu_dispatch_and_route(
                    a1, topk_ids_i32,
                    topk_weights_f32,
                    self._mc_full, M, K, topk,
                    num_experts))
            # Compact after fused kernel.
            mgr.gpu_dar_compact(
                self._mc, num_experts)
            expert_topk_ids = (
                mgr.compact_expert_topk_ids_buf[
                    :self._mc]
                .unsqueeze(1))
            expert_topk_weights = (
                mgr.compact_expert_topk_weights_buf[
                    :self._mc]
                .unsqueeze(1))
            data_remap = None  # Folded into compact

        # Record per-physical-expert load for EPLB
        # rebalancing. expert_num_tokens already has
        # physical expert counts from the fused kernel.
        elv = getattr(self, 'expert_load_view', None)
        if elv is not None:
            elv.add_(expert_num_tokens.to(elv.dtype))

        return lambda: self._receiver(
            a1, K, num_experts, quant_config,
            expert_map, expert_topk_ids,
            expert_topk_weights, expert_num_tokens,
            data_remap)

    def _receiver(
        self,
        a1_orig: torch.Tensor,
        K: int,
        num_experts: int,
        quant_config: FusedMoEQuantConfig,
        expert_map: Optional[torch.Tensor],
        expert_topk_ids: torch.Tensor,
        expert_topk_weights: torch.Tensor,
        expert_num_tokens: torch.Tensor,
        data_remap: Optional[torch.Tensor] = None,
    ) -> mk.PrepareResultType:
        mgr = self.p2p_manager
        mc = self._mc

        if self.use_integrated_routing:
            # Compacted path: gather mc_compact entries
            # using compact_data_remap which combines
            # section compaction + co-located dedup.
            compact_remap = (
                mgr.compact_data_remap_buf[:mc])
            expert_x = (
                mgr.dispatch_recv_tensor[compact_remap])
        else:
            # Non-integrated path: use original recv.
            expert_x = mgr.dispatch_recv_tensor[:mc]
            # Expand shared data for co-located dedup.
            if data_remap is not None:
                expert_x = expert_x[data_remap]

        # Post-dispatch quantization.
        expert_x_scale = None
        if not quant_config.is_block_quantized:
            if expert_x.numel() != 0:
                expert_x, expert_x_scale = (
                    moe_kernel_quantize_input(
                        expert_x,
                        quant_config.a1_scale,
                        quant_dtype=(
                            quant_config.quant_dtype),
                        per_act_token_quant=False,
                        block_shape=(
                            quant_config.block_shape)))
        else:
            expert_x, expert_x_scale = (
                moe_kernel_quantize_input(
                    expert_x,
                    quant_config.a1_scale,
                    quant_dtype=(
                        quant_config.quant_dtype),
                    per_act_token_quant=(
                        quant_config.per_act_token_quant),
                    block_shape=(
                        quant_config.block_shape)))

        # Slice to local experts only.
        local_expert_num_tokens = expert_num_tokens[
            self.rank_expert_offset:
            self.rank_expert_offset
            + self.num_local_experts]

        # expert_num_tokens_cpu=None for CUDA graph
        # compat (no D2H during graph capture).
        expert_tokens_meta = mk.ExpertTokensMetadata(
            expert_num_tokens=local_expert_num_tokens,
            expert_num_tokens_cpu=None,
            num_tokens_for_config=a1_orig.shape[0],
            topk_ids_for_masking=(
                expert_topk_ids.view(-1)))

        return (expert_x, expert_x_scale,
                expert_tokens_meta,
                expert_topk_ids,
                expert_topk_weights)

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
        # mc_full: combine iterates over original IPC
        # section layout (unchanged by compaction).
        mc_full = self._mc_full
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

        # Step 2: Combine + barrier + scatter-add.
        # Use mc_full (not mc_compact) because combine
        # reads dispatch_meta at original IPC positions
        # and uses compact_reverse to index expert_output.
        meta_bytes = (
            mgr.dispatch_meta_tensor[:mc_full]
            .contiguous().view(torch.uint8))
        if _DC_SPLIT_KERNELS:
            # Split path: 3 separate kernels.
            mgr.gpu_combine_p2p(
                fused_expert_output,
                meta_bytes, mc_full)
            mgr.gpu_p2p_barrier_reset_dispatch()
            mgr.gpu_scatter_add_direct(
                output, mc_full)
        else:
            # Fused path: single kernel launch.
            mgr.gpu_combine_and_scatter(
                fused_expert_output,
                meta_bytes,
                output,
                mc_full)

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
