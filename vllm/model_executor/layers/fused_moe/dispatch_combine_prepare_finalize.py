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

# Read env var directly to avoid circular import
# with layer.py (which imports this module).
import os
_MOE_LOAD_PROFILE_INTERVAL = int(
    os.environ.get('VLLM_MOE_LOAD_PROFILE_INTERVAL',
                    '0'))

# Routing mode threshold: M <= this uses routing_mode=0
# (minimize activated experts), M > this uses
# routing_mode=1 (balance tokens via section-level
# splitting across replicas).
ROUTING_MODE_THRESHOLD = 256


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

        # Ensure pre-allocated buffers exist (lazy init).
        if getattr(mgr, 'topk_ids_i32_buf', None) is None:
            mgr.init_prepare_buffers(num_experts)

        # Use integrated routing for all batch sizes when
        # enabled. routing_mode selects the algorithm:
        # 0 = minimize experts (decode), 1 = balance
        # tokens via section-level splitting (prefill).
        use_integrated = self.use_integrated_routing
        self._used_integrated = use_integrated

        if use_integrated:
            routing_mode = (
                0 if M <= ROUTING_MODE_THRESHOLD else 1)
            return self._prepare_integrated(
                a1, topk_weights, topk_ids,
                num_experts, expert_map,
                quant_config, M, K, topk,
                routing_mode)

        # Per-sender sections: entries are spread across
        # ws sections in the recv buffer, so mc must cover
        # the full buffer to reach all sections.
        self._mc = self.max_recv
        self._mc_full = self.max_recv

        # Reset compact_reverse to identity if integrated
        # routing is configured (previous call may have
        # modified it via dar_compact).
        # Guard: compact_reverse_buf is lazily allocated
        # in init_prepare_buffers(), which may not have
        # been called yet (e.g. during profile_run).
        if (self.use_integrated_routing
                and mgr.expert_num_tokens_buf is not None):
            torch.arange(
                self.max_recv,
                out=mgr.compact_reverse_buf)

        # Step 1: Launch dispatch P2P kernel.
        # No pre-dispatch barrier needed: dispatch_offset
        # was reset by previous layer's post-combine
        # barrier (RESET_DISPATCH mode), which includes
        # threadfence_system for cross-GPU visibility.
        # First layer uses init barrier + cudaMemset.
        # Use pre-allocated buffers for dtype conversion
        # (.to() allocates; .copy_() is graph-safe).
        topk_ids_i32 = mgr.topk_ids_i32_buf[:M]
        topk_ids_i32.copy_(topk_ids)
        topk_weights_f32 = mgr.topk_weights_f32_buf[:M]
        topk_weights_f32.copy_(topk_weights)

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

        # Track load for EPLB even in fallback path.
        if self.use_integrated_routing:
            elv = getattr(
                self, 'expert_load_view', None)
            if elv is not None:
                elv.add_(
                    expert_num_tokens)

        # data_remap computed by prepare_dispatch_recv:
        # maps each metadata entry to its compact data
        # position (sender_rank * M + token_idx).
        data_remap = mgr.data_remap_buf[:self._mc]
        return lambda: self._receiver(
            a1, K, num_experts, quant_config,
            expert_map, expert_topk_ids,
            expert_topk_weights, expert_num_tokens,
            data_remap)

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
        routing_mode: int = 0,
    ) -> mk.ReceiverType:
        """Integrated routing path: fused dispatch + route.

        topk_ids contain LOGICAL expert IDs. The fused
        kernel broadcasts tokens to all replica-holding
        ranks, performs push-based all-reduce of per-expert
        counts, runs deterministic routing, and filters
        tokens in a single kernel launch.

        routing_mode=0: minimize activated experts
          (increment rank_active by 1 per expert).
        routing_mode=1: balance tokens via section-level
          splitting (each section assigned independently
          to the least-loaded replica).
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

        # Pre-allocated dtype conversion buffers
        # (.to() allocates; .copy_() is graph-safe).
        topk_ids_i32 = mgr.topk_ids_i32_buf[:M]
        topk_ids_i32.copy_(topk_ids)
        topk_weights_f32 = mgr.topk_weights_f32_buf[:M]
        topk_weights_f32.copy_(topk_weights)

        if mgr.expert_num_tokens_buf is None:
            mgr.init_prepare_buffers(num_experts)

        # Fused path: single kernel launch.
        (expert_topk_ids,
         expert_topk_weights,
         expert_num_tokens,
         _data_remap) = (
            mgr.gpu_dispatch_and_route(
                a1, topk_ids_i32,
                topk_weights_f32,
                self._mc_full, M, K, topk,
                num_experts,
                routing_mode=routing_mode))
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
            elv.add_(expert_num_tokens)

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

        # Copy int32 remap indices into pre-allocated
        # int64 buffer (index_select requires int64).
        # .copy_() is CUDA graph safe; .long() is not
        # (it allocates a new tensor).
        remap_i64 = mgr.remap_i64_buf[:mc]
        if self._used_integrated:
            remap_i64.copy_(
                mgr.compact_data_remap_buf[:mc])
        else:
            remap_i64.copy_(data_remap)
        torch.index_select(
            mgr.dispatch_recv_tensor, 0,
            remap_i64,
            out=mgr.expert_x_buf[:mc])
        expert_x = mgr.expert_x_buf[:mc]

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

        # MoE load profiling (dispatch_combine path).
        if _MOE_LOAD_PROFILE_INTERVAL > 0:
            from vllm.model_executor.layers.fused_moe.layer \
                import _moe_load_profiler
            _moe_load_profiler.record(
                M=a1_orig.shape[0],
                expert_num_tokens=expert_num_tokens,
                ep_rank=self.rank_,
                ep_size=self.world_size_,
                experts_per_rank=self.experts_per_rank,
                num_local_experts=self.num_local_experts,
            )

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
