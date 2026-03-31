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

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig)
from vllm.model_executor.layers.fused_moe.utils import (
    moe_kernel_quantize_input)

logger = init_logger(__name__)

_MOE_LOAD_PROFILE_INTERVAL = int(
    os.environ.get('VLLM_MOE_LOAD_PROFILE_INTERVAL',
                    '0'))

# Minimum M to print per-layer expert compute breakdown.
# Only prefill batches (large M) are printed.
_EXPERT_PROFILE_M_THRESHOLD = int(
    os.environ.get('VLLM_DC_EXPERT_PROFILE_M', '256'))

# Buffer managers registered for profiling.
_registered_mgrs: list = []


def enable_dc_profiling():
    """Enable DC profiling on all registered buffer
    managers. Called by eplb_state.py after the first
    real (non-profile) EPLB rebalance."""
    for mgr in _registered_mgrs:
        mgr._profiling_after_rebalance = True

# Routing mode threshold: M <= this uses routing_mode=0
# (minimize activated experts), M > this uses
# routing_mode=1 (balance tokens via section-level
# splitting across replicas).
ROUTING_MODE_THRESHOLD = int(
    os.environ.get("VLLM_ROUTING_MODE_THRESHOLD", "256"))

# Which routing mode to use for large M (> threshold):
# 1 = greedy LPT (balance tokens across replicas),
# 2 = even round-robin (split sections across replicas).
PREFILL_ROUTING_MODE = int(
    os.environ.get("VLLM_PREFILL_ROUTING_MODE", "1"))

# Debug: dump routing decisions once per routing_mode,
# and again after every EPLB rebalance.
# Set VLLM_ROUTING_DEBUG=1 to enable.
_ROUTING_DEBUG = os.environ.get("VLLM_ROUTING_DEBUG", "0") == "1"
_routing_debug_done: set = set()  # track which modes we've dumped
_routing_debug_skip = 5  # skip first N calls (warmup/capture)


def routing_debug_reset():
    """Called after EPLB rebalance to re-dump on next call."""
    global _routing_debug_done, _routing_debug_skip
    _routing_debug_done.clear()
    _routing_debug_skip = 0  # don't skip after rebalance


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
        # Per-layer routing tables (set by layer.py on
        # EPLB rebalance, copied to shared buffer manager
        # at the start of each layer's prepare call).
        self._layer_routing_map = None
        self._layer_routing_count = None
        # Set by layer.py when integrated routing is on.
        self.expert_load_view = None

        # Expert compute profiling (per-layer state).
        self._expert_step_names = [
            'recv_start', 'compact_done',
            'gather_done', 'align_done',
            'expert_done', 'combine_start',
        ]
        self._expert_events: dict[
            str, torch.cuda.Event] = {}
        self._expert_M = 0
        self._expert_local_tokens = 0
        self._expert_num_tokens: (
            torch.Tensor | None) = None
        self._moe_layer_idx = -1
        self._print_counts: dict[int, int] = {}
        self._comm_print_counts: dict[int, int] = {}

        # Update config tensor with experts_per_rank.
        self.p2p_manager.update_experts_per_rank(
            self.experts_per_rank)
        # Register mgr for profiling enable.
        if (self.p2p_manager not in _registered_mgrs
                and self.p2p_manager._profiling_enabled):
            _registered_mgrs.append(self.p2p_manager)

    def record_expert_event(self, name: str):
        """Record a CUDA event for expert compute
        profiling. Only active after EPLB rebalance."""
        mgr = self.p2p_manager
        if not mgr._profiling_enabled:
            return
        if not mgr._profiling_after_rebalance:
            return
        if torch.cuda.is_current_stream_capturing():
            return
        if name not in self._expert_events:
            self._expert_events[name] = (
                torch.cuda.Event(enable_timing=True))
        self._expert_events[name].record()

    def accumulate_expert_times(self):
        """Print per-layer expert compute breakdown.
        Only prints after EPLB rebalance, throttled."""
        mgr = self.p2p_manager
        if not mgr._profiling_enabled:
            return
        if not mgr._profiling_after_rebalance:
            return
        if torch.cuda.is_current_stream_capturing():
            return
        M = self._expert_M
        local_tokens = self._expert_local_tokens
        # Only print for large M (prefill batches).
        if M < _EXPERT_PROFILE_M_THRESHOLD:
            return
        # Per-(layer, M) throttle: max 10 prints.
        counts = self._print_counts
        counts[M] = counts.get(M, 0) + 1
        if counts[M] > 10:
            return
        names = self._expert_step_names
        # Need all events recorded.
        for name in names:
            if name not in self._expert_events:
                logger.debug(
                    "expert_compute: missing event "
                    "'%s' (layer %d, M=%d), skipping",
                    name, self._moe_layer_idx, M)
                return
        # Synchronize to ensure events completed.
        torch.cuda.current_stream().synchronize()
        parts = []
        total_us = 0.0
        for i in range(len(names) - 1):
            e_start = self._expert_events[names[i]]
            e_end = self._expert_events[names[i + 1]]
            try:
                elapsed_us = (
                    e_start.elapsed_time(e_end)
                    * 1000.0)
                total_us += elapsed_us
                parts.append(
                    f"  {names[i]}: {elapsed_us:.1f}"
                    " us")
            except RuntimeError:
                parts.append(
                    f"  {names[i]}: N/A")
        # Per-expert token distribution.
        if (self._expert_num_tokens is not None
                and self._expert_num_tokens.numel() > 0):
            et = self._expert_num_tokens.cpu().tolist()
            mx = max(et)
            mn = min(et) if min(et) > 0 else 0
            mean_et = sum(et) / len(et)
            mx_i = et.index(mx)
            mn_i = et.index(mn)
            ratio = (mx / mn) if mn > 0 else float('inf')
            n_active = sum(1 for x in et if x > 0)
            total_tokens = sum(et)
            ru = getattr(self, '_router_unique', 0)
            rt = getattr(self, '_router_total', 0)
            parts.append(
                f"  expert_tokens: max={mx}(e{mx_i})"
                f" min={mn}(e{mn_i})"
                f" mean={mean_et:.0f}"
                f" ratio={ratio:.1f}x"
                f" activated={n_active}/{len(et)}"
                f" total={total_tokens}"
                f" router={ru}/{rt}")
            # rc=1 vs rc>1 token split for imbalance
            # decomposition.
            if (mgr._integrated_routing_enabled
                    and mgr._routing_count_tensor
                    is not None
                    and mgr._routing_map_tensor
                    is not None):
                rc = mgr._routing_count_tensor \
                    .cpu().tolist()
                l2p = mgr._routing_map_tensor \
                    .cpu().tolist()
                epr = mgr._physical_experts_per_rank
                max_rep = (len(l2p) // len(rc)
                           if len(rc) > 0 else 1)
                NL = len(rc)
                # Build phys->logical for local slots.
                p2l = {}
                for e in range(NL):
                    for rep in range(max_rep):
                        p = l2p[e * max_rep + rep]
                        if p >= 0:
                            p2l[p] = e
                rc1_sum = 0
                rc2_sum = 0
                base = mgr.rank * epr
                for j in range(len(et)):
                    phys = base + j
                    log_e = p2l.get(phys, -1)
                    if log_e >= 0 and log_e < NL:
                        if rc[log_e] <= 1:
                            rc1_sum += et[j]
                        else:
                            rc2_sum += et[j]
                total_tok = rc1_sum + rc2_sum
                frac = (rc1_sum / total_tok * 100
                        if total_tok > 0 else 0)
                from collections import Counter
                rc_dist = Counter(
                    int(rc[log_e])
                    for log_e in set(p2l.values())
                    if 0 <= log_e < NL)
                n_mapped = sum(
                    1 for j in range(len(et))
                    if p2l.get(base + j, -1) >= 0)
                parts.append(
                    f"  rc_split: rc1_tokens={rc1_sum}"
                    f" rc2_tokens={rc2_sum}"
                    f" rc1_frac={frac:.1f}%"
                    f" rc_dist="
                    f"{dict(sorted(rc_dist.items()))}"
                    f" mapped={n_mapped}/{len(et)}")
        logger.info(
            "DC profile [rank %d] layer %d "
            "expert_compute "
            "(total %.1f us, M=%d, "
            "local_tokens=%d):\n%s",
            mgr.rank, self._moe_layer_idx,
            total_us, M,
            local_tokens, "\n".join(parts))

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> Optional[int]:
        return None

    @property
    def skip_expert_chunking(self) -> bool:
        # DC flattens (token, expert) pairs to (mc, 1).
        # mc includes padding but fused_moe skips padding
        # via sorted_token_ids sentinel. Single pass avoids
        # extra kernel launches from chunking.
        return True

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

        # routing_mode selects the algorithm:
        # 0 = minimize experts (decode),
        # 1 = greedy LPT (balance tokens, prefill),
        # 2 = even round-robin (prefill).
        routing_mode = (
            0 if M <= ROUTING_MODE_THRESHOLD
            else PREFILL_ROUTING_MODE)
        return self._prepare_integrated(
            a1, topk_weights, topk_ids,
            num_experts, expert_map,
            quant_config, M, K, topk,
            routing_mode)

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

        # Restore this layer's routing tables into the
        # shared buffer manager (all layers share one mgr
        # but each has different EPLB placement).
        # Per-layer maps are pre-padded to mgr._max_replicas
        # at store time (in layer.py _maybe_init_integrated_routing).
        if (self._layer_routing_map is not None
                and mgr._integrated_routing_enabled):
            mgr._routing_map_tensor.copy_(
                self._layer_routing_map)
            mgr._routing_count_tensor.copy_(
                self._layer_routing_count)

        # Per-batch tight bound: compaction moves valid
        # entries from scattered per-sender sections into
        # contiguous positions, so mc can be tight.
        # Worst case: all ws source ranks send M*topk
        # entries, with max_rep=2 replicas each.
        mc = min(
            M * self.experts_per_token
            * self.world_size_ * 2,
            self.max_recv)

        # Pre-allocated dtype conversion buffers
        # (.to() allocates; .copy_() is graph-safe).
        topk_ids_i32 = mgr.topk_ids_i32_buf[:M]
        topk_ids_i32.copy_(topk_ids)
        topk_weights_f32 = mgr.topk_weights_f32_buf[:M]
        topk_weights_f32.copy_(topk_weights)

        if mgr.expert_num_tokens_buf is None:
            mgr.init_prepare_buffers(num_experts)

        # mc for combine kernel (iterates over original
        # IPC section layout, unchanged by compaction).
        mc_full = self.max_recv

        # Fused path: single kernel launch.
        (expert_topk_ids,
         expert_topk_weights,
         expert_num_tokens,
         _data_remap) = (
            mgr.gpu_dispatch_and_route(
                a1, topk_ids_i32,
                topk_weights_f32,
                mc_full, M, K, topk,
                num_experts,
                routing_mode=routing_mode))
        # Debug: dump l2p map and routing decisions once.
        global _routing_debug_skip
        if _ROUTING_DEBUG and _routing_debug_skip > 0:
            _routing_debug_skip -= 1
        if (_ROUTING_DEBUG
                and _routing_debug_skip == 0
                and routing_mode not in _routing_debug_done):
            _routing_debug_done.add(routing_mode)
            torch.cuda.synchronize()
            rank = mgr.rank
            ws = mgr.world_size
            NL = mgr._num_logical_experts
            mr = mgr._max_replicas
            l2p = mgr._routing_map_tensor.cpu().tolist()
            rc = mgr._routing_count_tensor.cpu().tolist()
            # Read routing_selection from GPU.
            import ctypes
            rs_size = ws * NL
            rs_host = torch.zeros(
                rs_size, dtype=torch.int32)
            rs_host_ptr = ctypes.c_void_p(
                rs_host.data_ptr())
            mgr._cuda_rt.cudaMemcpy(
                rs_host_ptr,
                mgr._raw_routing_selection,
                rs_size * 4)
            rs_tensor = rs_host.tolist()
            epr = mgr._physical_experts_per_rank
            logger.info(
                "[Routing Debug] rank=%d M=%d "
                "routing_mode=%d NL=%d ws=%d mr=%d "
                "epr=%d threshold=%d "
                "prefill_mode=%d",
                rank, M, routing_mode,
                NL, ws, mr, epr,
                ROUTING_MODE_THRESHOLD,
                PREFILL_ROUTING_MODE)
            # Print l2p map for multi-replica experts
            for e in range(NL):
                if rc[e] > 1:
                    replicas = [
                        l2p[e * mr + i]
                        for i in range(int(rc[e]))]
                    replica_ranks = [
                        p // epr for p in replicas]
                    if routing_mode == 0:
                        sel = rs_tensor[e]
                        sel_rank = (sel // epr
                                    if sel >= 0 else -1)
                        logger.info(
                            "  expert %d: rc=%d "
                            "l2p=%s (ranks %s) "
                            "sel=%d (rank %d)",
                            e, int(rc[e]),
                            replicas, replica_ranks,
                            sel, sel_rank)
                    else:
                        sels = [
                            rs_tensor[s * NL + e]
                            for s in range(ws)]
                        sel_ranks = [
                            p // epr if p >= 0 else -1
                            for p in sels]
                        logger.info(
                            "  expert %d: rc=%d "
                            "l2p=%s (ranks %s) "
                            "section_sel=%s "
                            "(ranks %s)",
                            e, int(rc[e]),
                            replicas, replica_ranks,
                            sels, sel_ranks)

        # Compact after fused kernel.
        mgr.gpu_dar_compact(
            mc, num_experts)
        self.record_expert_event('compact_done')
        # Actual compact count from per-expert token
        # counts (set by Phase D2 atomicAdd). Avoids
        # passing worst-case mc to fused_moe.
        mc_actual = int(expert_num_tokens.sum().item())
        if mc_actual <= 0:
            mc_actual = mc  # fallback
        expert_topk_ids = (
            mgr.compact_expert_topk_ids_buf[
                :mc_actual]
            .unsqueeze(1))
        expert_topk_weights = (
            mgr.compact_expert_topk_weights_buf[
                :mc_actual]
            .unsqueeze(1))
        # Record per-physical-expert load for EPLB
        # rebalancing. expert_num_tokens already has
        # physical expert counts from the fused kernel.
        if self.expert_load_view is not None:
            self.expert_load_view.add_(
                expert_num_tokens)

        # Store for profiling (must be before _receiver
        # lambda — accumulate_expert_times in _finalize
        # reads these after expert compute).
        self._expert_M = M
        self._expert_local_tokens = mc_actual
        self._expert_num_tokens = expert_num_tokens[
            self.rank_expert_offset:
            self.rank_expert_offset
            + self.num_local_experts]
        # Router unique expert count (before dispatch).
        if (mgr._profiling_enabled
                and mgr._profiling_after_rebalance):
            ids = topk_ids.view(-1)
            self._router_unique = int(
                ids.unique().numel())
            self._router_total = int(ids.numel())
            # Sanity check: log topk_ids stats once
            # per layer to verify they are logical IDs.
            if not hasattr(self, '_topk_sanity_logged'):
                self._topk_sanity_logged = True
                logger.info(
                    "topk_ids_check [rank %d layer %d] "
                    "shape=%s dtype=%s min=%d max=%d "
                    "unique=%d/%d M=%d topk=%d",
                    mgr.rank, self._moe_layer_idx,
                    list(topk_ids.shape),
                    topk_ids.dtype,
                    int(ids.min()), int(ids.max()),
                    self._router_unique,
                    self._router_total, M, topk)
            # Accumulate per-expert selection histogram
            # per layer. Print every 100 calls.
            if not hasattr(self, '_router_hist'):
                self._router_hist = torch.zeros(
                    num_experts, dtype=torch.int64,
                    device='cpu')
                self._router_hist_count = 0
            self._router_hist.scatter_add_(
                0, ids.long().cpu(),
                torch.ones_like(
                    ids, dtype=torch.int64,
                    device='cpu'))
            self._router_hist_count += 1
            if self._router_hist_count % 100 == 0:
                h = self._router_hist.tolist()
                n_dead = sum(1 for x in h if x == 0)
                top10 = sorted(
                    enumerate(h), key=lambda x: -x[1]
                )[:10]
                top10_s = ' '.join(
                    f'e{i}:{c}' for i, c in top10)
                logger.info(
                    "router_hist [rank %d] "
                    "calls=%d dead=%d/%d top10=[%s]",
                    mgr.rank,
                    self._router_hist_count,
                    n_dead, num_experts, top10_s)

        return lambda: self._receiver(
            a1, K, num_experts, quant_config,
            expert_map, expert_topk_ids,
            expert_topk_weights, expert_num_tokens,
            mc_actual)

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
        mc: int,
    ) -> mk.PrepareResultType:
        mgr = self.p2p_manager
        self.record_expert_event('recv_start')

        # Token data gather is now fused into
        # dar_compact_kernel (vectorized int4 copy).
        # No separate index_select needed.
        expert_x = mgr.expert_x_buf[:mc]
        self.record_expert_event('gather_done')

        # Post-dispatch quantization.
        # Always call quantize (no numel guard) for
        # CUDA graph compatibility.
        expert_x_scale = None
        if not quant_config.is_block_quantized:
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
        mc_full = self.max_recv
        mgr = self.p2p_manager

        # Weight multiplication is handled by
        # combine_and_scatter kernel: dispatch_meta stores
        # the real topk_weights (set during dispatch_and_route)
        # and applies them during P2P accumulation.
        # expert_topk_weights returned by dispatch are all 1.0,
        # so TopKWeightAndReduceContiguous.apply() was a no-op
        # (multiply by 1.0 + trivial topk=1 sum).
        # Skipping it saves ~2.4ms/layer.

        # Combine + barrier + scatter-add.
        # Use mc_full (not mc_compact) because combine
        # reads dispatch_meta at original IPC positions
        # and uses compact_reverse to index expert_output.
        self.record_expert_event('combine_start')
        meta_bytes = (
            mgr.dispatch_meta_tensor[:mc_full]
            .contiguous().view(torch.uint8))
        # Pass layer context for comm profiling.
        mgr._current_layer_idx = self._moe_layer_idx
        mgr._current_print_counts = (
            self._comm_print_counts)
        mgr.gpu_combine_and_scatter(
            fused_expert_output,
            meta_bytes,
            output,
            mc_full)
        self.accumulate_expert_times()

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
