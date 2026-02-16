# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PrepareAndFinalize for dispatch_combine all2all backend.

Routing-aware dispatch/combine with Standard activation format output,
using custom CUDA P2P kernels for low-latency GPU-to-GPU communication.
"""
from typing import Callable, Optional

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.distributed import get_ep_group
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceContiguous, TopKWeightAndReduceDelegate)
from vllm.model_executor.layers.fused_moe.utils import (
    moe_kernel_quantize_input)

logger = init_logger(__name__)


class DispatchCombinePrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):
    """Routing-aware dispatch/combine with Standard format output.

    Uses custom CUDA P2P kernels for direct GPU-to-GPU token transfer.
    Each rank only receives tokens whose selected experts reside on it.
    Outputs Standard (M_recv, K) format for standard expert kernels.

    The data flow is:
    1. Dispatch (prepare): route tokens to ranks via P2P → quantize
    2. Expert execution: standard TritonExperts on received tokens
    3. Combine (finalize): weight+reduce → send back via P2P → scatter-add
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

        # Update config tensor with experts_per_rank.
        self.p2p_manager.update_experts_per_rank(self.experts_per_rank)

        # State preserved between prepare and finalize.
        self._dispatch_meta_buf = None
        self._dispatch_recv_count = 0
        self._orig_num_tokens = 0

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
        self._orig_num_tokens = M

        if apply_router_weight_on_input:
            assert topk == 1, (
                "apply_router_weight_on_input only implemented for topk=1")
            a1 = a1 * topk_weights.to(a1.dtype)

        # Reset P2P offset counters before dispatch.
        self.p2p_manager.reset_offsets()

        # Synchronize all ranks to ensure offsets are reset.
        torch.cuda.synchronize()
        ep_group = get_ep_group()
        if ep_group.device_communicator.pynccl_comm is not None:
            ep_group.device_communicator.pynccl_comm.stream.synchronize()

        # Launch dispatch P2P kernel.
        # Each (token, expert_slot) pair determines a dest rank and
        # writes the token + metadata to that rank's recv buffer.
        topk_ids_i32 = topk_ids.to(torch.int32)
        topk_weights_f32 = topk_weights.to(torch.float32)

        torch.ops._C_dispatch_combine.dispatch_p2p(
            a1,
            topk_ids_i32,
            topk_weights_f32,
            self.p2p_manager.config_tensor,
            M, K, topk,
        )

        # Synchronize to ensure all P2P writes are complete.
        torch.cuda.synchronize()

        # Barrier: all ranks must finish dispatch before reading.
        if ep_group.device_communicator.pynccl_comm is not None:
            import torch.distributed as dist
            dist.barrier(group=ep_group.cpu_group)
        else:
            import torch.distributed as dist
            dist.barrier()

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
        # Read how many tokens were received.
        M_recv = self.p2p_manager.get_dispatch_recv_count()
        self._dispatch_recv_count = M_recv

        if M_recv == 0:
            # No tokens received for this rank's experts.
            empty_x = torch.empty(
                (0, K), dtype=a1_orig.dtype, device=a1_orig.device)
            empty_ids = torch.empty(
                (0, 1), dtype=torch.int64, device=a1_orig.device)
            empty_weights = torch.empty(
                (0, 1), dtype=torch.float32, device=a1_orig.device)
            expert_tokens_meta = mk.ExpertTokensMetadata(
                expert_num_tokens=torch.zeros(
                    self.num_local_experts, dtype=torch.int32,
                    device=a1_orig.device),
                expert_num_tokens_cpu=torch.zeros(
                    self.num_local_experts, dtype=torch.int32))
            return (empty_x, None, expert_tokens_meta,
                    empty_ids, empty_weights)

        # Slice the recv buffers to actual received count.
        expert_x = self.p2p_manager.dispatch_recv_buf[:M_recv].clone()
        dispatch_meta = self.p2p_manager.dispatch_meta_buf[:M_recv]

        # Save metadata for combine phase.
        self._dispatch_meta_buf = dispatch_meta

        # Extract expert IDs from metadata.
        # Metadata layout: [source_rank, source_token_idx, expert_id, weight]
        expert_topk_ids = dispatch_meta[:, 2].clone()  # (M_recv,)

        # Remap expert IDs from local → global space (like DeepEP HT).
        # The dispatch kernel sends global expert IDs. The standard expert
        # kernels expect global IDs and use expert_map to remap to local.
        # Handle -1 entries (invalid) by mapping to a safe expert.
        expert_topk_ids = torch.where(
            expert_topk_ids == -1,
            num_experts - 1 if self.rank_expert_offset == 0 else 0,
            expert_topk_ids)

        # Shape as (M_recv, 1) for topk=1 per received token-expert pair.
        expert_topk_ids = expert_topk_ids.unsqueeze(1).to(torch.int64)

        # Weights are all 1.0 for expert computation; actual weights
        # are applied in the combine phase.
        expert_topk_weights = torch.ones(
            (M_recv, 1), dtype=torch.float32, device=expert_x.device)

        # Post-dispatch quantization (like DeepEP HT).
        expert_x_scale = None
        if not quant_config.is_block_quantized:
            if expert_x.numel() != 0:
                expert_x, expert_x_scale = moe_kernel_quantize_input(
                    expert_x,
                    quant_config.a1_scale,
                    quant_dtype=quant_config.quant_dtype,
                    per_act_token_quant=False,
                    block_shape=quant_config.block_shape)
        else:
            expert_x, expert_x_scale = moe_kernel_quantize_input(
                expert_x,
                quant_config.a1_scale,
                quant_dtype=quant_config.quant_dtype,
                per_act_token_quant=quant_config.per_act_token_quant,
                block_shape=quant_config.block_shape)

        # Compute expert token counts.
        expert_num_tokens = torch.zeros(
            num_experts, dtype=torch.int32, device=expert_x.device)
        if M_recv > 0:
            flat_ids = expert_topk_ids.view(-1)
            ones = torch.ones_like(flat_ids, dtype=torch.int32)
            expert_num_tokens.scatter_add_(0, flat_ids.to(torch.int64), ones)

        # Slice to local experts only.
        local_expert_num_tokens = expert_num_tokens[
            self.rank_expert_offset:
            self.rank_expert_offset + self.num_local_experts]

        expert_tokens_meta = mk.ExpertTokensMetadata(
            expert_num_tokens=local_expert_num_tokens,
            expert_num_tokens_cpu=local_expert_num_tokens.cpu())

        return (expert_x, expert_x_scale, expert_tokens_meta,
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
            expert_map, apply_router_weight_on_input, quant_config)
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
        M_recv = self._dispatch_recv_count
        K = output.shape[-1]

        # Step 1: Apply weights + reduce on the dispatched tokens.
        # Each received token has topk=1, so this is mostly identity.
        if fused_expert_output.numel() != 0:
            if isinstance(weight_and_reduce_impl,
                          TopKWeightAndReduceDelegate):
                weight_and_reduce_impl = TopKWeightAndReduceContiguous()
            fused_expert_output = weight_and_reduce_impl.apply(
                output=None,
                fused_expert_output=fused_expert_output,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )

        # Reset combine offset counters.
        self.p2p_manager.combine_offset.zero_()
        torch.cuda.synchronize()

        # Step 2: Launch combine P2P kernel.
        # Send expert outputs back to originating ranks.
        if M_recv > 0:
            # Reinterpret dispatch_meta_buf as raw bytes for CUDA kernel.
            meta_bytes = self._dispatch_meta_buf.view(-1).to(
                torch.uint8).contiguous()
            torch.ops._C_dispatch_combine.combine_p2p(
                fused_expert_output,
                meta_bytes,
                self.p2p_manager.config_tensor,
                M_recv, K,
            )

        # Synchronize and barrier for combine completion.
        torch.cuda.synchronize()
        ep_group = get_ep_group()
        import torch.distributed as dist
        dist.barrier(group=ep_group.cpu_group)

        # Step 3: Read combine results and scatter-add to output.
        N_recv = self.p2p_manager.get_combine_recv_count()

        if N_recv > 0:
            combine_recv = self.p2p_manager.combine_recv_buf[:N_recv]
            combine_meta_bytes = self.p2p_manager.combine_meta_buf[
                :N_recv].view(-1).to(torch.uint8).contiguous()

            # Use float32 accumulator for precise atomic scatter-add,
            # then convert back to output dtype.
            accum = torch.zeros(
                output.shape, dtype=torch.float32,
                device=output.device)
            torch.ops._C_dispatch_combine.scatter_add_weighted(
                accum,
                combine_recv,
                combine_meta_bytes,
                N_recv, K,
            )
            output.copy_(accum.to(output.dtype))
        else:
            output.zero_()

        if do_async:
            return lambda: None  # Already complete
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
            output, fused_expert_output, topk_weights, topk_ids,
            apply_router_weight_on_input, weight_and_reduce_impl,
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
            output, fused_expert_output, topk_weights, topk_ids,
            apply_router_weight_on_input, weight_and_reduce_impl,
            do_async=False)
