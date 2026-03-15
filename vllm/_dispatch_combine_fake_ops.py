# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
register_fake (FakeTensor) implementations for all
_C_dispatch_combine custom ops.

Without these, torch.compile() cannot trace through
dispatch_combine ops and creates graph breaks at every
MoE layer, causing inter-GPU desynchronization when
using EP (expert parallel) with P2P IPC barriers.

Import this module to ensure registration happens
before any torch.compile tracing.
"""

import contextlib
from typing import TYPE_CHECKING

import torch

# Op schemas must be registered (via vllm._C import)
# before register_fake decorators can bind to them.
with contextlib.suppress(ImportError):
    import vllm._C  # noqa: F401

if TYPE_CHECKING:

    def register_fake(fn):
        return lambda name: fn
else:
    try:
        from torch.library import register_fake
    except ImportError:
        from torch.library import impl_abstract as register_fake


# ============================================================
# Utility ops
# ============================================================

@register_fake("_C_dispatch_combine::wrap_cuda_ptr")
def _wrap_cuda_ptr_fake(
    dummy: torch.Tensor,
    ptr: int,
    dim0: int,
    dim1: int,
    dtype_code: int,
) -> torch.Tensor:
    # dtype_code: 0=bf16, 1=fp16, 2=int32, 3=int64, 4=fp32
    dtype_map = {
        0: torch.bfloat16,
        1: torch.float16,
        2: torch.int32,
        3: torch.int64,
        4: torch.float32,
    }
    dtype = dtype_map.get(dtype_code, torch.float32)
    if dim1 <= 1:
        return torch.empty(dim0, dtype=dtype,
                           device=dummy.device)
    return torch.empty(dim0, dim1, dtype=dtype,
                       device=dummy.device)


# ============================================================
# Dispatch ops (forward pass)
# ============================================================

@register_fake("_C_dispatch_combine::dispatch_p2p")
def _dispatch_p2p_fake(
    input: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    config_tensor: torch.Tensor,
    M: int,
    K: int,
    topk: int,
) -> None:
    return


@register_fake("_C_dispatch_combine::prepare_dispatch_recv")
def _prepare_dispatch_recv_fake(
    dispatch_recv: torch.Tensor,
    expert_topk_ids: torch.Tensor,
    expert_topk_weights: torch.Tensor,
    expert_num_tokens: torch.Tensor,
    config_tensor: torch.Tensor,
    mc: int,
    K: int,
    num_experts: int,
) -> None:
    return


@register_fake(
    "_C_dispatch_combine::dispatch_and_route")
def _dispatch_and_route_fake(
    input: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    dispatch_recv: torch.Tensor,
    expert_topk_ids: torch.Tensor,
    expert_topk_weights: torch.Tensor,
    expert_num_tokens: torch.Tensor,
    expert_counts: torch.Tensor,
    data_remap: torch.Tensor,
    config_tensor: torch.Tensor,
    M: int,
    K: int,
    topk: int,
    mc: int,
    num_physical_experts: int,
    num_logical_experts: int,
    world_size: int,
) -> None:
    return


# ============================================================
# Combine ops (forward pass)
# ============================================================

@register_fake("_C_dispatch_combine::combine_p2p")
def _combine_p2p_fake(
    expert_output: torch.Tensor,
    dispatch_meta: torch.Tensor,
    compact_reverse: torch.Tensor,
    config_tensor: torch.Tensor,
    max_recv: int,
    K: int,
) -> None:
    return


@register_fake(
    "_C_dispatch_combine::combine_and_scatter")
def _combine_and_scatter_fake(
    expert_output: torch.Tensor,
    dispatch_meta: torch.Tensor,
    compact_reverse: torch.Tensor,
    output: torch.Tensor,
    config_tensor: torch.Tensor,
    mc: int,
    K: int,
    M: int,
) -> None:
    return


@register_fake(
    "_C_dispatch_combine::scatter_add_direct")
def _scatter_add_direct_fake(
    output: torch.Tensor,
    config_tensor: torch.Tensor,
    mc: int,
    K: int,
    M: int,
) -> None:
    return


# ============================================================
# Barrier ops
# ============================================================

@register_fake("_C_dispatch_combine::p2p_barrier")
def _p2p_barrier_fake(
    config_tensor: torch.Tensor,
) -> None:
    return


@register_fake(
    "_C_dispatch_combine::p2p_barrier_reset_dispatch")
def _p2p_barrier_reset_dispatch_fake(
    config_tensor: torch.Tensor,
) -> None:
    return


@register_fake("_C_dispatch_combine::dar_compact")
def _dar_compact_fake(
    expert_topk_ids: torch.Tensor,
    expert_topk_weights: torch.Tensor,
    data_remap: torch.Tensor,
    compact_expert_topk_ids: torch.Tensor,
    compact_expert_topk_weights: torch.Tensor,
    compact_data_remap: torch.Tensor,
    compact_reverse: torch.Tensor,
    config_tensor: torch.Tensor,
    mc_compact: int,
    num_physical_experts: int,
) -> None:
    return
