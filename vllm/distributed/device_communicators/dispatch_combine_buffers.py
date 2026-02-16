# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P2P buffer manager for dispatch_combine all2all backend.

Manages CUDA IPC buffer allocation and handle exchange for direct
GPU-to-GPU P2P memory access between EP ranks. Each rank allocates
receive buffers that remote ranks can write into directly via NVLink.
"""
import ctypes
from typing import Optional

import torch
import torch.distributed as dist

from vllm.logger import init_logger

logger = init_logger(__name__)


class DispatchCombineP2PManager:
    """Manages P2P buffer allocation and IPC handle exchange
    for dispatch_combine all2all communication.

    Buffer layout per rank:
    - dispatch_recv_buf: (max_recv, hidden_dim) - receives dispatched tokens
    - dispatch_meta_buf: (max_recv, 4) int32 - receives token metadata
    - dispatch_offset: (1,) int32 - atomic write offset counter
    - combine_recv_buf: (max_recv, hidden_dim) - receives combine results
    - combine_meta_buf: (max_recv, 4) int32 - receives combine metadata
    - combine_offset: (1,) int32 - atomic write offset counter
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        cpu_group,
        max_num_tokens: int,
        hidden_dim: int,
        topk: int,
        dtype: torch.dtype,
    ):
        self.rank = rank
        self.world_size = world_size
        self.cpu_group = cpu_group
        self.max_num_tokens = max_num_tokens
        self.hidden_dim = hidden_dim
        self.topk = topk
        self.dtype = dtype

        # Max tokens any rank can receive = all tokens from all ranks
        # could route to this rank's experts. Conservative upper bound.
        self.max_recv = max_num_tokens * topk * world_size

        device = torch.cuda.current_device()

        # Allocate local receive buffers.
        self.dispatch_recv_buf = torch.zeros(
            (self.max_recv, hidden_dim),
            dtype=dtype, device=device)
        self.dispatch_meta_buf = torch.zeros(
            (self.max_recv, 4),
            dtype=torch.int32, device=device)
        self.dispatch_offset = torch.zeros(
            1, dtype=torch.int32, device=device)

        self.combine_recv_buf = torch.zeros(
            (self.max_recv, hidden_dim),
            dtype=dtype, device=device)
        self.combine_meta_buf = torch.zeros(
            (self.max_recv, 4),
            dtype=torch.int32, device=device)
        self.combine_offset = torch.zeros(
            1, dtype=torch.int32, device=device)

        # Exchange CUDA IPC handles for P2P access.
        self._setup_p2p_mappings()

        # Build the config tensor for CUDA kernels.
        self.config_tensor = self._build_config_tensor()

        logger.info(
            "DispatchCombineP2PManager initialized: rank=%d, "
            "world_size=%d, max_recv=%d, hidden_dim=%d",
            rank, world_size, self.max_recv, hidden_dim)

    def _setup_p2p_mappings(self):
        """Exchange IPC handles and open remote buffer mappings."""
        from .cuda_wrapper import CudaRTLibrary
        cuda_rt = CudaRTLibrary()

        # Gather IPC handles for all local buffers.
        local_handles = {
            'dispatch_recv': cuda_rt.cudaIpcGetMemHandle(
                ctypes.c_void_p(self.dispatch_recv_buf.data_ptr())),
            'dispatch_meta': cuda_rt.cudaIpcGetMemHandle(
                ctypes.c_void_p(self.dispatch_meta_buf.data_ptr())),
            'dispatch_offset': cuda_rt.cudaIpcGetMemHandle(
                ctypes.c_void_p(self.dispatch_offset.data_ptr())),
            'combine_recv': cuda_rt.cudaIpcGetMemHandle(
                ctypes.c_void_p(self.combine_recv_buf.data_ptr())),
            'combine_meta': cuda_rt.cudaIpcGetMemHandle(
                ctypes.c_void_p(self.combine_meta_buf.data_ptr())),
            'combine_offset': cuda_rt.cudaIpcGetMemHandle(
                ctypes.c_void_p(self.combine_offset.data_ptr())),
        }

        # Exchange handles with all ranks.
        all_handles = [None] * self.world_size
        dist.all_gather_object(all_handles, local_handles,
                               group=self.cpu_group)

        # Open remote handles to get P2P pointers.
        self.remote_dispatch_recv_ptrs = []
        self.remote_dispatch_meta_ptrs = []
        self.remote_dispatch_offset_ptrs = []
        self.remote_combine_recv_ptrs = []
        self.remote_combine_meta_ptrs = []
        self.remote_combine_offset_ptrs = []

        for r in range(self.world_size):
            if r == self.rank:
                # Local rank - use local pointers directly.
                self.remote_dispatch_recv_ptrs.append(
                    self.dispatch_recv_buf.data_ptr())
                self.remote_dispatch_meta_ptrs.append(
                    self.dispatch_meta_buf.data_ptr())
                self.remote_dispatch_offset_ptrs.append(
                    self.dispatch_offset.data_ptr())
                self.remote_combine_recv_ptrs.append(
                    self.combine_recv_buf.data_ptr())
                self.remote_combine_meta_ptrs.append(
                    self.combine_meta_buf.data_ptr())
                self.remote_combine_offset_ptrs.append(
                    self.combine_offset.data_ptr())
            else:
                h = all_handles[r]
                self.remote_dispatch_recv_ptrs.append(
                    cuda_rt.cudaIpcOpenMemHandle(
                        h['dispatch_recv']).value)
                self.remote_dispatch_meta_ptrs.append(
                    cuda_rt.cudaIpcOpenMemHandle(
                        h['dispatch_meta']).value)
                self.remote_dispatch_offset_ptrs.append(
                    cuda_rt.cudaIpcOpenMemHandle(
                        h['dispatch_offset']).value)
                self.remote_combine_recv_ptrs.append(
                    cuda_rt.cudaIpcOpenMemHandle(
                        h['combine_recv']).value)
                self.remote_combine_meta_ptrs.append(
                    cuda_rt.cudaIpcOpenMemHandle(
                        h['combine_meta']).value)
                self.remote_combine_offset_ptrs.append(
                    cuda_rt.cudaIpcOpenMemHandle(
                        h['combine_offset']).value)

    def _build_config_tensor(self) -> torch.Tensor:
        """Build a raw-bytes tensor containing DispatchCombineConfig
        for passing to CUDA kernels."""
        import struct

        max_ranks = 64  # kMaxRanks in .cuh

        # Pack pointers as int64 (8 bytes each)
        data = bytearray()

        # remote_dispatch_recv[kMaxRanks]
        for r in range(max_ranks):
            ptr = self.remote_dispatch_recv_ptrs[r] \
                if r < self.world_size else 0
            data += struct.pack('Q', ptr)

        # remote_dispatch_meta[kMaxRanks]
        for r in range(max_ranks):
            ptr = self.remote_dispatch_meta_ptrs[r] \
                if r < self.world_size else 0
            data += struct.pack('Q', ptr)

        # remote_dispatch_offsets[kMaxRanks]
        for r in range(max_ranks):
            ptr = self.remote_dispatch_offset_ptrs[r] \
                if r < self.world_size else 0
            data += struct.pack('Q', ptr)

        # remote_combine_recv[kMaxRanks]
        for r in range(max_ranks):
            ptr = self.remote_combine_recv_ptrs[r] \
                if r < self.world_size else 0
            data += struct.pack('Q', ptr)

        # remote_combine_meta[kMaxRanks]
        for r in range(max_ranks):
            ptr = self.remote_combine_meta_ptrs[r] \
                if r < self.world_size else 0
            data += struct.pack('Q', ptr)

        # remote_combine_offsets[kMaxRanks]
        for r in range(max_ranks):
            ptr = self.remote_combine_offset_ptrs[r] \
                if r < self.world_size else 0
            data += struct.pack('Q', ptr)

        # Scalar fields
        data += struct.pack('i', self.rank)
        data += struct.pack('i', self.world_size)
        experts_per_rank = 0  # Set later by PrepareAndFinalize
        data += struct.pack('i', experts_per_rank)
        data += struct.pack('i', self.hidden_dim)
        data += struct.pack('i', self.max_num_tokens)

        config_bytes = bytes(data)
        config_tensor = torch.frombuffer(
            bytearray(config_bytes), dtype=torch.uint8
        ).cuda()
        return config_tensor

    def update_experts_per_rank(self, experts_per_rank: int):
        """Update the experts_per_rank field in the config tensor."""
        import struct
        max_ranks = 64
        # Offset: 6 pointer arrays * max_ranks * 8 bytes + 2 int32s
        offset = 6 * max_ranks * 8 + 2 * 4
        packed = struct.pack('i', experts_per_rank)
        # Update in-place on CPU then copy
        cpu_config = self.config_tensor.cpu()
        for i, b in enumerate(packed):
            cpu_config[offset + i] = b
        self.config_tensor.copy_(cpu_config)

    def reset_offsets(self):
        """Reset write offset counters before each dispatch/combine."""
        self.dispatch_offset.zero_()
        self.combine_offset.zero_()

    def get_dispatch_recv_count(self) -> int:
        """Get the number of tokens received in dispatch phase."""
        return self.dispatch_offset.item()

    def get_combine_recv_count(self) -> int:
        """Get the number of results received in combine phase."""
        return self.combine_offset.item()

    def destroy(self):
        """Release resources."""
        # Note: CUDA IPC handles are automatically closed when the
        # process exits. Explicit cleanup is not strictly necessary.
        pass
