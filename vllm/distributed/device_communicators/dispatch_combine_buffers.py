# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P2P buffer manager for dispatch_combine all2all backend.

Manages CUDA IPC buffer allocation and handle exchange for direct
GPU-to-GPU P2P memory access between EP ranks. Each rank allocates
receive buffers that remote ranks can write into directly via NVLink.

Buffers are allocated with cudaMalloc (not PyTorch's caching allocator)
to ensure base pointers required by cudaIpcGetMemHandle.
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

    Buffer layout per rank (all allocated via cudaMalloc):
    - dispatch_recv: (max_recv * hidden_dim * dtype_size) bytes
    - dispatch_meta: (max_recv * 16) bytes  (4 int32s per entry)
    - dispatch_offset: 4 bytes (1 int32 atomic counter)
    - combine_recv: (max_recv * hidden_dim * dtype_size) bytes
    - combine_meta: (max_recv * 16) bytes
    - combine_offset: 4 bytes
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
        self._dtype_size = torch.tensor(
            [], dtype=dtype).element_size()
        self._device = torch.cuda.current_device()

        logger.info(
            "DispatchCombineP2PManager: rank=%d, device=%d, "
            "world_size=%d, max_num_tokens=%d, "
            "hidden_dim=%d, topk=%d, dtype=%s, "
            "dtype_size=%d",
            rank, self._device, world_size,
            max_num_tokens, hidden_dim, topk,
            str(dtype), self._dtype_size)

        # Max tokens any rank can receive. With uniform routing
        # each rank receives ~(max_num_tokens * topk) entries.
        # Use 2x safety factor instead of the theoretical max
        # (max_num_tokens * topk * world_size) which consumes
        # too much GPU memory. The kernel bounds-checks against
        # max_recv so overflow entries are safely dropped.
        self.max_recv = max_num_tokens * topk * 2

        # Buffer sizes in bytes.
        self._recv_bytes = (
            self.max_recv * hidden_dim * self._dtype_size)
        self._meta_bytes = self.max_recv * 4 * 4  # 4 int32s
        self._offset_bytes = 4  # 1 int32

        from .cuda_wrapper import CudaRTLibrary
        self._cuda_rt = CudaRTLibrary()

        # Allocate raw IPC buffers with cudaMalloc.
        # cudaMalloc returns base pointers required by
        # cudaIpcGetMemHandle (PyTorch's caching allocator
        # may return sub-allocated pointers that are invalid
        # for IPC).
        self._raw_dispatch_recv = self._cuda_rt.cudaMalloc(
            self._recv_bytes)
        self._cuda_rt.cudaMemset(
            self._raw_dispatch_recv, 0, self._recv_bytes)

        self._raw_dispatch_meta = self._cuda_rt.cudaMalloc(
            self._meta_bytes)
        self._cuda_rt.cudaMemset(
            self._raw_dispatch_meta, 0, self._meta_bytes)

        self._raw_dispatch_offset = self._cuda_rt.cudaMalloc(
            self._offset_bytes)
        self._cuda_rt.cudaMemset(
            self._raw_dispatch_offset, 0, self._offset_bytes)

        self._raw_combine_recv = self._cuda_rt.cudaMalloc(
            self._recv_bytes)
        self._cuda_rt.cudaMemset(
            self._raw_combine_recv, 0, self._recv_bytes)

        self._raw_combine_meta = self._cuda_rt.cudaMalloc(
            self._meta_bytes)
        self._cuda_rt.cudaMemset(
            self._raw_combine_meta, 0, self._meta_bytes)

        self._raw_combine_offset = self._cuda_rt.cudaMalloc(
            self._offset_bytes)
        self._cuda_rt.cudaMemset(
            self._raw_combine_offset, 0, self._offset_bytes)

        # Exchange CUDA IPC handles for P2P access.
        self._setup_p2p_mappings()

        # Build the config tensor for CUDA kernels.
        self.config_tensor = self._build_config_tensor()

        logger.info(
            "DispatchCombineP2PManager initialized: rank=%d, "
            "world_size=%d, max_recv=%d, hidden_dim=%d",
            rank, world_size, self.max_recv, hidden_dim)

    @staticmethod
    def _handle_to_bytes(handle) -> bytes:
        """Convert cudaIpcMemHandle_t to bytes for pickling."""
        return bytes(handle)

    @staticmethod
    def _bytes_to_handle(data: bytes):
        """Convert bytes back to cudaIpcMemHandle_t."""
        from .cuda_wrapper import cudaIpcMemHandle_t
        handle = cudaIpcMemHandle_t()
        ctypes.memmove(ctypes.byref(handle), data,
                       min(len(data), ctypes.sizeof(handle)))
        return handle

    def _setup_p2p_mappings(self):
        """Exchange IPC handles and open remote buffer mappings.

        Handles are converted to raw bytes before exchange via
        all_gather_object to avoid any ctypes pickling issues
        (matches the pattern used by custom_all_reduce).
        """
        cuda_rt = self._cuda_rt
        to_bytes = self._handle_to_bytes

        # Get IPC handles and convert to bytes for exchange.
        local_handles = {
            'dispatch_recv': to_bytes(
                cuda_rt.cudaIpcGetMemHandle(
                    self._raw_dispatch_recv)),
            'dispatch_meta': to_bytes(
                cuda_rt.cudaIpcGetMemHandle(
                    self._raw_dispatch_meta)),
            'dispatch_offset': to_bytes(
                cuda_rt.cudaIpcGetMemHandle(
                    self._raw_dispatch_offset)),
            'combine_recv': to_bytes(
                cuda_rt.cudaIpcGetMemHandle(
                    self._raw_combine_recv)),
            'combine_meta': to_bytes(
                cuda_rt.cudaIpcGetMemHandle(
                    self._raw_combine_meta)),
            'combine_offset': to_bytes(
                cuda_rt.cudaIpcGetMemHandle(
                    self._raw_combine_offset)),
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
        from_bytes = self._bytes_to_handle

        for r in range(self.world_size):
            if r == self.rank:
                # Local rank - use local raw pointers.
                self.remote_dispatch_recv_ptrs.append(
                    self._raw_dispatch_recv.value)
                self.remote_dispatch_meta_ptrs.append(
                    self._raw_dispatch_meta.value)
                self.remote_dispatch_offset_ptrs.append(
                    self._raw_dispatch_offset.value)
                self.remote_combine_recv_ptrs.append(
                    self._raw_combine_recv.value)
                self.remote_combine_meta_ptrs.append(
                    self._raw_combine_meta.value)
                self.remote_combine_offset_ptrs.append(
                    self._raw_combine_offset.value)
            else:
                h = all_handles[r]
                self.remote_dispatch_recv_ptrs.append(
                    cuda_rt.cudaIpcOpenMemHandle(
                        from_bytes(
                            h['dispatch_recv'])).value)
                self.remote_dispatch_meta_ptrs.append(
                    cuda_rt.cudaIpcOpenMemHandle(
                        from_bytes(
                            h['dispatch_meta'])).value)
                self.remote_dispatch_offset_ptrs.append(
                    cuda_rt.cudaIpcOpenMemHandle(
                        from_bytes(
                            h['dispatch_offset'])).value)
                self.remote_combine_recv_ptrs.append(
                    cuda_rt.cudaIpcOpenMemHandle(
                        from_bytes(
                            h['combine_recv'])).value)
                self.remote_combine_meta_ptrs.append(
                    cuda_rt.cudaIpcOpenMemHandle(
                        from_bytes(
                            h['combine_meta'])).value)
                self.remote_combine_offset_ptrs.append(
                    cuda_rt.cudaIpcOpenMemHandle(
                        from_bytes(
                            h['combine_offset'])).value)

        # Validate all pointers are non-NULL.
        for r in range(self.world_size):
            assert self.remote_dispatch_recv_ptrs[r], \
                f"NULL dispatch_recv ptr for rank {r}"
            assert self.remote_dispatch_meta_ptrs[r], \
                f"NULL dispatch_meta ptr for rank {r}"
            assert self.remote_dispatch_offset_ptrs[r], \
                f"NULL dispatch_offset ptr for rank {r}"
            assert self.remote_combine_recv_ptrs[r], \
                f"NULL combine_recv ptr for rank {r}"
            assert self.remote_combine_meta_ptrs[r], \
                f"NULL combine_meta ptr for rank {r}"
            assert self.remote_combine_offset_ptrs[r], \
                f"NULL combine_offset ptr for rank {r}"

        # Log pointer values for debugging.
        for r in range(self.world_size):
            logger.info(
                "Rank %d → remote rank %d ptrs: "
                "recv=0x%x meta=0x%x offset=0x%x "
                "c_recv=0x%x c_meta=0x%x c_off=0x%x",
                self.rank, r,
                self.remote_dispatch_recv_ptrs[r],
                self.remote_dispatch_meta_ptrs[r],
                self.remote_dispatch_offset_ptrs[r],
                self.remote_combine_recv_ptrs[r],
                self.remote_combine_meta_ptrs[r],
                self.remote_combine_offset_ptrs[r])

        # Validate IPC pointers by reading 4 bytes from each
        # remote buffer via cudaMemcpy. This catches invalid
        # IPC handles before any kernel launch.
        tmp = torch.empty(
            1, dtype=torch.int32, device=self._device)
        for r in range(self.world_size):
            if r == self.rank:
                continue
            try:
                self._cuda_rt.cudaMemcpy(
                    ctypes.c_void_p(tmp.data_ptr()),
                    ctypes.c_void_p(
                        self.remote_dispatch_recv_ptrs[r]),
                    4)
            except RuntimeError as e:
                logger.error(
                    "Rank %d: IPC validation FAILED for "
                    "rank %d dispatch_recv ptr 0x%x: %s",
                    self.rank, r,
                    self.remote_dispatch_recv_ptrs[r], e)
                raise
            try:
                self._cuda_rt.cudaMemcpy(
                    ctypes.c_void_p(tmp.data_ptr()),
                    ctypes.c_void_p(
                        self.remote_dispatch_offset_ptrs[r]),
                    4)
            except RuntimeError as e:
                logger.error(
                    "Rank %d: IPC validation FAILED for "
                    "rank %d dispatch_offset ptr 0x%x: %s",
                    self.rank, r,
                    self.remote_dispatch_offset_ptrs[r], e)
                raise
        logger.info(
            "Rank %d: IPC pointer validation passed "
            "for all %d remote ranks.",
            self.rank, self.world_size - 1)

    def _build_config_tensor(self) -> torch.Tensor:
        """Build a raw-bytes tensor containing
        DispatchCombineConfig for passing to CUDA kernels."""
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

        # Scalar fields (must match DispatchCombineConfig)
        data += struct.pack('i', self.rank)
        data += struct.pack('i', self.world_size)
        experts_per_rank = 0  # Set later via update_experts_per_rank
        data += struct.pack('i', experts_per_rank)
        data += struct.pack('i', self.hidden_dim)
        data += struct.pack('i', self.max_num_tokens)
        data += struct.pack('i', self.max_recv)

        config_bytes = bytes(data)
        config_tensor = torch.frombuffer(
            bytearray(config_bytes), dtype=torch.uint8
        ).cuda()
        return config_tensor

    def update_experts_per_rank(self, experts_per_rank: int):
        """Update experts_per_rank in the config tensor."""
        import struct
        max_ranks = 64
        # Offset: 6 pointer arrays * max_ranks * 8 + 2 int32s
        offset = 6 * max_ranks * 8 + 2 * 4
        packed = struct.pack('i', experts_per_rank)
        cpu_config = self.config_tensor.cpu()
        for i, b in enumerate(packed):
            cpu_config[offset + i] = b
        self.config_tensor.copy_(cpu_config)

    def reset_offsets(self):
        """Reset write offset counters before dispatch."""
        self._cuda_rt.cudaMemset(
            self._raw_dispatch_offset, 0, self._offset_bytes)
        self._cuda_rt.cudaMemset(
            self._raw_combine_offset, 0, self._offset_bytes)

    def reset_combine_offset(self):
        """Reset only the combine offset counter."""
        self._cuda_rt.cudaMemset(
            self._raw_combine_offset, 0, self._offset_bytes)

    def get_dispatch_recv_count(self) -> int:
        """Get number of tokens received in dispatch phase."""
        tmp = torch.empty(
            1, dtype=torch.int32, device=self._device)
        self._cuda_rt.cudaMemcpy(
            ctypes.c_void_p(tmp.data_ptr()),
            self._raw_dispatch_offset,
            self._offset_bytes)
        return tmp.item()

    def get_combine_recv_count(self) -> int:
        """Get number of results received in combine phase."""
        tmp = torch.empty(
            1, dtype=torch.int32, device=self._device)
        self._cuda_rt.cudaMemcpy(
            ctypes.c_void_p(tmp.data_ptr()),
            self._raw_combine_offset,
            self._offset_bytes)
        return tmp.item()

    def read_dispatch_recv(self, count: int) -> torch.Tensor:
        """Copy dispatched tokens from IPC buffer to a tensor."""
        nbytes = count * self.hidden_dim * self._dtype_size
        tensor = torch.empty(
            (count, self.hidden_dim),
            dtype=self.dtype, device=self._device)
        self._cuda_rt.cudaMemcpy(
            ctypes.c_void_p(tensor.data_ptr()),
            self._raw_dispatch_recv,
            nbytes)
        return tensor

    def read_dispatch_meta(self, count: int) -> torch.Tensor:
        """Copy dispatch metadata from IPC buffer to a tensor.
        Returns (count, 4) int32 tensor."""
        nbytes = count * 4 * 4  # 4 int32 fields per entry
        tensor = torch.empty(
            (count, 4),
            dtype=torch.int32, device=self._device)
        self._cuda_rt.cudaMemcpy(
            ctypes.c_void_p(tensor.data_ptr()),
            self._raw_dispatch_meta,
            nbytes)
        return tensor

    def read_combine_recv(self, count: int) -> torch.Tensor:
        """Copy combine results from IPC buffer to a tensor."""
        nbytes = count * self.hidden_dim * self._dtype_size
        tensor = torch.empty(
            (count, self.hidden_dim),
            dtype=self.dtype, device=self._device)
        self._cuda_rt.cudaMemcpy(
            ctypes.c_void_p(tensor.data_ptr()),
            self._raw_combine_recv,
            nbytes)
        return tensor

    def read_combine_meta_bytes(
            self, count: int) -> torch.Tensor:
        """Copy combine metadata as raw bytes for CUDA kernel.
        Returns 1-D uint8 tensor of size count * 16."""
        nbytes = count * 4 * 4  # 16 bytes per entry
        tensor = torch.empty(
            nbytes, dtype=torch.uint8, device=self._device)
        self._cuda_rt.cudaMemcpy(
            ctypes.c_void_p(tensor.data_ptr()),
            self._raw_combine_meta,
            nbytes)
        return tensor

    def destroy(self):
        """Release cudaMalloc'd buffers."""
        self._cuda_rt.cudaFree(self._raw_dispatch_recv)
        self._cuda_rt.cudaFree(self._raw_dispatch_meta)
        self._cuda_rt.cudaFree(self._raw_dispatch_offset)
        self._cuda_rt.cudaFree(self._raw_combine_recv)
        self._cuda_rt.cudaFree(self._raw_combine_meta)
        self._cuda_rt.cudaFree(self._raw_combine_offset)
