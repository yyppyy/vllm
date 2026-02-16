# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P2P buffer manager for dispatch_combine all2all backend.

Manages CUDA IPC buffer allocation and handle exchange for direct
GPU-to-GPU P2P memory access between EP ranks. Each rank allocates
receive buffers that remote ranks can write into directly via NVLink.

Buffers are allocated with cudaMalloc (not PyTorch's caching allocator)
to ensure base pointers required by cudaIpcGetMemHandle.

All runtime operations use GPU-side CUDA kernels (not host-side
cudaMemset/cudaMemcpy) for CUDA graph compatibility.
"""
import ctypes
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

    Pre-allocated PyTorch tensors for GPU-side copy destinations:
    - dispatch_recv_tensor: (max_recv, hidden_dim) dtype
    - dispatch_meta_tensor: (max_recv, 4) int32
    - combine_recv_tensor: (max_recv, hidden_dim) dtype
    - combine_meta_tensor: (max_recv, 4) int32
    - accum_tensor: (max_num_tokens, hidden_dim) float32
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
            "hidden_dim=%d, topk=%d, dtype=%s",
            rank, self._device, world_size,
            max_num_tokens, hidden_dim, topk, str(dtype))

        # Max tokens any rank can receive.
        self.max_recv = max_num_tokens * topk * 2

        # Buffer sizes in bytes.
        self._recv_bytes = (
            self.max_recv * hidden_dim * self._dtype_size)
        self._meta_bytes = self.max_recv * 4 * 4
        self._offset_bytes = 4

        from .cuda_wrapper import CudaRTLibrary
        self._cuda_rt = CudaRTLibrary()

        # Allocate raw IPC buffers with cudaMalloc.
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

        # Pre-allocate PyTorch tensors for GPU-side copy
        # destinations. Fixed size = max_recv for CUDA graph
        # compatibility (no dynamic allocation at runtime).
        self.dispatch_recv_tensor = torch.zeros(
            (self.max_recv, hidden_dim),
            dtype=dtype,
            device=self._device)
        self.dispatch_meta_tensor = torch.zeros(
            (self.max_recv, 4),
            dtype=torch.int32,
            device=self._device)
        self.combine_recv_tensor = torch.zeros(
            (self.max_recv, hidden_dim),
            dtype=dtype,
            device=self._device)
        self.combine_meta_tensor = torch.zeros(
            (self.max_recv, 4),
            dtype=torch.int32,
            device=self._device)

        # Float32 accumulator for scatter-add.
        self.accum_tensor = torch.zeros(
            (max_num_tokens, hidden_dim),
            dtype=torch.float32,
            device=self._device)

        # NCCL barrier tensor (tiny, for all-reduce barrier).
        self._barrier_tensor = torch.zeros(
            1, dtype=torch.int32, device=self._device)

        logger.info(
            "DispatchCombineP2PManager initialized: rank=%d,"
            " world_size=%d, max_recv=%d, hidden_dim=%d",
            rank, world_size, self.max_recv, hidden_dim)

    @staticmethod
    def _handle_to_bytes(handle) -> bytes:
        """Convert cudaIpcMemHandle_t to bytes."""
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
        """Exchange IPC handles and open remote mappings."""
        cuda_rt = self._cuda_rt
        to_bytes = self._handle_to_bytes

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

        all_handles = [None] * self.world_size
        dist.all_gather_object(all_handles, local_handles,
                               group=self.cpu_group)

        self.remote_dispatch_recv_ptrs = []
        self.remote_dispatch_meta_ptrs = []
        self.remote_dispatch_offset_ptrs = []
        self.remote_combine_recv_ptrs = []
        self.remote_combine_meta_ptrs = []
        self.remote_combine_offset_ptrs = []
        from_bytes = self._bytes_to_handle

        for r in range(self.world_size):
            if r == self.rank:
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

    def _build_config_tensor(self) -> torch.Tensor:
        """Build a raw-bytes tensor containing
        DispatchCombineConfig for CUDA kernels."""
        import struct

        max_ranks = 64

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
        # experts_per_rank set later via update_experts_per_rank
        data += struct.pack('i', 0)
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
        offset = 6 * max_ranks * 8 + 2 * 4
        packed = struct.pack('i', experts_per_rank)
        cpu_config = self.config_tensor.cpu()
        for i, b in enumerate(packed):
            cpu_config[offset + i] = b
        self.config_tensor.copy_(cpu_config)

    # ================================================================
    # GPU-side ops (CUDA-graph compatible)
    # ================================================================

    def gpu_reset_offsets(self):
        """Reset dispatch+combine offset counters (GPU kernel).
        """
        torch.ops._C_dispatch_combine.reset_offsets(
            self.config_tensor)

    def gpu_reset_combine_offset(self):
        """Reset combine offset counter (GPU kernel)."""
        torch.ops._C_dispatch_combine.reset_combine_offset(
            self.config_tensor)

    def gpu_copy_dispatch_recv(self):
        """Copy dispatch recv from IPC to pre-allocated tensor.
        """
        torch.ops._C_dispatch_combine.copy_dispatch_recv(
            self.dispatch_recv_tensor,
            self.config_tensor,
            self.max_recv,
            self.hidden_dim)

    def gpu_copy_dispatch_meta(self):
        """Copy dispatch metadata from IPC to tensor."""
        torch.ops._C_dispatch_combine.copy_dispatch_meta(
            self.dispatch_meta_tensor,
            self.config_tensor,
            self.max_recv)

    def gpu_copy_combine_recv(self):
        """Copy combine recv from IPC to tensor."""
        torch.ops._C_dispatch_combine.copy_combine_recv(
            self.combine_recv_tensor,
            self.config_tensor,
            self.max_recv,
            self.hidden_dim)

    def gpu_copy_combine_meta(self):
        """Copy combine metadata from IPC to tensor."""
        torch.ops._C_dispatch_combine.copy_combine_meta(
            self.combine_meta_tensor,
            self.config_tensor,
            self.max_recv)

    def nccl_barrier(self, ep_group):
        """GPU-side barrier via NCCL all-reduce.
        Uses GroupCoordinator.all_reduce which routes
        through torch.ops.vllm.all_reduce custom op
        for CUDA graph compatibility."""
        self._barrier_tensor.fill_(0)
        ep_group.all_reduce(self._barrier_tensor)

    def destroy(self):
        """Release cudaMalloc'd buffers."""
        self._cuda_rt.cudaFree(self._raw_dispatch_recv)
        self._cuda_rt.cudaFree(self._raw_dispatch_meta)
        self._cuda_rt.cudaFree(self._raw_dispatch_offset)
        self._cuda_rt.cudaFree(self._raw_combine_recv)
        self._cuda_rt.cudaFree(self._raw_combine_meta)
        self._cuda_rt.cudaFree(self._raw_combine_offset)
