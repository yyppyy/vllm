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
import os
import torch
import torch.distributed as dist

import vllm._dispatch_combine_fake_ops  # noqa: F401
from vllm.logger import init_logger

# Fine-grained kernel profiling.
# Set VLLM_DC_PROFILE=N to print per-step averages
# every N batches. 0 = disabled.
_DC_PROFILE_INTERVAL = int(
    os.environ.get('VLLM_DC_PROFILE', '0'))

# Must match kDarNumSteps, kCasNumSteps, kTotalProfileSlots
# in dispatch_combine.cuh.
_DAR_NUM_STEPS = 10
_CAS_NUM_STEPS = 6
_TOTAL_PROFILE_SLOTS = _DAR_NUM_STEPS + _CAS_NUM_STEPS

_DAR_STEP_NAMES = [
    "read_counters",
    "scan_write",
    "expert_flush",
    "threadfence_sys",
    "grid_sync",
    "expert_push",
    "barrier",
    "phase_c_route",
    "phase_d2_filter",
    "end",
]

_CAS_STEP_NAMES = [
    "read_counters",
    "zero_accum",
    "scan_write",
    "grid_sync",
    "barrier",
    "end",
]

logger = init_logger(__name__)


class DispatchCombineP2PManager:
    """Manages P2P buffer allocation and IPC handle exchange
    for dispatch_combine all2all communication.

    Buffer layout per rank (all allocated via cudaMalloc):
    - dispatch_recv: (max_recv * hidden_dim * dtype_size) bytes
    - dispatch_meta: (max_recv * 16) bytes  (4 int32s per entry)
    - dispatch_offset: (64 * 4) bytes (int32[kMaxRanks] per-sender)
    - combine_recv: (max_recv * hidden_dim * dtype_size) bytes
    - combine_meta: (max_recv * 16) bytes
    - combine_offset: (64 * 4) bytes (int32[kMaxRanks] per-sender)
    - local_dispatch_counters: (64 * 4) bytes (int32[kMaxRanks])
    - local_combine_counters: (64 * 4) bytes (int32[kMaxRanks])

    Non-owning PyTorch tensor views over IPC buffers
    (via wrap_cuda_ptr):
    - dispatch_recv_tensor: (max_recv, hidden_dim) dtype
    - dispatch_meta_tensor: (max_recv, 4) int32
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
        # Per-sender section offsets: int32[kMaxRanks].
        self._offset_bytes = 4 * 64

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

        # Local per-destination counters for local atomicAdd.
        # Position = rank * section_size + local_counter.
        self._local_counter_bytes = 4 * 64  # int32[kMaxRanks]
        self._raw_local_dispatch_counters = (
            self._cuda_rt.cudaMalloc(
                self._local_counter_bytes))
        self._cuda_rt.cudaMemset(
            self._raw_local_dispatch_counters, 0,
            self._local_counter_bytes)
        self._raw_local_combine_counters = (
            self._cuda_rt.cudaMalloc(
                self._local_counter_bytes))
        self._cuda_rt.cudaMemset(
            self._raw_local_combine_counters, 0,
            self._local_counter_bytes)

        # Grid-wide sync counter for fused combine kernel.
        self._raw_combine_done_counter = (
            self._cuda_rt.cudaMalloc(4))
        self._cuda_rt.cudaMemset(
            self._raw_combine_done_counter, 0, 4)

        # P2P barrier signal buffer.
        # Layout: alignas(128) flags[64] (256 bytes)
        #         + counter (4 bytes) = 260 bytes.
        # Allocate 512 bytes for safety.
        self._signal_bytes = 512
        self._raw_signals = self._cuda_rt.cudaMalloc(
            self._signal_bytes)
        self._cuda_rt.cudaMemset(
            self._raw_signals, 0, self._signal_bytes)

        # Integrated routing buffers (EPLB).
        # Allocated lazily; set to None until
        # init_integrated_routing() is called.
        self._raw_expert_counts = None
        self._raw_local_expert_counts = None
        self._raw_routing_selection = None
        self._raw_routing_ready_flag = None
        self._raw_phase_a_done_counter = None
        self._routing_map_tensor = None
        self._routing_count_tensor = None
        self._num_logical_experts = 0
        self._max_replicas = 0
        self._physical_experts_per_rank = 0
        self.remote_expert_counts_ptrs = []
        self._integrated_routing_enabled = False
        self._experts_per_rank = 0

        # Fine-grained profiling.
        self._profiling_enabled = (_DC_PROFILE_INTERVAL > 0)
        self._profiling_interval = _DC_PROFILE_INTERVAL
        self._raw_profiling_timestamps = None
        if self._profiling_enabled:
            # int64[kTotalProfileSlots] on GPU.
            ts_bytes = _TOTAL_PROFILE_SLOTS * 8
            self._raw_profiling_timestamps = (
                self._cuda_rt.cudaMalloc(ts_bytes))
            self._cuda_rt.cudaMemset(
                self._raw_profiling_timestamps,
                0, ts_bytes)
            # Host-side accumulators (nanoseconds).
            self._dar_accum = [0.0] * _DAR_NUM_STEPS
            self._cas_accum = [0.0] * _CAS_NUM_STEPS
            self._profile_batch_count = 0
            logger.info(
                "DC profiling enabled: print every "
                "%d batches", self._profiling_interval)

        # Exchange CUDA IPC handles for P2P access.
        self._setup_p2p_mappings()

        # Build the config tensor for CUDA kernels.
        self.config_tensor = self._build_config_tensor()

        # Wrap IPC buffers as non-owning PyTorch tensors.
        # No runtime copy needed; expert compute reads
        # directly from IPC memory.
        # dtype_code: 0=bf16, 1=fp16, 2=int32
        dtype_code = (
            0 if dtype == torch.bfloat16 else 1)
        ct = self.config_tensor  # dummy for dispatch
        self.dispatch_recv_tensor = (
            torch.ops._C_dispatch_combine.wrap_cuda_ptr(
                ct, self._raw_dispatch_recv.value,
                self.max_recv, hidden_dim,
                dtype_code))
        self.dispatch_meta_tensor = (
            torch.ops._C_dispatch_combine.wrap_cuda_ptr(
                ct, self._raw_dispatch_meta.value,
                self.max_recv, 4, 2))  # int32
        # Persistent buffers for fused prepare kernel.
        # Pre-allocated once; reused every MoE layer.
        # expert_topk_ids: (max_recv,) int64
        # expert_topk_weights: (max_recv,) float32
        # expert_num_tokens: set later via
        # init_prepare_buffers() when num_experts known.
        self.expert_topk_ids_buf = torch.zeros(
            self.max_recv, dtype=torch.int64,
            device=f'cuda:{self._device}')
        self.expert_topk_weights_buf = torch.zeros(
            self.max_recv, dtype=torch.float32,
            device=f'cuda:{self._device}')
        self.expert_num_tokens_buf = None
        self.data_remap_buf = None
        self._num_experts = None

        # Init barrier: sync all ranks after IPC setup.
        # Ensures cudaMemset zeroed offsets are visible
        # cross-GPU before first dispatch_p2p.
        self.gpu_p2p_barrier()

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
            'signals': to_bytes(
                cuda_rt.cudaIpcGetMemHandle(
                    self._raw_signals)),
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
        self.remote_signals_ptrs = []
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
                self.remote_signals_ptrs.append(
                    self._raw_signals.value)
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
                self.remote_signals_ptrs.append(
                    cuda_rt.cudaIpcOpenMemHandle(
                        from_bytes(
                            h['signals'])).value)

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
            assert self.remote_signals_ptrs[r], \
                f"NULL signals ptr for rank {r}"

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

        # self_signals (1 pointer)
        data += struct.pack(
            'Q', self.remote_signals_ptrs[self.rank])

        # peer_signals[kMaxRanks]
        for r in range(max_ranks):
            ptr = self.remote_signals_ptrs[r] \
                if r < self.world_size else 0
            data += struct.pack('Q', ptr)

        # Scalar fields
        data += struct.pack('i', self.rank)
        data += struct.pack('i', self.world_size)
        data += struct.pack('i', self._experts_per_rank)
        data += struct.pack('i', self.hidden_dim)
        data += struct.pack('i', self.max_num_tokens)
        data += struct.pack('i', self.max_recv)

        # ---- Integrated routing fields (EPLB) ----
        # remote_expert_counts[kMaxRanks]
        for r in range(max_ranks):
            ptr = (self.remote_expert_counts_ptrs[r]
                   if (self._integrated_routing_enabled
                       and r < self.world_size)
                   else 0)
            data += struct.pack('Q', ptr)

        # logical_to_physical_map (1 pointer)
        ptr = (self._routing_map_tensor.data_ptr()
               if self._integrated_routing_enabled
               else 0)
        data += struct.pack('Q', ptr)

        # logical_replica_count (1 pointer)
        ptr = (self._routing_count_tensor.data_ptr()
               if self._integrated_routing_enabled
               else 0)
        data += struct.pack('Q', ptr)

        # routing_selection (1 pointer)
        ptr = (self._raw_routing_selection.value
               if self._integrated_routing_enabled
               else 0)
        data += struct.pack('Q', ptr)

        # Scalars for integrated routing
        data += struct.pack(
            'i', self._num_logical_experts)
        data += struct.pack(
            'i', self._max_replicas)
        data += struct.pack(
            'i', self._physical_experts_per_rank)

        # 4 bytes padding for 8-byte alignment of
        # routing_ready_flag pointer
        data += struct.pack('i', 0)

        # routing_ready_flag (1 pointer)
        ptr = (self._raw_routing_ready_flag.value
               if self._integrated_routing_enabled
               else 0)
        data += struct.pack('Q', ptr)

        # phase_a_done_counter (1 pointer)
        ptr = (self._raw_phase_a_done_counter.value
               if self._integrated_routing_enabled
               else 0)
        data += struct.pack('Q', ptr)

        # local_expert_counts (1 pointer)
        ptr = (self._raw_local_expert_counts.value
               if self._integrated_routing_enabled
               else 0)
        data += struct.pack('Q', ptr)

        # ---- Per-sender section support ----
        section_size = (self.max_recv // self.world_size
                        if self.world_size > 0 else 0)
        data += struct.pack('i', section_size)  # dispatch
        data += struct.pack('i', section_size)  # combine
        data += struct.pack(
            'Q', self._raw_local_dispatch_counters.value)
        data += struct.pack(
            'Q', self._raw_local_combine_counters.value)

        # combine_done_counter (1 pointer)
        data += struct.pack(
            'Q', self._raw_combine_done_counter.value)

        # profiling_timestamps (1 pointer)
        ptr = (self._raw_profiling_timestamps.value
               if self._profiling_enabled
               else 0)
        data += struct.pack('Q', ptr)

        config_bytes = bytes(data)
        config_tensor = torch.frombuffer(
            bytearray(config_bytes), dtype=torch.uint8
        ).cuda()
        return config_tensor

    def update_experts_per_rank(self, experts_per_rank: int):
        """Update experts_per_rank in the config tensor."""
        import struct
        self._experts_per_rank = experts_per_rank
        max_ranks = 64
        # 6 ptr arrays + self_signals(1) + peer_signals(64)
        # + rank(4) + world_size(4) = offset to experts_per_rank
        offset = (6 * max_ranks * 8
                  + 8 + max_ranks * 8
                  + 2 * 4)
        packed = struct.pack('i', experts_per_rank)
        cpu_config = self.config_tensor.cpu()
        for i, b in enumerate(packed):
            cpu_config[offset + i] = b
        self.config_tensor.copy_(cpu_config)

    # ================================================================
    # GPU-side ops (CUDA-graph compatible)
    # ================================================================

    def gpu_p2p_barrier(self):
        """P2P flag-based barrier (GPU kernel)."""
        torch.ops._C_dispatch_combine.p2p_barrier(
            self.config_tensor)

    def gpu_scatter_add_direct(
            self, output: torch.Tensor, mc: int):
        """Scatter-add from IPC combine buffers directly
        into output using native bf16/fp16 atomicAdd.
        Output zeroed inline by kernel."""
        M = output.shape[0]
        torch.ops._C_dispatch_combine\
            .scatter_add_direct(
                output, self.config_tensor,
                mc, self.hidden_dim, M)

    def gpu_p2p_barrier_reset_dispatch(self):
        """Reset dispatch offset + P2P barrier."""
        torch.ops._C_dispatch_combine\
            .p2p_barrier_reset_dispatch(
                self.config_tensor)

    def gpu_combine_and_scatter(
            self,
            expert_output: torch.Tensor,
            dispatch_meta: torch.Tensor,
            output: torch.Tensor,
            mc: int):
        """Fused combine P2P + barrier + scatter-add.
        Replaces combine_p2p + barrier_reset_dispatch +
        scatter_add_direct in a single kernel launch."""
        M = output.shape[0]
        torch.ops._C_dispatch_combine\
            .combine_and_scatter(
                expert_output,
                dispatch_meta,
                self.compact_reverse_buf,
                output,
                self.config_tensor,
                mc, self.hidden_dim, M)

        if self._profiling_enabled:
            if self._read_and_accumulate_timestamps('cas'):
                self._maybe_print_profile()

    def gpu_combine_p2p(
            self,
            expert_output: torch.Tensor,
            dispatch_meta: torch.Tensor,
            mc: int):
        """Combine P2P writes (standalone kernel)."""
        torch.ops._C_dispatch_combine.combine_p2p(
            expert_output, dispatch_meta,
            self.compact_reverse_buf,
            self.config_tensor,
            mc, self.hidden_dim)

    # ================================================================
    # Split-phase dispatch+route ops (profiling mode)
    # ================================================================

    def gpu_dar_phase_a(
            self,
            input_tensor: torch.Tensor,
            topk_ids: torch.Tensor,
            topk_weights: torch.Tensor,
            M: int, K: int, topk: int):
        """Phase A: broadcast dispatch + expert count
        accumulation."""
        torch.ops._C_dispatch_combine\
            .dar_phase_a(
                input_tensor, topk_ids,
                topk_weights,
                self.config_tensor,
                M, K, topk,
                self._num_logical_experts,
                self.world_size)

    def gpu_dar_push_and_barrier(self):
        """Allgather push + P2P barrier
        (RESET_COMBINE)."""
        torch.ops._C_dispatch_combine\
            .dar_push_and_barrier(
                self.config_tensor)

    def gpu_dar_phase_c(self):
        """Phase C: deterministic router."""
        torch.ops._C_dispatch_combine\
            .dar_phase_c(
                self.config_tensor,
                self._num_logical_experts,
                self.world_size)

    def gpu_dar_phase_d1(self, num_experts: int):
        """Phase D1: zero expert_num_tokens for D2."""
        torch.ops._C_dispatch_combine\
            .dar_phase_d1(
                self.expert_num_tokens_buf,
                num_experts)

    def gpu_dar_phase_d2(
            self, mc: int, num_experts: int):
        """Phase D2: routing filter."""
        torch.ops._C_dispatch_combine\
            .dar_phase_d2(
                self.expert_topk_ids_buf,
                self.expert_topk_weights_buf,
                self.expert_num_tokens_buf,
                self.data_remap_buf,
                self.config_tensor,
                mc, num_experts)

    def gpu_dar_phase_e(self):
        """Phase E: zero counts for next invocation."""
        torch.ops._C_dispatch_combine\
            .dar_phase_e(self.config_tensor)

    def gpu_dar_compact(
            self, mc_compact: int,
            num_experts: int):
        """Compact valid entries from scattered sections
        into contiguous positions."""
        torch.ops._C_dispatch_combine\
            .dar_compact(
                self.expert_topk_ids_buf,
                self.expert_topk_weights_buf,
                self.data_remap_buf,
                self.compact_expert_topk_ids_buf,
                self.compact_expert_topk_weights_buf,
                self.compact_data_remap_buf,
                self.compact_reverse_buf,
                self.config_tensor,
                mc_compact, num_experts)

    def init_prepare_buffers(self, num_experts: int):
        """Allocate expert_num_tokens buffer once
        num_experts is known (set by PrepareAndFinalize
        constructor via update_experts_per_rank)."""
        if (self.expert_num_tokens_buf is not None
                and self._num_experts == num_experts):
            return
        self._num_experts = num_experts
        self.expert_num_tokens_buf = torch.zeros(
            num_experts, dtype=torch.int32,
            device=f'cuda:{self._device}')
        # data_remap: maps dispatch_recv entries to
        # group leaders for co-located expert dedup.
        # Identity by default (each entry maps to self).
        self.data_remap_buf = torch.arange(
            self.max_recv, dtype=torch.int32,
            device=f'cuda:{self._device}')
        # Compact buffers for section compaction.
        # After dispatch receive, valid entries are
        # scattered across per-sender sections.
        # Compaction gathers them into contiguous
        # positions so downstream element-wise kernels
        # operate on mc_compact instead of max_recv.
        dev = f'cuda:{self._device}'
        self.compact_expert_topk_ids_buf = (
            torch.full((self.max_recv,),
                       num_experts,
                       dtype=torch.int64,
                       device=dev))
        self.compact_expert_topk_weights_buf = (
            torch.zeros(self.max_recv,
                        dtype=torch.float32,
                        device=dev))
        self.compact_data_remap_buf = (
            torch.zeros(self.max_recv,
                        dtype=torch.int32,
                        device=dev))
        # Identity default: compact_reverse[i] = i.
        # dar_compact_kernel overwrites with actual
        # compact mapping when compaction runs.
        self.compact_reverse_buf = (
            torch.arange(self.max_recv,
                         dtype=torch.int32,
                         device=dev))

    def gpu_prepare_dispatch_recv(
            self, mc: int, num_experts: int):
        """Fused stamp/zero + routing metadata
        extraction. Replaces stamp_and_zero_dispatch +
        8 PyTorch ops in _receiver(). Returns
        (expert_topk_ids, expert_topk_weights,
         expert_num_tokens) sliced to mc."""
        if self.expert_num_tokens_buf is None:
            self.init_prepare_buffers(num_experts)
        torch.ops._C_dispatch_combine\
            .prepare_dispatch_recv(
                self.dispatch_recv_tensor,
                self.expert_topk_ids_buf,
                self.expert_topk_weights_buf,
                self.expert_num_tokens_buf,
                self.config_tensor,
                mc, self.hidden_dim,
                num_experts)
        return (
            self.expert_topk_ids_buf[:mc]
            .unsqueeze(1),
            self.expert_topk_weights_buf[:mc]
            .unsqueeze(1),
            self.expert_num_tokens_buf,
        )

    # ================================================================
    # Integrated routing (EPLB) support
    # ================================================================

    def init_integrated_routing(
        self,
        num_logical_experts: int,
        max_replicas: int,
        physical_experts_per_rank: int,
    ):
        """Allocate buffers for integrated routing.

        Must be called before gpu_dispatch_and_route().
        Exchanges IPC handles for expert_counts buffer
        so all ranks can push atomicAdds.
        """
        if self._integrated_routing_enabled:
            return

        self._num_logical_experts = num_logical_experts
        self._max_replicas = max_replicas
        self._physical_experts_per_rank = (
            physical_experts_per_rank)

        cuda_rt = self._cuda_rt
        dev = f'cuda:{self._device}'

        # Expert counts buffer (IPC-shared, allgather
        # pattern). Layout: ws sections of NL int32s.
        # Sender s writes to section [s*NL, (s+1)*NL).
        # Receiver sums all sections after barrier.
        ec_bytes = self.world_size * num_logical_experts * 4
        self._raw_expert_counts = (
            cuda_rt.cudaMalloc(ec_bytes))
        cuda_rt.cudaMemset(
            self._raw_expert_counts, 0, ec_bytes)

        # Routing selection buffer (local only).
        rs_bytes = num_logical_experts * 4
        self._raw_routing_selection = (
            cuda_rt.cudaMalloc(rs_bytes))
        cuda_rt.cudaMemset(
            self._raw_routing_selection, 0, rs_bytes)

        # Routing ready flag (local only, 4 bytes).
        self._raw_routing_ready_flag = (
            cuda_rt.cudaMalloc(4))
        cuda_rt.cudaMemset(
            self._raw_routing_ready_flag, 0, 4)

        # Phase A done counter (local only, 4 bytes).
        # Grid-wide sync: all blocks atomicAdd after
        # Phase A; block 0 spins until all done.
        self._raw_phase_a_done_counter = (
            cuda_rt.cudaMalloc(4))
        cuda_rt.cudaMemset(
            self._raw_phase_a_done_counter, 0, 4)

        # Local expert counts buffer (local only).
        # Batched all-reduce: blocks accumulate here,
        # block 0 pushes aggregate to all ranks.
        # Zeroed by Phase E; first invocation here.
        lec_bytes = num_logical_experts * 4
        self._raw_local_expert_counts = (
            cuda_rt.cudaMalloc(lec_bytes))
        cuda_rt.cudaMemset(
            self._raw_local_expert_counts,
            0, lec_bytes)

        # Routing tables (GPU tensors, updated on
        # EPLB rebalance).
        self._routing_map_tensor = torch.zeros(
            num_logical_experts * max_replicas,
            dtype=torch.int32, device=dev)
        self._routing_count_tensor = torch.zeros(
            num_logical_experts,
            dtype=torch.int64, device=dev)

        # Exchange IPC handles for expert_counts.
        to_bytes = self._handle_to_bytes
        local_ec_handle = to_bytes(
            cuda_rt.cudaIpcGetMemHandle(
                self._raw_expert_counts))
        all_ec_handles = [None] * self.world_size
        dist.all_gather_object(
            all_ec_handles, local_ec_handle,
            group=self.cpu_group)

        from_bytes = self._bytes_to_handle
        self.remote_expert_counts_ptrs = []
        for r in range(self.world_size):
            if r == self.rank:
                self.remote_expert_counts_ptrs.append(
                    self._raw_expert_counts.value)
            else:
                ptr = cuda_rt.cudaIpcOpenMemHandle(
                    from_bytes(all_ec_handles[r]))
                self.remote_expert_counts_ptrs.append(
                    ptr.value)

        # Wrap local expert_counts as a non-owning tensor
        # for cudaMemsetAsync in the host wrapper (avoids
        # host-side dereference of device config pointer,
        # required for CUDA graph compatibility).
        ct = self.config_tensor  # dummy for device
        self._expert_counts_tensor = (
            torch.ops._C_dispatch_combine.wrap_cuda_ptr(
                ct, self._raw_expert_counts.value,
                self.world_size * num_logical_experts,
                1, 2))  # int32

        # Enable flag BEFORE rebuild so _build_config_tensor
        # packs the routing pointers (not NULL).
        self._integrated_routing_enabled = True

        # Rebuild config tensor with new fields.
        self.config_tensor = (
            self._build_config_tensor())

        logger.info(
            "Integrated routing initialized: "
            "NL=%d, max_replicas=%d, epr=%d",
            num_logical_experts, max_replicas,
            physical_experts_per_rank)

    def update_routing_tables(
        self,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ):
        """Update GPU-resident routing tables.

        Called when EPLB rebalances expert placement.
        """
        assert self._integrated_routing_enabled
        # logical_to_physical_map: (NL, max_replicas)
        self._routing_map_tensor.copy_(
            logical_to_physical_map.to(
                torch.int32).reshape(-1))
        self._routing_count_tensor.copy_(
            logical_replica_count.to(torch.int64))

    # ================================================================
    # Fine-grained profiling
    # ================================================================

    def _read_and_accumulate_timestamps(
            self, kernel_name: str) -> bool:
        """Read GPU timestamp buffer, accumulate deltas.

        Returns True if timestamps were accumulated,
        False if skipped (disabled or graph capture)."""
        if not self._profiling_enabled:
            return False
        # Skip during CUDA graph capture: synchronize
        # and cudaMemcpy are illegal in capture mode.
        if torch.cuda.is_current_stream_capturing():
            return False
        # Synchronize to ensure kernel has completed
        # and timestamps are written.
        torch.cuda.current_stream().synchronize()

        # Copy int64 timestamps from GPU to host.
        nbytes = _TOTAL_PROFILE_SLOTS * 8
        ts_host = (ctypes.c_int64
                    * _TOTAL_PROFILE_SLOTS)()
        self._cuda_rt.cudaMemcpy(
            ctypes.cast(ts_host, ctypes.c_void_p),
            self._raw_profiling_timestamps,
            nbytes)
        ts = [ts_host[i] for i in
              range(_TOTAL_PROFILE_SLOTS)]

        # globaltimer is in nanoseconds on NVIDIA GPUs.
        if kernel_name == 'dar':
            for i in range(_DAR_NUM_STEPS - 1):
                if ts[i] > 0 and ts[i + 1] > 0:
                    delta_ns = ts[i + 1] - ts[i]
                    self._dar_accum[i] += delta_ns
        elif kernel_name == 'cas':
            base = _DAR_NUM_STEPS
            for i in range(_CAS_NUM_STEPS - 1):
                if (ts[base + i] > 0
                        and ts[base + i + 1] > 0):
                    delta_ns = (ts[base + i + 1]
                                - ts[base + i])
                    self._cas_accum[i] += delta_ns
        return True

    def _maybe_print_profile(self):
        """Print and reset averages if interval reached."""
        if not self._profiling_enabled:
            return
        self._profile_batch_count += 1
        if (self._profile_batch_count
                % self._profiling_interval != 0):
            return

        n = self._profiling_interval
        # Print dispatch_and_route steps.
        parts = []
        for i in range(_DAR_NUM_STEPS - 1):
            avg_us = (self._dar_accum[i] / n
                      / 1000.0)
            name = _DAR_STEP_NAMES[i]
            parts.append(f"  {name}: {avg_us:.1f} us")
        total_dar = sum(self._dar_accum) / n / 1000.0
        logger.info(
            "DC profile [rank %d] dispatch_and_route "
            "(avg %d batches, total %.1f us):\n%s",
            self.rank, n, total_dar,
            "\n".join(parts))

        # Print combine_and_scatter steps.
        parts = []
        for i in range(_CAS_NUM_STEPS - 1):
            avg_us = (self._cas_accum[i] / n
                      / 1000.0)
            name = _CAS_STEP_NAMES[i]
            parts.append(f"  {name}: {avg_us:.1f} us")
        total_cas = sum(self._cas_accum) / n / 1000.0
        logger.info(
            "DC profile [rank %d] combine_and_scatter "
            "(avg %d batches, total %.1f us):\n%s",
            self.rank, n, total_cas,
            "\n".join(parts))

        # Reset accumulators.
        self._dar_accum = [0.0] * _DAR_NUM_STEPS
        self._cas_accum = [0.0] * _CAS_NUM_STEPS

    def gpu_dispatch_and_route(
        self,
        input_tensor: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        mc: int,
        M: int,
        K: int,
        topk: int,
        num_experts: int,
    ):
        """Launch the fused dispatch+route+filter kernel.

        Returns (expert_topk_ids, expert_topk_weights,
                 expert_num_tokens) sliced to mc.
        """
        assert self._integrated_routing_enabled
        if self.expert_num_tokens_buf is None:
            self.init_prepare_buffers(num_experts)

        torch.ops._C_dispatch_combine\
            .dispatch_and_route(
                input_tensor,
                topk_ids,
                topk_weights,
                self.dispatch_recv_tensor,
                self.expert_topk_ids_buf,
                self.expert_topk_weights_buf,
                self.expert_num_tokens_buf,
                self._expert_counts_tensor,
                self.data_remap_buf,
                self.config_tensor,
                M, K, topk, mc,
                num_experts,
                self._num_logical_experts,
                self.world_size)

        if self._profiling_enabled:
            self._read_and_accumulate_timestamps('dar')

        return (
            self.expert_topk_ids_buf[:mc]
            .unsqueeze(1),
            self.expert_topk_weights_buf[:mc]
            .unsqueeze(1),
            self.expert_num_tokens_buf,
            self.data_remap_buf[:mc],
        )

    def destroy(self):
        """Release cudaMalloc'd buffers."""
        self._cuda_rt.cudaFree(self._raw_dispatch_recv)
        self._cuda_rt.cudaFree(self._raw_dispatch_meta)
        self._cuda_rt.cudaFree(self._raw_dispatch_offset)
        self._cuda_rt.cudaFree(self._raw_combine_recv)
        self._cuda_rt.cudaFree(self._raw_combine_meta)
        self._cuda_rt.cudaFree(self._raw_combine_offset)
        self._cuda_rt.cudaFree(
            self._raw_local_dispatch_counters)
        self._cuda_rt.cudaFree(
            self._raw_local_combine_counters)
        self._cuda_rt.cudaFree(
            self._raw_combine_done_counter)
        self._cuda_rt.cudaFree(self._raw_signals)
        if self._raw_expert_counts is not None:
            self._cuda_rt.cudaFree(
                self._raw_expert_counts)
        if self._raw_routing_selection is not None:
            self._cuda_rt.cudaFree(
                self._raw_routing_selection)
        if self._raw_routing_ready_flag is not None:
            self._cuda_rt.cudaFree(
                self._raw_routing_ready_flag)
        if self._raw_phase_a_done_counter is not None:
            self._cuda_rt.cudaFree(
                self._raw_phase_a_done_counter)
        if self._raw_local_expert_counts is not None:
            self._cuda_rt.cudaFree(
                self._raw_local_expert_counts)
        if self._raw_profiling_timestamps is not None:
            self._cuda_rt.cudaFree(
                self._raw_profiling_timestamps)
