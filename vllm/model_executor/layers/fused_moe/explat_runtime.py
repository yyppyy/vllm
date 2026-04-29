# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime singleton for the per-(rank, layer, batch) MoE profiler.

Owns the GPU stamps, GPU armed flag, GPU sequence counter, and the
pinned-host ringbuffer that the explat C++ ops write into. Spawns a
daemon thread that:

  Phase 1: waits for VLLM_EXP_LATENCY_READY_FILE to appear, then writes
           1 into the GPU armed tensor. From the next replay onward the
           logger kernel writes one slot per (rank, layer, batch).

  Phase 2: polls the GPU sequence counter once a second; once the
           counter has been stable for 30 s the thread drains all slots
           into VLLM_EXP_LATENCY_LOG_PATH and exits. No periodic writes,
           no atexit hook.

When VLLM_EXP_LATENCY_PROFILE is unset / 0, every entrypoint here is a
no-op and no GPU memory is allocated.
"""
import os
import threading
import time
from typing import Optional

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# ---------------------------------------------------------------------
# Slot layout (must match csrc/explat_logger.cu)
#
#   int64 seq                              [0]
#   int64 (rank low32 | layer high32)      [1]
#   int64 (M low32 | num_local_experts)    [2]
#   int64 align_ns                         [3]
#   int64 gemm_gu_ns                       [4]
#   int64 silu_ns                          [5]
#   int64 quant_ns                         [6]
#   int64 gemm_dn_ns                       [7]
#   int64 expert_tokens packed (E_MAX/2)   [8 ...]
#
# E_MAX must be even.
# ---------------------------------------------------------------------

_E_MAX_DEFAULT = 64
_N_SLOTS_DEFAULT = 65536
_IDLE_THRESHOLD_S = 30.0
_PHASE1_POLL_S = 0.05
_PHASE2_POLL_S = 1.0


def _read_int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


_ENABLED: Optional[bool] = None
_E_MAX: int = _E_MAX_DEFAULT
_N_SLOTS: int = _N_SLOTS_DEFAULT
_LOG_PATH: str = './server_explat.log'

_armed: Optional[torch.Tensor] = None
_counter: Optional[torch.Tensor] = None
_ringbuf: Optional[torch.Tensor] = None  # pinned host int64 tensor
_poller_started: bool = False
_init_lock = threading.Lock()


def is_enabled() -> bool:
    """Return True if VLLM_EXP_LATENCY_PROFILE is set to a nonzero
    integer. Cached after the first call."""
    global _ENABLED, _E_MAX, _N_SLOTS, _LOG_PATH
    if _ENABLED is None:
        _ENABLED = _read_int_env('VLLM_EXP_LATENCY_PROFILE', 0) != 0
        _E_MAX = _read_int_env('VLLM_EXP_LATENCY_E_MAX',
                                _E_MAX_DEFAULT)
        if _E_MAX % 2 != 0:
            _E_MAX += 1
        _N_SLOTS = _read_int_env('VLLM_EXP_LATENCY_N_SLOTS',
                                  _N_SLOTS_DEFAULT)
        _LOG_PATH = os.environ.get('VLLM_EXP_LATENCY_LOG_PATH',
                                    _LOG_PATH)
    return _ENABLED


def get_e_max() -> int:
    is_enabled()
    return _E_MAX


def _slot_stride_int64() -> int:
    return 8 + _E_MAX // 2


def alloc_layer_stamps() -> torch.Tensor:
    """Allocate a fresh int64[6] CUDA tensor for one MoE layer."""
    return torch.zeros(6, dtype=torch.int64, device='cuda')


def _ensure_buffers_locked() -> None:
    """Allocate the singleton armed/counter/ringbuf tensors. Caller
    must hold _init_lock."""
    global _armed, _counter, _ringbuf
    if _armed is None:
        _armed = torch.zeros(1, dtype=torch.int32, device='cuda')
    if _counter is None:
        _counter = torch.zeros(1, dtype=torch.int32, device='cuda')
    if _ringbuf is None:
        _ringbuf = torch.zeros((_N_SLOTS, _slot_stride_int64()),
                                dtype=torch.int64,
                                pin_memory=True)


def get_armed_tensor() -> torch.Tensor:
    with _init_lock:
        _ensure_buffers_locked()
    return _armed  # type: ignore[return-value]


def get_counter_tensor() -> torch.Tensor:
    with _init_lock:
        _ensure_buffers_locked()
    return _counter  # type: ignore[return-value]


def get_ringbuf_tensor() -> torch.Tensor:
    with _init_lock:
        _ensure_buffers_locked()
    return _ringbuf  # type: ignore[return-value]


def _format_slot(buf_row: torch.Tensor) -> Optional[str]:
    """Format one slot from the pinned int64 row into an ExpLat line.
    Returns None if the slot looks invalid (uninitialized)."""
    seq = int(buf_row[0].item())
    rl = int(buf_row[1].item())
    mn = int(buf_row[2].item())
    rank = rl & 0xFFFFFFFF
    if rank >= 0x80000000:
        rank -= 1 << 32
    layer = (rl >> 32) & 0xFFFFFFFF
    if layer >= 0x80000000:
        layer -= 1 << 32
    M = mn & 0xFFFFFFFF
    num_local_experts = (mn >> 32) & 0xFFFFFFFF
    align_ns = int(buf_row[3].item())
    gemm_gu_ns = int(buf_row[4].item())
    silu_ns = int(buf_row[5].item())
    quant_ns = int(buf_row[6].item())
    gemm_dn_ns = int(buf_row[7].item())
    pet: list[int] = []
    pairs = num_local_experts // 2 + (num_local_experts % 2)
    for i in range(pairs):
        packed = int(buf_row[8 + i].item()) & 0xFFFFFFFFFFFFFFFF
        v0 = packed & 0xFFFFFFFF
        v1 = (packed >> 32) & 0xFFFFFFFF
        pet.append(v0)
        if len(pet) < num_local_experts:
            pet.append(v1)
    return (
        f"ExpLat seq={seq} rank={rank} layer={layer} M={M} "
        f"num_local_experts={num_local_experts} "
        f"align_ns={align_ns} gemm_gu_ns={gemm_gu_ns} "
        f"silu_ns={silu_ns} quant_ns={quant_ns} "
        f"gemm_dn_ns={gemm_dn_ns} "
        f"per_expert_tokens={pet}"
    )


def _drain_and_write(last_seen: int) -> int:
    """Walk slots 0..last_seen of the pinned ringbuffer and write one
    line per slot to the log file. Returns the number of lines written.
    If last_seen > _N_SLOTS, only the most recent _N_SLOTS slots are
    dumped (in seq order) and a warning is logged."""
    if _ringbuf is None or last_seen <= 0:
        return 0
    if last_seen > _N_SLOTS:
        logger.warning(
            "explat ringbuffer wrapped (counter=%d, n_slots=%d); "
            "dumping the most recent %d slots only",
            last_seen, _N_SLOTS, _N_SLOTS)
        first = last_seen - _N_SLOTS
        seqs = range(first, last_seen)
    else:
        seqs = range(0, last_seen)
    lines: list[str] = []
    for seq in seqs:
        slot_idx = seq % _N_SLOTS
        line = _format_slot(_ringbuf[slot_idx])
        if line is not None:
            lines.append(line)
    if not lines:
        return 0
    os.makedirs(os.path.dirname(os.path.abspath(_LOG_PATH)) or '.',
                exist_ok=True)
    with open(_LOG_PATH, 'a') as f:
        f.write("\n".join(lines))
        f.write("\n")
    return len(lines)


def _poller_thread() -> None:
    """Run the two-phase poller. See module docstring."""
    ready_file = os.environ.get('VLLM_EXP_LATENCY_READY_FILE', '')
    # Phase 1: wait for the ready file. If unset, arm immediately.
    while True:
        if not ready_file or os.path.exists(ready_file):
            try:
                # One-time GPU scatter, outside any graph.
                if _armed is not None:
                    _armed.fill_(1)
                logger.info(
                    "explat poller: armed (ready_file=%s)",
                    ready_file or '<none>')
            except Exception as e:
                logger.warning(
                    "explat poller: failed to arm (%s)", e)
            break
        time.sleep(_PHASE1_POLL_S)

    # Phase 2: idle-watch the counter.
    last_seen = 0
    last_change_time = time.monotonic()
    while True:
        time.sleep(_PHASE2_POLL_S)
        if _counter is None:
            continue
        try:
            cur = int(_counter.cpu().item())
        except Exception as e:
            logger.warning(
                "explat poller: counter read failed (%s)", e)
            continue
        if cur > last_seen:
            last_seen = cur
            last_change_time = time.monotonic()
            continue
        idle = time.monotonic() - last_change_time
        if last_seen > 0 and idle >= _IDLE_THRESHOLD_S:
            try:
                n = _drain_and_write(last_seen)
                logger.info(
                    "explat poller: drained %d slots to %s "
                    "(counter=%d)",
                    n, _LOG_PATH, last_seen)
            except Exception as e:
                logger.warning(
                    "explat poller: drain failed (%s)", e)
            return


def ensure_poller_started() -> None:
    """Idempotent. Spawns the daemon thread on the first call."""
    global _poller_started
    if _poller_started:
        return
    with _init_lock:
        if _poller_started:
            return
        _ensure_buffers_locked()
        t = threading.Thread(target=_poller_thread,
                              name='explat-poller',
                              daemon=True)
        t.start()
        _poller_started = True
        logger.info(
            "explat poller started (n_slots=%d, e_max=%d, "
            "log_path=%s)", _N_SLOTS, _E_MAX, _LOG_PATH)
