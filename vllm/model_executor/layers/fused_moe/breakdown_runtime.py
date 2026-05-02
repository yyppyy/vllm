# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime singleton for the per-(rank, layer, batch) MoE *breakdown*
profiler.

Owns the GPU armed flag, GPU sequence counter, GPU `py_stamps` tensors
(one per layer), and the pinned-host ringbuffer that the
`log_breakdown` C++ op writes into.

Mirrors `explat_runtime.py` exactly in structure, but uses the
`VLLM_BREAKDOWN_*` env-var family so the two profilers can run side
by side without colliding.

The poller daemon thread:
  Phase 1: waits for VLLM_BREAKDOWN_READY_FILE to appear; flips
           the GPU `armed` flag from 0 -> 1.
  Phase 2: polls the GPU counter every 1s; once the counter has
           been stable for 30s the thread drains all slots into
           VLLM_BREAKDOWN_LOG_PATH and exits.

When VLLM_BREAKDOWN_PROFILE is unset / 0, every entrypoint here is a
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
# Slot layout (must match csrc/explat_logger.cu log_breakdown_kernel)
#
#   int64 seq                            [0]
#   int64 (rank low32 | layer high32)    [1]
#   int64 M                              [2]
#   int64 attention_ns                   [3]
#   int64 gating_ns                      [4]
#   int64 routing_ns                     [5]
#   int64 dispatch_ns                    [6]
#   int64 expert_ns                      [7]
#   int64 combine_ns                     [8]
# ---------------------------------------------------------------------

SLOT_STRIDE_INT64 = 9
PY_STAMPS_LEN = 5  # attn_start, attn_end, gate_start, gate_end, dispatch_end

# Pre-allocated rows so the attention block, MoE block, and
# modular_kernel can all write to the same int64[5] view without
# passing tensors through the module hierarchy. Bumped if a model
# has more than this many layers.
_PY_STAMPS_MAX_LAYERS = 128

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
_N_SLOTS: int = _N_SLOTS_DEFAULT
_LOG_PATH: str = './server_breakdown.log'

_armed: Optional[torch.Tensor] = None
_counter: Optional[torch.Tensor] = None
_ringbuf: Optional[torch.Tensor] = None  # pinned host int64 tensor
_py_stamps_table: Optional[torch.Tensor] = None  # int64[max_layers, 5]
_poller_started: bool = False
_init_lock = threading.Lock()


def is_enabled() -> bool:
    """Return True if VLLM_BREAKDOWN_PROFILE is set to a nonzero
    integer. Cached after the first call."""
    global _ENABLED, _N_SLOTS, _LOG_PATH
    if _ENABLED is None:
        _ENABLED = _read_int_env('VLLM_BREAKDOWN_PROFILE', 0) != 0
        _N_SLOTS = _read_int_env('VLLM_BREAKDOWN_N_SLOTS',
                                  _N_SLOTS_DEFAULT)
        _LOG_PATH = os.environ.get('VLLM_BREAKDOWN_LOG_PATH',
                                    _LOG_PATH)
    return _ENABLED


def _ensure_py_stamps_table_locked() -> None:
    """Allocate the int64[max_layers, 5] py-stamps table. Caller must
    hold `_init_lock`."""
    global _py_stamps_table
    if _py_stamps_table is None:
        _py_stamps_table = torch.zeros(
            (_PY_STAMPS_MAX_LAYERS, PY_STAMPS_LEN),
            dtype=torch.int64, device='cuda')


def _ensure_buffers_locked() -> None:
    """Allocate the singleton armed/counter/ringbuf tensors. Caller
    must hold _init_lock.

    `device='cpu'` is required on the pinned ringbuffer because vLLM
    pushes a CUDA device guard during model construction; without an
    explicit device here `torch.zeros(..., pin_memory=True)` would try
    to allocate a CUDA tensor and fail with "Only dense CPU tensors
    can be pinned"."""
    global _armed, _counter, _ringbuf
    if _armed is None:
        _armed = torch.zeros(1, dtype=torch.int32, device='cuda')
    if _counter is None:
        _counter = torch.zeros(1, dtype=torch.int32, device='cuda')
    if _ringbuf is None:
        _ringbuf = torch.zeros((_N_SLOTS, SLOT_STRIDE_INT64),
                                dtype=torch.int64,
                                device='cpu',
                                pin_memory=True)


def init() -> None:
    """One-shot, idempotent allocation of every GPU/pinned tensor the
    breakdown profiler needs (py_stamps table + armed/counter/ringbuf).
    Must be called from non-compiled code (the lock here cannot be
    traced by Dynamo). After this returns, every getter below is
    lock-free and Dynamo-safe."""
    if not is_enabled():
        return
    with _init_lock:
        _ensure_py_stamps_table_locked()
        _ensure_buffers_locked()


def get_py_stamps_row(layer_idx: int) -> torch.Tensor:
    """Return the int64[5] view of the global py-stamps table for
    `layer_idx`. Lock-free; assumes `init()` was called from a
    non-compiled context (e.g. the model layer's `__init__`).

    The same row is hit by:
      * the attention block (`record_stamp(row, 0)` and
        `record_stamp(row, 1)`)
      * the MoE block (`record_stamp(row, 2)` for gate_start)
      * `modular_kernel.forward` (`record_stamp(row, 3)` for gate_end
        and `record_stamp(row, 4)` for dispatch_end, then
        `log_breakdown(... row ...)`).
    Stream ordering ensures `log_breakdown` reads the layer's writes
    before the next layer overwrites them.
    """
    if _py_stamps_table is None:
        raise RuntimeError(
            "breakdown profiler: get_py_stamps_row() called before "
            "init(). Call breakdown_runtime.init() once from "
            "non-compiled code before the first compiled forward.")
    if layer_idx >= _py_stamps_table.size(0):
        raise RuntimeError(
            f"breakdown profiler: layer_idx={layer_idx} exceeds "
            f"pre-allocated row count {_PY_STAMPS_MAX_LAYERS}; "
            f"bump _PY_STAMPS_MAX_LAYERS in breakdown_runtime.py")
    return _py_stamps_table[layer_idx]


def get_armed_tensor() -> torch.Tensor:
    if _armed is None:
        raise RuntimeError(
            "breakdown profiler: get_armed_tensor() called before "
            "init().")
    return _armed


def get_counter_tensor() -> torch.Tensor:
    if _counter is None:
        raise RuntimeError(
            "breakdown profiler: get_counter_tensor() called before "
            "init().")
    return _counter


def get_ringbuf_tensor() -> torch.Tensor:
    if _ringbuf is None:
        raise RuntimeError(
            "breakdown profiler: get_ringbuf_tensor() called before "
            "init().")
    return _ringbuf


def _format_slot(buf_row: torch.Tensor) -> Optional[str]:
    """Format one slot from the pinned int64 row into a Breakdown
    line. Returns None if the slot looks invalid (uninitialized)."""
    seq = int(buf_row[0].item())
    rl = int(buf_row[1].item())
    rank = rl & 0xFFFFFFFF
    if rank >= 0x80000000:
        rank -= 1 << 32
    layer = (rl >> 32) & 0xFFFFFFFF
    if layer >= 0x80000000:
        layer -= 1 << 32
    M = int(buf_row[2].item()) & 0xFFFFFFFF
    attention_ns = int(buf_row[3].item())
    gating_ns = int(buf_row[4].item())
    routing_ns = int(buf_row[5].item())
    dispatch_ns = int(buf_row[6].item())
    expert_ns = int(buf_row[7].item())
    combine_ns = int(buf_row[8].item())
    return (
        f"Breakdown seq={seq} rank={rank} layer={layer} M={M} "
        f"attention_ns={attention_ns} gating_ns={gating_ns} "
        f"routing_ns={routing_ns} dispatch_ns={dispatch_ns} "
        f"expert_ns={expert_ns} combine_ns={combine_ns}"
    )


def _drain_and_write(last_seen: int) -> int:
    """Walk slots 0..last_seen of the pinned ringbuffer and write one
    line per slot to the log file. Returns the number of lines
    written. If last_seen > _N_SLOTS, only the most recent _N_SLOTS
    slots are dumped (in seq order) and a warning is logged."""
    if _ringbuf is None or last_seen <= 0:
        return 0
    if last_seen > _N_SLOTS:
        logger.warning(
            "breakdown ringbuffer wrapped (counter=%d, n_slots=%d); "
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
    """Two-phase poller. See module docstring."""
    ready_file = os.environ.get('VLLM_BREAKDOWN_READY_FILE', '')
    while True:
        if not ready_file or os.path.exists(ready_file):
            try:
                if _armed is not None:
                    _armed.fill_(1)
                logger.info(
                    "breakdown poller: armed (ready_file=%s)",
                    ready_file or '<none>')
            except Exception as e:
                logger.warning(
                    "breakdown poller: failed to arm (%s)", e)
            break
        time.sleep(_PHASE1_POLL_S)

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
                "breakdown poller: counter read failed (%s)", e)
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
                    "breakdown poller: drained %d slots to %s "
                    "(counter=%d)",
                    n, _LOG_PATH, last_seen)
            except Exception as e:
                logger.warning(
                    "breakdown poller: drain failed (%s)", e)
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
                              name='breakdown-poller',
                              daemon=True)
        t.start()
        _poller_started = True
        logger.info(
            "breakdown poller started (n_slots=%d, log_path=%s)",
            _N_SLOTS, _LOG_PATH)
