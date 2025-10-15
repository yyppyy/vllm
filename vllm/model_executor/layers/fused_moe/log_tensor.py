import os
import atexit
from typing import List, Optional
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# Env var for file prefix, e.g. "/tmp/myrun" -> "/tmp/myrun.ep3.pt"
_PREFIX_ENV = "TOPK_DUMP_PREFIX"

_prefix: Optional[str] = os.getenv(_PREFIX_ENV) or None
_ep_rank: Optional[int] = None
_buffer: List[torch.Tensor] = []

def set_file_prefix(prefix: Optional[str]) -> None:
    """Override prefix at runtime (or set None to disable logging)."""
    global _prefix
    _prefix = prefix

def _out_path(suffix: str = ".pt") -> str:
    assert _prefix is not None, "No prefix set"
    assert _ep_rank is not None, "EP rank not set yet"
    return f"{_prefix}.ep{_ep_rank}{suffix}"

def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

def _atomic_save(obj, out_path: str) -> None:
    # tmp_path = out_path + ".tmp"
    _ensure_parent_dir(out_path)
    torch.save(obj, out_path)
    # os.replace(tmp_path, out_path)

def record_topk_for_batch(ep_rank: int, topk_tensor: torch.Tensor) -> Optional[str]:
    """
    Append this rank's top-k tensor for the next batch.
    Nothing is written immediately; data is flushed once at exit (or via flush()).
    Returns the eventual output path (or None if logging disabled).
    """
    global _ep_rank
    if _prefix is None:
        return None

    if _ep_rank is None:
        _ep_rank = ep_rank
    elif _ep_rank != ep_rank:
        raise RuntimeError(f"record_topk_for_batch called with ep_rank={ep_rank}, "
                           f"but logger already initialized for ep_rank={_ep_rank}")

    t = topk_tensor.detach().cpu().clone()
    _buffer.append(t)
    return _out_path()

def flush() -> Optional[str]:
    """Write the buffered list to disk atomically. Returns path or None if no-op."""
    if _prefix is None or _ep_rank is None or not _buffer:
        logger.warning("not logging topk tensor on this rank")
        return None
    out_path = _out_path()
    _atomic_save(_buffer, out_path)
    return out_path

# Always flush at process exit so you don't lose the tail.
atexit.register(flush)