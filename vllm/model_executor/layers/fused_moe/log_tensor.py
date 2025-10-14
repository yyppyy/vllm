# moe_topk_localdump.py
import os
from typing import Dict, List, Optional
import torch

# Env var for file prefix, e.g. "/tmp/myrun" -> "/tmp/myrun.ep1.pt"
_PREFIX_ENV = "TOPK_DUMP_PREFIX"
_THROTTLE_ENV = "TOPK_DUMP_THROTTLE"

# File prefix (can be overridden via set_file_prefix)
_prefix: Optional[str] = os.getenv(_PREFIX_ENV) or None
_throttle: int = int(os.getenv(_THROTTLE_ENV, "10"))

# Keep a separate buffer per ep_rank so a single process could (safely) log multiple ranks if needed.
_buffers: Dict[int, List[torch.Tensor]] = {}

def _out_path_for_rank(ep_rank: int, suffix: str = ".pt") -> str:
    return f"{_prefix}.ep{ep_rank}{suffix}"

def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

def _atomic_save(obj, out_path: str) -> None:
    tmp_path = out_path + ".tmp"
    _ensure_parent_dir(out_path)
    torch.save(obj, tmp_path)
    os.replace(tmp_path, out_path)

def record_topk_for_batch(ep_rank: int, topk_tensor: torch.Tensor) -> Optional[str]:
    """
    Append this rank's top-k tensor for the *next* batch index and
    immediately dump the whole list to disk atomically.

    Returns the output file path for convenience.
    """
    if _prefix is None:
        return None
    
    # Make sure tensor is CPU and detached so files are portable/stable.
    t = topk_tensor.detach().cpu().clone()

    buf = _buffers.setdefault(ep_rank, [])
    buf.append(t)

    out_path = _out_path_for_rank(ep_rank)
    if len(buf) % _throttle == 0:
        _atomic_save(buf, out_path)
    return out_path
