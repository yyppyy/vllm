# moe_topk_localdump.py
import os
from typing import Dict, List, Optional
import torch

# Env var for file prefix, e.g. "/tmp/myrun" -> "/tmp/myrun.ep1.pt"
_PREFIX_ENV = "TOPK_DUMP_PREFIX"

# File prefix (can be overridden via set_file_prefix)
_prefix: str = os.getenv(_PREFIX_ENV, "moe_topk_dump")

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

def record_topk_for_batch(ep_rank: int, topk_tensor: torch.Tensor) -> str:
    """
    Append this rank's top-k tensor for the *next* batch index and
    immediately dump the whole list to disk atomically.

    Returns the output file path for convenience.
    """
    # Make sure tensor is CPU and detached so files are portable/stable.
    t = topk_tensor.detach().cpu().clone()

    buf = _buffers.setdefault(ep_rank, [])
    buf.append(t)

    out_path = _out_path_for_rank(ep_rank)
    _atomic_save(buf, out_path)
    return out_path
