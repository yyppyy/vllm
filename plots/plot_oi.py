#!/usr/bin/env python3
"""
Arithmetic intensity of a DeepSeek-V3-style MoE layer (no attention, no dense layers).

Symbols (matching your analysis):
  B: batch size
  S: tokens per sequence this step (S=1 for decode; S=T for prefill)
  D: model hidden size
  MD: expert FFN inner dim (per-expert up/down: [D x MD] and [MD x D])
  MR: routed experts per token (top-k), e.g., 8 for DeepSeek-V3
  MS: shared experts per layer (if any; 0 if none)
  MA: number of expert invocations per token for routing accounting (≈ MR; default=MR)
  elem_bytes: bytes per weight/activation element (fp16/bf16=2, fp32=4, etc.)
  E: (optional) total experts; not needed for FLOPs, only if you want to clamp counts

FLOPs used:
  moe_router_flops = B * S * D * MR * 2
  moe_per_token_flops = 2 * D * MD * 2   # two GEMMs per expert (up & down), each 2*multiplies
  moe_shared_expert_flops = MS * B * S * moe_per_token_flops
  moe_avg_tok_per_routed_expert = max(B * S * MA / MR, 1)
  moe_avg_routed_expert_flops = MR * moe_avg_tok_per_routed_expert * moe_per_token_flops
  moe_flops = moe_router_flops + moe_shared_expert_flops + moe_avg_routed_expert_flops

Bytes model (choose):
  weights-only:
    router_w_bytes   = D * MR * elem_bytes
    expert_w_bytes   = (MS + MR) * (2 * D * MD) * elem_bytes   # up & down per expert
    total_bytes      = router_w_bytes + expert_w_bytes
  weights+activations:
    + read input X:        B * S * D * elem_bytes
    + write output Y:      B * S * D * elem_bytes
"""

from dataclasses import dataclass
import argparse
from typing import Literal
from pathlib import Path
from collections import defaultdict
from pathlib import Path
import numpy as np
from utils import *

import matplotlib.pyplot as plt

BytesModel = Literal["weights-only", "weights+activations"]

@dataclass
class MoEConfig:
    D: int                  # hidden size
    MD: int                 # expert FFN inner dim
    MR: int                 # routed experts per token (top-k)
    MS: int = 0             # shared experts per layer
    MA: int = None          # routing multiplicity per token (~= MR). If None, set to MR.
    elem_bytes: int = 2     # 2 for fp16/bf16, 4 for fp32
    E: int | None = None    # total experts (optional; not required)

def moe_flops(B: int, S: int, cfg: MoEConfig, imbalance_factor: int = 3) -> float:
    MR = cfg.MR
    MS = cfg.MS
    MA = cfg.MA if cfg.MA is not None else cfg.MR
    D, MD = cfg.D, cfg.MD

    # Router: one linear D -> MR per token
    moe_router_flops = B * S * D * MR * 2

    # Per-expert FFN (two GEMMs per token routed to that expert)
    moe_per_token_flops = 2 * D * MD * 2  # (D*MD + MD*D) * 2

    # Shared experts: every token passes through MS experts
    moe_shared_expert_flops = MS * B * S * moe_per_token_flops

    # Routed experts: distribute B*S*MA expert-invocations across MR experts
    moe_avg_tok_per_routed_expert = max(B * S * MA / MR * imbalance_factor, 1)
    moe_avg_routed_expert_flops = MR * moe_avg_tok_per_routed_expert * moe_per_token_flops
    
    return float(moe_router_flops + moe_shared_expert_flops + moe_avg_routed_expert_flops)

def moe_bytes(B: int, S: int, cfg: MoEConfig, model: BytesModel) -> float:
    """
    Lower-bound data-movement cost per layer invocation:
    - weights-only: router + experts weights (dominant at small B,S)
    - weights+activations: + input and output activations (one read + one write)
    Assumes each *distinct used* expert's weights are streamed once.
    """
    D, MD, MR, MS, eb = cfg.D, cfg.MD, cfg.MR, cfg.MS, cfg.elem_bytes

    # Router weights (D x MR)
    router_w_bytes = D * MR * eb

    # Expert weights: two matrices per expert (up & down): (D*MD + MD*D) = 2*D*MD
    expert_w_bytes = (MS + MR) * (2 * D * MD) * eb

    total_bytes = router_w_bytes + expert_w_bytes

    if model == "weights+activations":
        activ_rd_wr = (B * S * D * eb) + (B * S * D * eb)  # read X, write Y
        total_bytes += activ_rd_wr

    return float(total_bytes)

def arithmetic_intensity(B: int, S: int, cfg: MoEConfig, bytes_model: BytesModel) -> tuple[float, float, float]:
    flops = moe_flops(B, S, cfg)
    bytes_ = moe_bytes(B, S, cfg, bytes_model)
    return flops, bytes_, (flops / max(bytes_, 1e-9))

def main():
    p = argparse.ArgumentParser(description="Arithmetic intensity of a DeepSeek-V3-style MoE layer.")
    # p.add_argument("--batch", "-B", type=int, required=True, help="Batch size (B).")
    p.add_argument("--seq", "-S", type=int, default=1, help="Tokens per sequence this step (S). Use 1 for decode.")
    p.add_argument("--D", type=int, help="Hidden size D.", default=7168)
    p.add_argument("--MD", type=int, help="Expert inner dim MD.", default=2048)
    p.add_argument("--MR", type=int, help="Top-k routed experts per token.", default=256)
    p.add_argument("--MS", type=int, help="Shared experts per layer (0 if none).", default=1)
    p.add_argument("--MA", type=int, help="Routing multiplicity per token (defaults to MR).", default=8)
    p.add_argument("--elem-bytes", type=int, default=2, choices=[1,2,4], help="Element bytes (e.g., 2 for bf16/fp16).")
    p.add_argument("--bytes-model", type=str, default="weights+activations",
                   choices=["weights-only", "weights+activations"],
                   help="Byte model for arithmetic intensity.")
    p.add_argument("--output-dir", type=str, default=".",
                    help="Where to save figures (default: plots)")
    args = p.parse_args()

    cfg = MoEConfig(D=args.D, MD=args.MD, MR=args.MR, MS=args.MS,
                    MA=args.MA, elem_bytes=args.elem_bytes)

    batches = [1, 4, 16, 64, 256, 1024]
    model_ois = {}
    gpu_ois = [
        # ('A100', 153),
        ('H100', 295),
        ('B200', 281),
    ]
    for batch in batches:
        flops, bytes_, oi = arithmetic_intensity(batch, args.seq, cfg, args.bytes_model)
        model_ois[batch] = oi

        print(f"Config: B={batch}, S={args.seq}, D={cfg.D}, MD={cfg.MD}, MR={cfg.MR}, MS={cfg.MS}, "
            f"elem_bytes={cfg.elem_bytes}, bytes_model={args.bytes_model}")
        print(f"MoE layer FLOPs: {flops:,.0f}")
        print(f"Estimated bytes moved: {bytes_:,.0f}")
        print(f"Arithmetic intensity (FLOPs/byte): {oi:.3f}")
    
    
    set_paper_style()
    apply_color_cycle(len(gpu_ois) + 1, "tableau10")
    
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = plt.gca()

    # clean axes
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.grid(True, axis="x", linestyle="--", alpha=0.35)

    # integer x axis from your `reps`
    x_vals = np.array(batches, dtype=float)
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xticks(x_vals)
    ax.set_xticklabels([str(int(xx)) for xx in x_vals])

    series_idx = 0

    y_vals = [model_ois[batch] for batch in batches]
    max_h = max(y_vals)

    marker = MARKERS[series_idx % len(MARKERS)]
    series_idx += 1
    ax.plot(
        x_vals, y_vals,
        marker=marker, linewidth=2.2, markersize=5.5,
        label=f"DeepSeek-V3 MoE FFN",
    )
    
    colors = PALETTES["tableau10"][1:]
    idx = 0
    for gpu, oi in gpu_ois:
        ax.axhline(y=oi, linestyle='--', linewidth=2.2, label=gpu, color=colors[idx])
        idx += 1
    max_h = max(max_h, max(x[1] for x in gpu_ois))

    ax.set_ylabel('Operational Intensity (FLOP/Byte)')
    ax.set_xlabel('Batch Size (Tokens)')
    # ax.set_title('')
    ax.legend()
    if max_h > 0:
        ax.set_ylim(-10, max_h * 1.15)

    fig.tight_layout()

    base = Path(args.output_dir) / f"deepseekv3_vs_gpu_oi"
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{base}.pdf", transparent=True)
    plt.close(fig)

if __name__ == "__main__":
    main()
