#!/usr/bin/env python3
"""CDF of the fraction of MoE experts in the memory-bound regime.

For each (model, dataset) pair this script reads every ExpLat record
in `results/vllm_results_final/<run_hash>/server_tokcnt.log` and
emits one figure. Each (rank, layer, batch) record contributes one
data point: the fraction of that record's local experts whose
processed token count is below the model-specific memory-bound
threshold. The script draws one CDF line per unique global batch
size M.

The threshold (tokens/expert below which a gated FFN expert is
memory-bandwidth-bound) is derived from each model's HuggingFace
config.json (fetched online and cached under plots/.cache/) plus a
GPU roofline preset (default H100 SXM bf16 dense).

Roofline derivation (GEMM-only, SRAM-tile-aware)
------------------------------------------------
Per-expert gated FFN at M tokens, counting only the three GEMMs
(gate, up, down) — silu+mul is an O(M*MD) elementwise kernel and is
dropped from the byte budget by request:

  FLOPs(M) = 6 * M * D * MD
  Bytes(M) = elem_bytes * 3 * (D*MD                    # weights
                                + M*(D + MD))          # activations
                                                       # (x read twice,
                                                       #  intermediate,
                                                       #  y written)
  AI(M)    = FLOPs / Bytes
           = (2 * M * D * MD) / (elem_bytes * (D*MD + M*(D + MD)))

By default we cap the kernel's achievable AI only at the GPU compute
ridge (`peak_FLOPs / peak_BW`). A well-tuned GEMM keeps weights/
activations hot in L2 across tiles and saturates compute long before
the per-tile-isolated AI (`2·BM·BN / ((BM+BN)·elem_bytes)`) becomes
the binding constraint. Pass `--block-m`/`--block-n` to opt into the
conservative *no-L2-reuse* SRAM tile cap:

  AI_tile  = 2 * BM * BN / ((BM + BN) * elem_bytes)
  ridge_eff = min(peak_FLOPs / peak_BW, AI_tile)

Memory-bound iff AI(M) < ridge_eff. Solving:

  M_thresh = ridge_eff * elem_bytes * D * MD
             / (2 * D * MD - ridge_eff * elem_bytes * (D + MD))
"""

import argparse
import ast
import json
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from style import (apply_style, paper_figure, save_fig, palette,
                   style_axes, style_legend)

apply_style()

RESULTS_DIR = Path("results/vllm_results_final")
OUTPUT_DIR = Path("plots")
CACHE_DIR = Path("plots/.cache")

DATASET_NAMES = {
    0: "InstructCoder",
    1: "Edit_5k_char",
    2: "ShareGPT",
}

# ---------------------------------------------------------------------
# Model name -> HF repo
# ---------------------------------------------------------------------
HF_PATHS = {
    "Qwen3-30B-A3B": "Qwen/Qwen3-30B-A3B",
    "ERNIE-4.5-21B-A3B-PT": "baidu/ERNIE-4.5-21B-A3B-PT",
}

# Final fallback if HF is unreachable. Keep keys aligned with what
# `extract_dims` reads.
KNOWN_CONFIGS = {
    "Qwen3-30B-A3B": {
        "hidden_size": 2048,
        "moe_intermediate_size": 768,
        "torch_dtype": "bfloat16",
    },
    "ERNIE-4.5-21B-A3B-PT": {
        "hidden_size": 2560,
        "moe_intermediate_size": 1536,
        "torch_dtype": "bfloat16",
    },
}


def base_model_name(name: str) -> str:
    """Strip the topk/num_experts suffix from synthetic Qwen variants
    so HF lookups hit the real repo."""
    m = re.match(r"^(Qwen3-30B-A3B)(-\d+-\d+)?$", name)
    if m:
        return m.group(1)
    return name


def fetch_hf_config(model_name: str) -> dict:
    """Return the HF config.json for `model_name`, with a local cache.
    Falls back to KNOWN_CONFIGS if the network is unreachable."""
    base = base_model_name(model_name)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"hf_{base.replace('/', '_')}.json"
    if cache.exists():
        try:
            return json.loads(cache.read_text())
        except json.JSONDecodeError:
            pass
    hf_path = HF_PATHS.get(base, base)
    url = f"https://huggingface.co/{hf_path}/raw/main/config.json"
    try:
        with urllib.request.urlopen(url, timeout=15) as f:
            data = json.loads(f.read())
        cache.write_text(json.dumps(data, indent=2))
        return data
    except Exception as e:
        print(f"  WARN: HF fetch for {base} failed ({e}); using "
              f"KNOWN_CONFIGS fallback.", file=sys.stderr)
        return KNOWN_CONFIGS.get(base, {})


def extract_dims(cfg: dict) -> tuple[int, int, int]:
    """Return (D, MD, elem_bytes) for the per-expert gated FFN."""
    D = cfg.get("hidden_size") or cfg.get("d_model")
    MD = (cfg.get("moe_intermediate_size")
          or cfg.get("expert_intermediate_size")
          or cfg.get("intermediate_size"))
    if D is None or MD is None:
        raise ValueError(f"could not extract hidden/intermediate dims "
                          f"from config: {cfg}")
    dtype = (cfg.get("torch_dtype") or "bfloat16").lower()
    elem_bytes = {
        "bfloat16": 2, "float16": 2, "half": 2,
        "float32": 4, "float": 4,
        "fp8": 1, "float8": 1,
    }.get(dtype, 2)
    return int(D), int(MD), int(elem_bytes)


# ---------------------------------------------------------------------
# Roofline / threshold
# ---------------------------------------------------------------------
GPU_PRESETS = {
    # (peak_dense_bf16_TFLOPs, peak_HBM_TB_per_s)
    "A100_40GB": (312, 1.555),   # A100-SXM4-40GB,  HBM2  @ 1.555 TB/s
    "A100_80GB": (312, 2.04),    # A100-SXM4-80GB,  HBM2e @ 2.04  TB/s
    "H100":      (989, 3.35),    # H100-SXM        HBM3  @ 3.35  TB/s
    "H200":      (989, 4.80),
    "B200":      (2250, 8.00),
}


def tile_ai(block_m: int, block_n: int, elem_bytes: int) -> float:
    """Maximum arithmetic intensity an output (BM, BN) GEMM tile can
    deliver: AI_tile = 2*BM*BN / ((BM+BN) * elem_bytes)."""
    return 2.0 * block_m * block_n / ((block_m + block_n) * elem_bytes)


def memory_bound_threshold(D: int, MD: int, elem_bytes: int,
                            gpu: str,
                            block_m: int | None = None,
                            block_n: int | None = None
                            ) -> tuple[float, float, float]:
    """Solve AI(M) = ridge_eff for M, using GEMM-only bytes.

    By default `ridge_eff = GPU_ridge` (the kernel is assumed to
    saturate the GPU compute roofline thanks to L2 reuse). When both
    `block_m` and `block_n` are provided the SRAM-tile ceiling
    `AI_tile = 2*BM*BN / ((BM+BN)*elem_bytes)` is taken as a second
    upper bound on `ridge_eff`.

    Returns (M_thresh, ridge_eff, gpu_ridge).
    """
    flops, bw = GPU_PRESETS[gpu]
    gpu_ridge = flops * 1e12 / (bw * 1e12)         # FLOPs / byte
    ridge_eff = gpu_ridge
    if block_m is not None and block_n is not None:
        sram_ridge = tile_ai(block_m, block_n, elem_bytes)
        ridge_eff = min(ridge_eff, sram_ridge)
    num = ridge_eff * elem_bytes * D * MD
    den = 2.0 * D * MD - ridge_eff * elem_bytes * (D + MD)
    if den <= 0:
        # AI saturates below ridge_eff at all M -> always memory-bound.
        return float("inf"), ridge_eff, gpu_ridge
    return num / den, ridge_eff, gpu_ridge


# ---------------------------------------------------------------------
# Log parsing (mirror of plot_activated_vs_latency.py)
# ---------------------------------------------------------------------
EXPLAT_RE = re.compile(
    r"ExpLat seq=(?P<seq>-?\d+) "
    r"rank=(?P<rank>-?\d+) layer=(?P<layer>-?\d+) "
    r"M=(?P<M>\d+) "
    r"num_local_experts=(?P<num_local_experts>\d+) "
    r"align_ns=[-\d]+ gemm_gu_ns=[-\d]+ silu_ns=[-\d]+ "
    r"quant_ns=[-\d]+ gemm_dn_ns=[-\d]+ "
    r"per_expert_tokens=(?P<pet>\[[^\]]*\])"
)


def parse_dirname(dirname: str) -> dict | None:
    for backend_name in ["allgather_reducescatter", "dispatch_combine"]:
        pattern = (
            r"^(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_"
            + re.escape(backend_name)
            + r"_(\d+)_(\d+)_(\d+)_(.+?)(?:_g(\d+))?$"
        )
        m = re.match(pattern, dirname)
        if m:
            return {
                "model": m.group(10),
                "dataset": int(m.group(7)),
                "batch": int(m.group(5)),  # bench-config batch size
            }
    return None


def parse_explat_log(path: Path):
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return
    for m in EXPLAT_RE.finditer(text):
        try:
            pet = ast.literal_eval(m.group("pet"))
        except (ValueError, SyntaxError):
            continue
        if not pet:
            continue
        yield {
            "M": int(m.group("M")),
            "rank": int(m.group("rank")),
            "layer": int(m.group("layer")),
            "pet": pet,
        }


# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------
def plot_cdf(per_run_records, model, dataset, threshold, out_dir,
             min_per_line=20):
    """One CDF line per LOG FILE (= one bench batch-size config).

    `per_run_records` is a dict {bench_batch: [records...]}. Each
    record contributes one data point: the *fraction* of its local
    experts whose tokens are below the threshold (count divided by
    num_local_experts, normalized to [0, 1]). All per-record M values
    inside one log are merged into the same line.
    """
    fracs_by_run: dict[int, list[float]] = defaultdict(list)
    for batch, records in per_run_records.items():
        for r in records:
            n = len(r["pet"])
            if n == 0:
                continue
            cnt = sum(1 for t in r["pet"] if t < threshold)
            fracs_by_run[batch].append(cnt / n)
    fracs_by_run = {
        b: vs for b, vs in fracs_by_run.items()
        if len(vs) >= min_per_line
    }
    if not fracs_by_run:
        return None

    fig, ax = paper_figure()
    # Tight margins: log-scale y tick labels ("0.001"…) are wider
    # than the linear set the global PANEL_MARGINS were tuned for, so
    # left needs a small bump from 0.17, but otherwise pull the axes
    # close to the canvas edge.
    fig.subplots_adjust(left=0.19, right=0.99,
                         bottom=0.19, top=0.97)
    batches = sorted(fracs_by_run.keys())
    colors = palette(len(batches), name="tableau10")
    smallest_y = 100.0
    # Per-batch fraction of records where ALL local experts are
    # memory-bound (= the size of the CDF's step at x=1.0). Used to
    # draw a reference line per batch.
    all_bound_frac: dict[int, float] = {}
    for i, b in enumerate(batches):
        vals = sorted(fracs_by_run[b])
        n = len(vals)
        x = np.array(vals, dtype=float)
        y = np.arange(1, n + 1) / n * 100.0
        smallest_y = min(smallest_y, y[0])
        n_all = int(np.sum(x >= 1.0 - 1e-9))
        all_bound_frac[b] = n_all / n if n else 0.0
        # Anchor the right end at 1.0 (every record is at most
        # 100 % memory-bound).
        x = np.concatenate((x, [1.0]))
        y = np.concatenate((y, [100.0]))
        ax.plot(x, y, drawstyle="steps-post",
                label=f"B={b}", color=colors[i],
                linewidth=2.0)

    # Fixed ticks at 0, 0.2, 0.4, 0.6, 0.8, 1.0 so all CDFs share a
    # uniform x-axis regardless of where the data starts rising.
    xticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{t:.1f}" for t in xticks])
    # Log y so the small-CDF region (where lines slowly rise from 0)
    # is legible even when most of the mass is jammed near 1.0.
    y_bottom = 10 ** np.floor(np.log10(max(smallest_y, 1e-3)))
    style_axes(
        ax,
        x_label="Memory-bound experts (fraction)",
        y_label="CDF (%)",
        x_lim=(0.0, 1.0),
        y_lim=(y_bottom, 100),
        y_log=True,
    )
    # Keep the rightmost x-tick label inside the canvas (default
    # center alignment makes it spill past the figure edge).
    xtl = ax.get_xticklabels()
    if xtl:
        xtl[0].set_horizontalalignment("left")
        xtl[-1].set_horizontalalignment("right")

    # Reference lines: for each batch, draw a horizontal dotted line
    # at the CDF level just before its final jump to 100 %. The gap
    # from that line up to the top of the plot is the fraction of
    # records where every local expert is memory-bound. We annotate
    # each line with a short percentage; labels are staggered
    # horizontally so multiple batches with close y_pre don't collide.
    eligible = [(i, b) for i, b in enumerate(batches)
                if 0.0 < all_bound_frac[b] < 1.0
                and (1.0 - all_bound_frac[b]) * 100.0 > y_bottom]
    if eligible:
        # Reserve x-axis space [0.04, 0.96] in axes-fraction for the
        # row of percentage labels.
        x_lo, x_hi = 0.04, 0.96
        n_lab = len(eligible)
        x_step = (x_hi - x_lo) / max(n_lab, 1)
        for slot, (i, b) in enumerate(eligible):
            frac_all = all_bound_frac[b]
            y_pre = (1.0 - frac_all) * 100.0
            ax.axhline(y_pre, color=colors[i], linestyle=":",
                       linewidth=0.9, alpha=0.7, zorder=1)
            x_pos = x_lo + slot * x_step
            ax.text(x_pos, y_pre, f"{frac_all * 100:.1f}%",
                    color=colors[i],
                    fontsize=plt.rcParams["legend.fontsize"],
                    va="bottom", ha="left",
                    transform=ax.get_yaxis_transform())

    style_legend(ax, loc="lower right")

    ds_name = DATASET_NAMES.get(dataset, f"dataset{dataset}")
    safe_model = model.replace("/", "_")
    out = out_dir / f"membound_cdf_{safe_model}_{ds_name}.pdf"
    save_fig(fig, out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument(
        "--gpu", choices=tuple(GPU_PRESETS.keys()),
        default="A100_40GB",
        help="Roofline preset (default: A100 40GB bf16 dense).")
    p.add_argument(
        "--block-m", type=int, default=128,
        help="Tile rows. ridge_eff is capped at "
             "AI_tile = 2*BM*BN/((BM+BN)*elem_bytes); BM=128, "
             "BN=256 is the production sweet spot for A100/H100 "
             "bf16 GEMMs. Pass 0 to disable the SRAM cap entirely "
             "and use ridge_eff = GPU compute ridge.")
    p.add_argument(
        "--block-n", type=int, default=256,
        help="Tile cols (default: 256). See --block-m.")
    p.add_argument(
        "--min-per-line", type=int, default=20,
        help="Drop log files with fewer than this many records "
             "(default: 20). Avoids drawing CDFs from a handful "
             "of warmup samples.")
    args = p.parse_args()

    # Two-level grouping:
    #   key1 = (model, dataset) -> figure
    #   key2 = bench batch-size config (from dirname) -> CDF line
    grouped: dict[tuple[str, int], dict[int, list[dict]]] = defaultdict(
        lambda: defaultdict(list))
    seen_logs = 0
    seen_runs_per_pair: dict[tuple[str, int], list[int]] = defaultdict(list)
    for d in sorted(args.results_dir.iterdir()):
        if not d.is_dir():
            continue
        cfg = parse_dirname(d.name)
        if cfg is None:
            continue
        log = d / "server_tokcnt.log"
        if not log.exists():
            continue
        seen_logs += 1
        key = (cfg["model"], cfg["dataset"])
        bench_batch = cfg["batch"]
        seen_runs_per_pair[key].append(bench_batch)
        for rec in parse_explat_log(log):
            grouped[key][bench_batch].append(rec)
    if not grouped:
        print(f"No ExpLat records under {args.results_dir}",
              file=sys.stderr)
        return 1
    print(f"Parsed {seen_logs} log files into {len(grouped)} "
          f"(model, dataset) groups")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for (model, dataset), per_run in sorted(grouped.items()):
        try:
            cfg = fetch_hf_config(model)
            D, MD, elem_bytes = extract_dims(cfg)
        except Exception as e:
            print(f"  ERROR: {model}: {e}", file=sys.stderr)
            continue
        # block_m/block_n == 0 -> disable SRAM cap.
        bm = args.block_m if args.block_m and args.block_m > 0 else None
        bn = args.block_n if args.block_n and args.block_n > 0 else None
        threshold, ridge_eff, gpu_ridge = memory_bound_threshold(
            D, MD, elem_bytes, args.gpu, bm, bn)
        ds_name = DATASET_NAMES.get(dataset, f"dataset{dataset}")
        n_total = sum(len(v) for v in per_run.values())
        if bm is not None and bn is not None:
            tile = tile_ai(bm, bn, elem_bytes)
            ridge_str = (f"ridge_eff={ridge_eff:.1f} "
                          f"(gpu={gpu_ridge:.1f}, "
                          f"tile[{bm}x{bn}]={tile:.1f}) ")
        else:
            ridge_str = f"ridge_eff={ridge_eff:.1f} (no SRAM cap) "
        print(f"{model} / {ds_name}: D={D} MD={MD} "
              f"elem_bytes={elem_bytes} GPU={args.gpu} "
              f"=> {ridge_str}"
              f"threshold={threshold:.1f} tokens/expert "
              f"({n_total} records across "
              f"batches={sorted(per_run.keys())})")
        out = plot_cdf(per_run, model, dataset, threshold,
                       args.output_dir,
                       min_per_line=args.min_per_line)
        if out is not None:
            print(f"  wrote {out}")
        else:
            print(f"  skipped (no log file with "
                  f">= {args.min_per_line} records)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
