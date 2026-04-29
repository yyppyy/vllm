#!/usr/bin/env python3
"""Plot per-(rank, layer, batch) activated-experts vs MoE kernel latency.

Reads ExpLat records emitted by bench_exp_vs_latency.sh from
`results/vllm_results_final/<run_hash>/server_explat.log`. Each ExpLat
record becomes one scatter point.

  x = number of experts with > 0 tokens in this rank/layer/batch
  y = latency in microseconds; toggle via --metric:
        expert_compute (default) — sum of all 5 kernels
        gemm                     — gemm_gu_us + gemm_dn_us only

Produces one figure per (model, dataset) pair, combining all other
configurations (batch size, replicas, GPUs, EP, threshold, groups, etc.)
into the same figure. Adds an OLS linear-fit line.

Directory naming follows plot_throughput_latency.parse_dirname.
"""

import argparse
import ast
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("results/vllm_results_final")
OUTPUT_DIR = Path("plots")

DATASET_NAMES = {
    0: "InstructCoder",
    1: "TextEdit",
    2: "ShareGPT",
}


def parse_dirname(dirname):
    """Mirror of plot_throughput_latency.parse_dirname."""
    for backend_name in ["allgather_reducescatter", "dispatch_combine"]:
        pattern = (
            r"^(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_"
            + re.escape(backend_name)
            + r"_(\d+)_(\d+)_(\d+)_(.+?)(?:_g(\d+))?$"
        )
        m = re.match(pattern, dirname)
        if m:
            return {
                "gpus": int(m.group(1)),
                "ep_degree": int(m.group(2)),
                "use_ep": int(m.group(3)),
                "replicas": int(m.group(4)),
                "batch": int(m.group(5)),
                "routing": int(m.group(6)),
                "backend": backend_name,
                "dataset": int(m.group(7)),
                "profiler": int(m.group(8)),
                "threshold": int(m.group(9)),
                "model": m.group(10),
                "groups": int(m.group(11)) if m.group(11) else 1,
            }
    return None


# One ExpLat line. Format emitted by the in-process explat poller:
#   ExpLat seq=N rank=R layer=L M=M num_local_experts=E
#          align_ns=… gemm_gu_ns=… silu_ns=… quant_ns=… gemm_dn_ns=…
#          per_expert_tokens=[…]
EXPLAT_RE = re.compile(
    r"ExpLat seq=(?P<seq>-?\d+) "
    r"rank=(?P<rank>-?\d+) layer=(?P<layer>-?\d+) "
    r"M=(?P<M>\d+) "
    r"num_local_experts=(?P<num_local_experts>\d+) "
    r"align_ns=(?P<align>[-\d]+) "
    r"gemm_gu_ns=(?P<gemm_gu>[-\d]+) "
    r"silu_ns=(?P<silu>[-\d]+) "
    r"quant_ns=(?P<quant>[-\d]+) "
    r"gemm_dn_ns=(?P<gemm_dn>[-\d]+) "
    r"per_expert_tokens=(?P<pet>\[[^\]]*\])"
)


def parse_explat_log(path, min_batch, max_batch, block_size_m):
    """Yield one dict per ExpLat record in the log file. Drops records
    with M < min_batch or M > max_batch (None on either bound disables
    that side of the filter).

    post_pad is computed assuming BLOCK_SIZE_M == block_size_m (the
    Triton autotuner picks 16/32/64; default 16 matches the small-M
    case the user runs)."""
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return
    bsm = int(block_size_m)
    for m in EXPLAT_RE.finditer(text):
        M = int(m.group("M"))
        if min_batch is not None and M < min_batch:
            continue
        if max_batch is not None and M > max_batch:
            continue
        try:
            pet = ast.literal_eval(m.group("pet"))
        except (ValueError, SyntaxError):
            continue
        if not pet:
            continue
        n_active = sum(1 for x in pet if x > 0)
        if n_active == 0:
            # Layer/batch where this rank received no tokens -- not
            # informative for the activation-vs-latency relationship.
            continue
        max_expert = max(pet)
        # Σ_e ceil(t_e / BSM) * BSM -- the actual padded work fed
        # into the fused_moe_kernel.
        post_pad = sum(((t + bsm - 1) // bsm) * bsm for t in pet)
        # Convert ns -> us for plotting. The new explat path emits
        # raw GPU %globaltimer deltas in nanoseconds.
        gu_us = float(m.group("gemm_gu")) / 1000.0
        dn_us = float(m.group("gemm_dn")) / 1000.0
        align_us = float(m.group("align")) / 1000.0
        silu_us = float(m.group("silu")) / 1000.0
        quant_us = float(m.group("quant")) / 1000.0
        sum_us = align_us + gu_us + silu_us + quant_us + dn_us
        # Per-rank receive count is the sum of per-expert tokens; mc is
        # not directly reported by the new format but can be derived
        # client-side from cfg if needed.
        yield {
            "n_active": n_active,
            "n_local_experts": len(pet),
            "expert_compute_us": sum_us,
            "gemm_us": gu_us + dn_us,
            "gemm_gu_us": gu_us,
            "gemm_dn_us": dn_us,
            "align_us": align_us,
            "silu_us": silu_us,
            "quant_us": quant_us,
            "M": M,
            "local_tokens": sum(pet),
            "mc": sum(pet),  # mc not in new format; fallback to local_tokens
            "max_expert": max_expert,
            "post_pad": post_pad,
            "rank": int(m.group("rank")),
            "layer": int(m.group("layer")),
            "seq": int(m.group("seq")),
        }


def collect(results_dir, log_filename, min_batch, max_batch,
            block_size_m):
    """Group ExpLat records by (model, dataset)."""
    grouped = defaultdict(list)
    seen_dirs = 0
    seen_logs = 0
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir():
            continue
        cfg = parse_dirname(d.name)
        if cfg is None:
            continue
        seen_dirs += 1
        log = d / log_filename
        if not log.exists():
            continue
        seen_logs += 1
        key = (cfg["model"], cfg["dataset"])
        for rec in parse_explat_log(log, min_batch, max_batch,
                                    block_size_m):
            rec["cfg"] = cfg
            grouped[key].append(rec)
    print(f"  scanned {seen_dirs} run dirs, {seen_logs} had {log_filename}")
    return grouped


XAXIS_LABELS = {
    "n_active":     "activated experts (per rank · layer · batch)",
    "local_tokens": "local_tokens received by rank "
                    "(post-dispatch)",
    "post_pad":     "Σ_e ceil(t_e/BSM)·BSM "
                    "(post-padded fused_moe input)",
    "mc":           "mc = min(M·top_k·2, max_recv) "
                    "(buffer-capacity bound)",
    "M":            "global batch size M",
}
XAXIS_TAGS = {"n_active": "nact", "local_tokens": "lt",
              "post_pad": "pp", "mc": "mc", "M": "M"}

COLOR_BY_LABELS = {
    "max_expert":   "max per-expert tokens (skew)",
    "local_tokens": "local_tokens",
    "n_active":     "activated experts",
    "M":            "global batch size M",
    "post_pad":     "post_pad",
}


METRIC_INFO = {
    # metric -> (record key, ylabel, filename tag)
    "expert_compute": ("expert_compute_us",
                       "expert compute latency (us)\n[sum of 5 kernels]",
                       "exp"),
    "gemm":           ("gemm_us",
                       "fused_moe_kernel latency (us)\n"
                       "[gemm_gu + gemm_dn]",
                       "gemm"),
    "gemm_gu":        ("gemm_gu_us",
                       "gate_up GEMM latency (us)",
                       "gu"),
    "gemm_dn":        ("gemm_dn_us",
                       "down GEMM latency (us)",
                       "dn"),
    "align":          ("align_us",
                       "moe_align_block_size latency (us)",
                       "align"),
    "silu":           ("silu_us",
                       "silu / silu_and_mul_ep latency (us)",
                       "silu"),
    "quant":          ("quant_us",
                       "moe_kernel_quantize_input latency (us)",
                       "quant"),
}


def make_plot(model, dataset, records, metric, xaxis, color_by,
              out_dir):
    if not records:
        return
    xs = np.array([r[xaxis] for r in records], dtype=float)
    rec_key, ylabel, metric_tag = METRIC_INFO[metric]
    ys = np.array([r[rec_key] for r in records], dtype=float)

    a, b = np.polyfit(xs, ys, 1)
    r = float(np.corrcoef(xs, ys)[0, 1]) if len(xs) > 1 else float("nan")

    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    if color_by != "none":
        cs = np.array([r[color_by] for r in records], dtype=float)
        sc = ax.scatter(xs, ys, c=cs, cmap="viridis",
                        s=12, alpha=0.55, edgecolor="none")
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(COLOR_BY_LABELS[color_by], fontsize=9)
    else:
        ax.scatter(xs, ys, s=10, alpha=0.25, edgecolor="none",
                   color="#1f77b4", label=f"n={len(xs)}")
    x_line = np.array([xs.min(), xs.max()])
    ax.plot(x_line, a * x_line + b, color="#d62728", linewidth=1.6,
            label=f"y = {a:.3f}x + {b:.1f}   R = {r:.3f}  "
                  f"(n={len(xs)})")

    ax.set_xlabel(XAXIS_LABELS[xaxis])
    ax.set_ylabel(ylabel)
    dataset_name = DATASET_NAMES.get(dataset, f"dataset{dataset}")
    ax.set_title(f"{model}  /  {dataset_name}")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    safe_model = model.replace("/", "_")
    color_tag = "" if color_by == "none" else f"_by{color_by}"
    out = out_dir / (
        f"activated_vs_latency_{metric_tag}_"
        f"x{XAXIS_TAGS[xaxis]}{color_tag}_"
        f"{safe_model}_{dataset_name}.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}  (n={len(xs)}, slope={a:.3f}, R={r:.3f})")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                   help="Root containing per-run hash subdirectories.")
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                   help="Where to drop the .pdf figures.")
    p.add_argument(
        "--metric",
        choices=tuple(METRIC_INFO.keys()),
        default="expert_compute",
        help="y-axis. expert_compute (default) = sum of 5 kernels; "
             "gemm = gemm_gu+gemm_dn; gemm_gu / gemm_dn / align / "
             "silu / quant = individual kernels (use these to "
             "diagnose dispatch-overhead vs real GPU work — the "
             "non-Triton kernels (silu, quant) should be cheap "
             "and scale with token count if timing is honest).")
    p.add_argument("--log-filename", default="server_explat.log",
                   help="Per-run log filename to parse "
                        "(default: server_explat.log).")
    p.add_argument("--min-batch", type=int, default=4,
                   help="Drop ExpLat records with M < min_batch "
                        "(default: 4). Set to 0 or negative to "
                        "disable the filter.")
    p.add_argument("--max-batch", type=int, default=128,
                   help="Drop ExpLat records with M > max_batch "
                        "(default: 128). Set to 0 or negative to "
                        "disable the filter.")
    p.add_argument(
        "--xaxis",
        choices=tuple(XAXIS_LABELS.keys()),
        default="n_active",
        help="x-axis predictor (default: n_active). "
             "local_tokens = actual tokens received by this rank; "
             "post_pad = Σ ceil(t_e/BSM)·BSM (padded fused_moe "
             "input); mc = the buffer-capacity bound the kernel "
             "grid is actually sized by; M = global batch size.")
    p.add_argument(
        "--color-by",
        choices=("none", "max_expert", "local_tokens",
                 "n_active", "M", "post_pad"),
        default="none",
        help="If set, color scatter points by this field "
             "(default: single color). max_expert highlights "
             "per-expert skew.")
    p.add_argument(
        "--block-size-m", type=int, default=16,
        help="Triton BLOCK_SIZE_M assumed when computing post_pad "
             "(default: 16; the autotuner usually picks 16 for "
             "small M).")
    args = p.parse_args()

    min_batch = args.min_batch if args.min_batch > 0 else None
    max_batch = args.max_batch if args.max_batch > 0 else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    grouped = collect(args.results_dir, args.log_filename,
                      min_batch, max_batch, args.block_size_m)
    if not grouped:
        print(f"No ExpLat records found under {args.results_dir} "
              f"(looking for {args.log_filename}).", file=sys.stderr)
        return 1
    for (model, dataset), records in sorted(grouped.items()):
        print(f"{model} / dataset={dataset}: {len(records)} records")
        make_plot(model, dataset, records, args.metric,
                  args.xaxis, args.color_by, args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
