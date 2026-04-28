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


# One ExpLat line. The vLLM logger prepends a level/time/file prefix,
# so we don't anchor to start of line.
EXPLAT_RE = re.compile(
    r"ExpLat rank=(?P<rank>-?\d+) layer=(?P<layer>-?\d+) "
    r"M=(?P<M>\d+) local_tokens=(?P<local_tokens>\d+) "
    r"expert_compute_us=(?P<expert_compute>[-\d.]+) "
    r"window_us=(?P<window>[-\d.]+) "
    r"gemm_gu_us=(?P<gemm_gu>[-\d.]+) "
    r"gemm_dn_us=(?P<gemm_dn>[-\d.]+) "
    r"align_us=(?P<align>[-\d.]+) "
    r"silu_us=(?P<silu>[-\d.]+) "
    r"quant_us=(?P<quant>[-\d.]+) "
    r"per_expert_tokens=(?P<pet>\[[^\]]*\])"
)


def parse_explat_log(path):
    """Yield one dict per ExpLat record in the log file."""
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
        n_active = sum(1 for x in pet if x > 0)
        if n_active == 0:
            # Layer/batch where this rank received no tokens -- not
            # informative for the activation-vs-latency relationship.
            continue
        yield {
            "n_active": n_active,
            "n_local_experts": len(pet),
            "expert_compute_us": float(m.group("expert_compute")),
            "gemm_us": (float(m.group("gemm_gu"))
                        + float(m.group("gemm_dn"))),
            "M": int(m.group("M")),
            "rank": int(m.group("rank")),
            "layer": int(m.group("layer")),
        }


def collect(results_dir, log_filename):
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
        for rec in parse_explat_log(log):
            rec["cfg"] = cfg
            grouped[key].append(rec)
    print(f"  scanned {seen_dirs} run dirs, {seen_logs} had {log_filename}")
    return grouped


def make_plot(model, dataset, records, metric, out_dir):
    if not records:
        return
    xs = np.array([r["n_active"] for r in records], dtype=float)
    if metric == "gemm":
        ys = np.array([r["gemm_us"] for r in records], dtype=float)
        ylabel = "fused_moe_kernel latency (us)\n[gemm_gu + gemm_dn]"
        metric_tag = "gemm"
    else:
        ys = np.array([r["expert_compute_us"] for r in records], dtype=float)
        ylabel = "expert compute latency (us)\n[sum of 5 kernels]"
        metric_tag = "exp"

    a, b = np.polyfit(xs, ys, 1)
    r = float(np.corrcoef(xs, ys)[0, 1]) if len(xs) > 1 else float("nan")

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.scatter(xs, ys, s=10, alpha=0.25, edgecolor="none",
               color="#1f77b4", label=f"n={len(xs)}")
    x_line = np.array([xs.min(), xs.max()])
    ax.plot(x_line, a * x_line + b, color="#d62728", linewidth=1.6,
            label=f"y = {a:.3f}x + {b:.1f}   R = {r:.3f}")

    ax.set_xlabel("activated experts (per rank · layer · batch)")
    ax.set_ylabel(ylabel)
    dataset_name = DATASET_NAMES.get(dataset, f"dataset{dataset}")
    ax.set_title(f"{model}  /  {dataset_name}")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    safe_model = model.replace("/", "_")
    out = out_dir / (
        f"activated_vs_latency_{metric_tag}_{safe_model}_{dataset_name}.pdf")
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
        choices=("expert_compute", "gemm"),
        default="expert_compute",
        help="y-axis: expert_compute (default; sum of 5 kernels) "
             "or gemm (gemm_gu + gemm_dn only).")
    p.add_argument("--log-filename", default="server_explat.log",
                   help="Per-run log filename to parse "
                        "(default: server_explat.log).")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    grouped = collect(args.results_dir, args.log_filename)
    if not grouped:
        print(f"No ExpLat records found under {args.results_dir} "
              f"(looking for {args.log_filename}).", file=sys.stderr)
        return 1
    for (model, dataset), records in sorted(grouped.items()):
        print(f"{model} / dataset={dataset}: {len(records)} records")
        make_plot(model, dataset, records, args.metric, args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
