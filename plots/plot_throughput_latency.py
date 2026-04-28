#!/usr/bin/env python3
"""Plot total token throughput vs p99 TPOT/TTFT for EP vs TP comparison.

Reads bench_result.json files from results/vllm_results_final/.
Groups by dataset (0=random, 2=sharegpt) and backend configuration.
"""

import json
import math
import os
import re
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_DIR = Path("results/vllm_results_final")
OUTPUT_DIR = Path("plots")

# Knob: only group size is filtered globally; model is iterated per-run.
GROUP_FILTER = 1

# Directory name format:
# {gpus}_{ep}_{use_ep}_{replicas}_{batch}_{routing}_{backend}_{dataset}_{profiler}_{threshold}_{model}_g{groups}
# e.g. 8_8_1_64_32_2_dispatch_combine_0_0_256_Qwen3-30B-A3B-8-128_g1

# Threshold sentinel meaning "match any threshold > 0".
ANY_POSITIVE = ">0"


def model_num_experts(model_name):
    """Extract num_experts from a "...-{topk}-{num_experts}" model name.

    Returns None if the suffix isn't an integer.
    """
    m = re.search(r"-(\d+)$", model_name or "")
    return int(m.group(1)) if m else None


def build_configs(replicas_15x):
    """Legend configurations keyed by (use_ep, replicas, backend, threshold).

    The 1.5x rows use replicas_15x = num_experts // 2 for the current model.
    """
    return {
        # TP: use_ep=0, backend=allgather_reducescatter
        (0, 0, "allgather_reducescatter", 0): {
            "label": "TP",
            "color": "#1f77b4",
            "marker": "o",
        },
        # EP 1.0x: use_ep=1, 0 rep, dispatch_combine
        (1, 0, "dispatch_combine", 0): {
            "label": "EP 1.0x",
            "color": "#ff7f0e",
            "marker": "s",
        },
        # EP 1.5x: use_ep=1, replicas_15x rep, threshold=0
        (1, replicas_15x, "dispatch_combine", 0): {
            "label": "EP 1.5x",
            "color": "#2ca02c",
            "marker": "^",
        },
        # METRO 1.5x: use_ep=1, replicas_15x rep, any threshold > 0
        (1, replicas_15x, "dispatch_combine", ANY_POSITIVE): {
            "label": "METRO 1.5x",
            "color": "#d62728",
            "marker": "D",
        },
    }

DATASET_NAMES = {
    0: "InstructCoder",
    1: "TextEdit",
    2: "ShareGPT",
}


def parse_dirname(dirname):
    """Parse result directory name into config dict."""
    # Match: {gpus}_{ep}_{use_ep}_{replicas}_{batch}_{routing}_{backend}_{dataset}_{profiler}_{threshold}_{model}[_g{groups}]
    # backend can be multi-word with underscores, so match known backends.
    # _g{groups} suffix is optional for backward compatibility with older runs.
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


def trim_top(values):
    """Drop the top max(1, ceil(1% * n)) values; return the rest.

    Targets the structural prefill-queue tail where each client run
    exhibits a small number of "always-slow" prompts.
    """
    n = len(values)
    if n == 0:
        return []
    k = max(1, math.ceil(0.01 * n))
    if k >= n:
        return []
    return sorted(values, reverse=True)[k:]


def aggregate_bench_files(paths):
    """Aggregate one or more bench_result*.json files for a single run.

    Throughput is the simple average of total_token_throughput across
    files. P99 and median latency both use ONLY the last client (highest
    CLIENT_IDX) after trimming its top max(1, ceil(1% * n_last))
    per-prompt latencies; the trimmed set's max is reported as P99 and
    its statistical median as the median. Falls back to per-file
    aggregate {p99,median}_*_ms when --save-detailed data is absent
    (legacy files).
    """
    def client_idx(p):
        m = re.search(r"bench_result_(\d+)\.json$", p.name)
        return int(m.group(1)) if m else -1

    paths = sorted(paths, key=client_idx)

    throughputs = []
    last_ttft_ms = []
    last_tpot_ms = []
    legacy = {k: [] for k in
              ("p99_ttft", "p99_tpot", "median_ttft", "median_tpot")}
    for i, p in enumerate(paths):
        try:
            data = json.loads(p.read_text())
        except json.JSONDecodeError:
            print(f"  Skipping invalid JSON: {p}")
            continue
        if "total_token_throughput" in data:
            throughputs.append(data["total_token_throughput"])
        is_last = (i == len(paths) - 1)
        ttfts = data.get("ttfts")  # seconds, per request
        itls = data.get("itls")    # seconds, list[list] per request
        olens = data.get("output_lens")
        has_detailed = bool(ttfts) and bool(itls) and bool(olens)
        if is_last:
            if has_detailed:
                last_ttft_ms = [t * 1000.0 for t in ttfts]
                last_tpot_ms = [
                    (sum(il) / (ol - 1)) * 1000.0
                    for il, ol in zip(itls, olens) if ol > 1 and il
                ]
            else:
                # Legacy aggregate-only file: no per-prompt data to trim.
                for key in legacy:
                    field = f"{key}_ms"
                    if field in data:
                        legacy[key].append(data[field])
    if not throughputs:
        return None
    avg_throughput = sum(throughputs) / len(throughputs)

    def _avg(xs):
        return sum(xs) / len(xs) if xs else 0

    if last_ttft_ms and last_tpot_ms:
        trimmed_ttft = trim_top(last_ttft_ms)
        trimmed_tpot = trim_top(last_tpot_ms)
        p99_ttft = max(trimmed_ttft) if trimmed_ttft else 0
        p99_tpot = max(trimmed_tpot) if trimmed_tpot else 0
        median_ttft = statistics.median(trimmed_ttft) if trimmed_ttft else 0
        median_tpot = statistics.median(trimmed_tpot) if trimmed_tpot else 0
    else:
        p99_ttft = _avg(legacy["p99_ttft"])
        p99_tpot = _avg(legacy["p99_tpot"])
        median_ttft = _avg(legacy["median_ttft"])
        median_tpot = _avg(legacy["median_tpot"])
    return {
        "throughput": avg_throughput,
        "p99_ttft": p99_ttft,
        "p99_tpot": p99_tpot,
        "median_ttft": median_ttft,
        "median_tpot": median_tpot,
    }


def load_results():
    """Load and aggregate per-run bench_result*.json files."""
    results = []
    for d in sorted(RESULTS_DIR.iterdir()):
        if not d.is_dir():
            continue
        cfg = parse_dirname(d.name)
        if cfg is None:
            print(f"  Skipping unparseable dir: {d.name}")
            continue
        # Prefer multi-client bench_result_*.json; fall back to legacy
        # single-file bench_result.json.
        bench_files = sorted(d.glob("bench_result_*.json"))
        if not bench_files:
            legacy = d / "bench_result.json"
            if legacy.exists():
                bench_files = [legacy]
        if not bench_files:
            continue
        agg = aggregate_bench_files(bench_files)
        if agg is None:
            print(f"  Skipping, no usable data: {d.name}")
            continue
        cfg.update(agg)
        results.append(cfg)
    return results


def model_slug(model_name):
    """Filename-safe lowercase form of a model name."""
    return re.sub(r"[^a-z0-9]+", "-", model_name.lower()).strip("-")


def plot_dataset(results, model, configs, dataset_id, metric, ylabel,
                 filename):
    """Plot throughput vs metric for one (model, dataset)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for config_key, style in configs.items():
        use_ep, replicas, backend, threshold = config_key
        if threshold == ANY_POSITIVE:
            thr_match = lambda t: t > 0
        else:
            thr_match = lambda t, th=threshold: t == th
        # Filter matching results
        pts = [
            r for r in results
            if r["dataset"] == dataset_id
            and r["use_ep"] == use_ep
            and r["replicas"] == replicas
            and r["backend"] == backend
            and thr_match(r["threshold"])
        ]
        if not pts:
            continue
        # Sort by batch size
        pts.sort(key=lambda r: r["batch"])
        x = [r["throughput"] for r in pts]
        y = [r[metric] for r in pts]
        batches = [r["batch"] for r in pts]

        ax.plot(y, x,
                label=style["label"],
                color=style["color"],
                marker=style["marker"],
                markersize=8,
                linewidth=2)

        # Annotate batch sizes
        for xi, yi, b in zip(x, y, batches):
            ax.annotate(f"B={b}",
                        (yi, xi),
                        textcoords="offset points",
                        xytext=(5, 5),
                        fontsize=7,
                        color=style["color"])

    dataset_name = DATASET_NAMES.get(dataset_id, f"Dataset {dataset_id}")
    ax.set_xlabel(ylabel, fontsize=12)
    ax.set_ylabel("Total Token Throughput (tok/s)", fontsize=12)
    ax.set_title(f"{model} | {dataset_name}: {ylabel} vs Throughput",
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    out_path = OUTPUT_DIR / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def main():
    print("Loading results...")
    results = load_results()
    # Filter by group only; iterate over all available models.
    results = [r for r in results if r.get("groups") == GROUP_FILTER]
    if not results:
        print(f"No results found for g{GROUP_FILTER}!")
        sys.exit(1)

    models = sorted({r["model"] for r in results})
    print(f"Models (g{GROUP_FILTER}): {models}")

    for model in models:
        model_results = [r for r in results if r["model"] == model]
        num_experts = model_num_experts(model)
        if num_experts is not None:
            replicas_15x = num_experts // 2
            replicas_src = f"name (num_experts={num_experts})"
        else:
            # Fall back to observed replicas in the data: the largest
            # nonzero replicas seen for use_ep=1 runs is the 1.5x count.
            observed = sorted({
                r["replicas"] for r in model_results
                if r["use_ep"] == 1 and r["replicas"] > 0
            })
            if observed:
                replicas_15x = max(observed)
                replicas_src = f"data (observed nonzero replicas={observed})"
            else:
                replicas_15x = -1  # unmatchable
                replicas_src = "none (no 1.5x rows)"
        configs = build_configs(replicas_15x)
        slug = model_slug(model)
        datasets = sorted({r["dataset"] for r in model_results})
        print(f"  [{model}] {len(model_results)} runs, "
              f"replicas_15x={replicas_15x} via {replicas_src}, "
              f"datasets={datasets}")

        metric_specs = [
            ("p99_tpot",    "P99 TPOT (ms)",    "p99tpot"),
            ("p99_ttft",    "P99 TTFT (ms)",    "p99ttft"),
            ("median_tpot", "Median TPOT (ms)", "mediantpot"),
            ("median_ttft", "Median TTFT (ms)", "medianttft"),
        ]
        for ds in [0, 1, 2]:
            if ds not in datasets:
                continue
            ds_name = DATASET_NAMES.get(ds, str(ds)).lower()
            for metric_key, metric_label, metric_slug in metric_specs:
                plot_dataset(
                    model_results, model, configs, ds, metric_key,
                    metric_label,
                    f"throughput_vs_{metric_slug}_{slug}_{ds_name}.pdf",
                )

    print("Done!")


if __name__ == "__main__":
    main()
