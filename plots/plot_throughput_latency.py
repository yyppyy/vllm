#!/usr/bin/env python3
"""Plot total token throughput vs p99 TPOT/TTFT for EP vs TP comparison.

Reads bench_result.json files from results/vllm_results_final/.
Groups by dataset (0=random, 2=sharegpt) and backend configuration.
"""

import json
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_DIR = Path("results/vllm_results_final")
OUTPUT_DIR = Path("plots")

# Directory name format:
# {gpus}_{ep}_{use_ep}_{replicas}_{batch}_{routing}_{backend}_{dataset}_{profiler}_{threshold}_{model}
# e.g. 8_8_1_64_32_2_dispatch_combine_0_0_256_Qwen3-30B-A3B-4-128

# Legend configurations: (use_ep, replicas, backend, threshold) -> label
CONFIGS = {
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
    # EP 1.5x: use_ep=1, 64 rep, threshold=0
    (1, 64, "dispatch_combine", 0): {
        "label": "EP 1.5x",
        "color": "#2ca02c",
        "marker": "^",
    },
    # METRO 1.5x: use_ep=1, 64 rep, threshold=256
    (1, 64, "dispatch_combine", 256): {
        "label": "METRO 1.5x",
        "color": "#d62728",
        "marker": "D",
    },
}

DATASET_NAMES = {
    0: "Random",
    2: "ShareGPT",
}


def parse_dirname(dirname):
    """Parse result directory name into config dict."""
    # Match: {gpus}_{ep}_{use_ep}_{replicas}_{batch}_{routing}_{backend}_{dataset}_{profiler}_{threshold}_{model}
    # backend can be multi-word with underscores, so match known backends
    for backend_name in ["allgather_reducescatter", "dispatch_combine"]:
        pattern = (
            r"^(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_"
            + re.escape(backend_name)
            + r"_(\d+)_(\d+)_(\d+)_(.+)$"
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
            }
    return None


def load_results():
    """Load all bench_result.json files."""
    results = []
    for d in sorted(RESULTS_DIR.iterdir()):
        if not d.is_dir():
            continue
        bench_file = d / "bench_result.json"
        if not bench_file.exists():
            continue
        cfg = parse_dirname(d.name)
        if cfg is None:
            print(f"  Skipping unparseable dir: {d.name}")
            continue
        try:
            data = json.loads(bench_file.read_text())
        except json.JSONDecodeError:
            print(f"  Skipping invalid JSON: {bench_file}")
            continue
        cfg["throughput"] = data.get("total_token_throughput", 0)
        cfg["p99_tpot"] = data.get("p99_tpot_ms", 0)
        cfg["p99_ttft"] = data.get("p99_ttft_ms", 0)
        results.append(cfg)
    return results


def filter_model(results, model_prefix="Qwen3"):
    """Filter results for a specific model prefix."""
    return [r for r in results if model_prefix in r.get("model", "")]


def plot_dataset(results, dataset_id, metric, ylabel, filename):
    """Plot throughput vs metric for one dataset."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for config_key, style in CONFIGS.items():
        use_ep, replicas, backend, threshold = config_key
        # Filter matching results
        pts = [
            r for r in results
            if r["dataset"] == dataset_id
            and r["use_ep"] == use_ep
            and r["replicas"] == replicas
            and r["backend"] == backend
            and r["threshold"] == threshold
        ]
        if not pts:
            continue
        # Sort by batch size
        pts.sort(key=lambda r: r["batch"])
        x = [r["throughput"] for r in pts]
        y = [r[metric] for r in pts]
        batches = [r["batch"] for r in pts]

        ax.plot(x, y,
                label=style["label"],
                color=style["color"],
                marker=style["marker"],
                markersize=8,
                linewidth=2)

        # Annotate batch sizes
        for xi, yi, b in zip(x, y, batches):
            ax.annotate(f"B={b}",
                        (xi, yi),
                        textcoords="offset points",
                        xytext=(5, 5),
                        fontsize=7,
                        color=style["color"])

    dataset_name = DATASET_NAMES.get(dataset_id, f"Dataset {dataset_id}")
    ax.set_xlabel("Total Token Throughput (tok/s)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"{dataset_name}: Throughput vs {ylabel}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = OUTPUT_DIR / filename
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)


def main():
    print("Loading results...")
    results = load_results()
    results = filter_model(results, "Qwen3")
    print(f"Found {len(results)} Qwen3 results")

    if not results:
        print("No results found!")
        sys.exit(1)

    # Find which datasets are available
    datasets = sorted(set(r["dataset"] for r in results))
    print(f"Datasets: {datasets}")

    for ds in [0, 2]:
        if ds not in datasets:
            print(f"  Dataset {ds} not found, skipping")
            continue
        ds_name = DATASET_NAMES.get(ds, str(ds)).lower()
        plot_dataset(results, ds, "p99_tpot",
                     "P99 TPOT (ms)",
                     f"throughput_vs_p99tpot_{ds_name}.png")
        plot_dataset(results, ds, "p99_ttft",
                     "P99 TTFT (ms)",
                     f"throughput_vs_p99ttft_{ds_name}.png")

    print("Done!")


if __name__ == "__main__":
    main()
