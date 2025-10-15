#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt

# Filename pattern: bench_result_${NUM_GPUS}_${EP_DEGREE}_${NUM_REPLICAS}_${BATCH_SIZE}.json
FILENAME_RE = re.compile(
    r"^bench_result_(?P<num_gpus>\d+)_(?P<ep_degree>\d+)_(?P<num_replicas>\d+)_(?P<batch_size>\d+)\.json$"
)

METRICS = [
    "total_token_throughput",
    "mean_ttft_ms", "p95_ttft_ms", "p99_ttft_ms",
    "mean_itl_ms", "p95_itl_ms", "p99_itl_ms",
    "mean_tpot__ms", "p95_tpot__ms", "p99_tpot_ms",
]

def parse_list(arg):
    """
    Accepts values like:
      - 2,4,8
      - 32..256:32  (start..end:step)
      - empty string => no filter
    Returns a set of ints or None (no filter).
    """
    if arg is None or arg.strip() == "":
        return None

    arg = arg.strip()
    if ".." in arg:
        # range form: start..end[:step]
        parts = arg.split(":")
        rng = parts[0]
        step = int(parts[1]) if len(parts) > 1 else 1
        start, end = [int(x) for x in rng.split("..")]
        return set(range(start, end + 1, step))
    else:
        return set(int(x) for x in arg.split(","))

def load_results(results_dir, filters):
    """
    Scan results_dir for matching files and load JSON.

    Group by (num_gpus, ep_degree), and within each group
    store a mapping: num_replicas -> {batch_size -> data}.
    """
    # results[(g, ep)][rep][batch_size] = data
    results = defaultdict(lambda: defaultdict(dict))
    missing_metrics = set()

    for path in Path(results_dir).glob("bench_result_*.json"):
        m = FILENAME_RE.match(path.name)
        if not m:
            continue

        num_gpus = int(m.group("num_gpus"))
        ep_degree = int(m.group("ep_degree"))
        num_replicas = int(m.group("num_replicas"))
        batch_size = int(m.group("batch_size"))

        # Apply filters
        if filters["num_gpus"] is not None and num_gpus not in filters["num_gpus"]:
            continue
        if filters["ep_degree"] is not None and ep_degree not in filters["ep_degree"]:
            continue
        if filters["num_replicas"] is not None and num_replicas not in filters["num_replicas"]:
            continue
        if filters["batch_size"] is not None and batch_size not in filters["batch_size"]:
            continue

        try:
            data = json.loads(path.read_text())
        except Exception as e:
            print(f"Failed to read {path}: {e}")
            continue

        for metric in METRICS:
            if metric not in data:
                missing_metrics.add(metric)

        results[(num_gpus, ep_degree)][num_replicas][batch_size] = data

    if missing_metrics:
        print("Warning: some files lacked metrics:", ", ".join(sorted(missing_metrics)))
    return results

def plot_group(group_key, rep_to_bsdata, outdir):
    """
    For a specific (num_gpus, ep_degree) group, plot each metric vs BATCH_SIZE.
    Each NUM_REPLICAS value becomes a separate line.
    """
    num_gpus, ep_degree = group_key

    # Skip empty groups
    if not rep_to_bsdata:
        return

    # One plot per metric
    for metric in METRICS:
        plt.figure()

        # Plot one line per num_replicas
        for rep, bs_to_data in sorted(rep_to_bsdata.items()):
            if not bs_to_data:
                continue
            batch_sizes = sorted(bs_to_data.keys())
            y = [bs_to_data[bs].get(metric, float("nan")) for bs in batch_sizes]
            plt.plot(batch_sizes, y, marker="o", label=f"NUM_REPLICAS={rep}")

        plt.xlabel("BATCH_SIZE")
        plt.ylabel(metric)
        plt.title(
            f"{metric} vs BATCH_SIZE\n"
            f"NUM_GPUS={num_gpus}, EP_DEGREE={ep_degree}"
        )
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend(title="Replica lines", frameon=False)

        outpath = Path(outdir) / f"{metric}_g{num_gpus}_ep{ep_degree}.png"
        outpath.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(outpath, dpi=150)
        plt.close()

def main():
    ap = argparse.ArgumentParser(description="Plot vLLM benchmark JSONs vs BATCH_SIZE, one line per NUM_REPLICAS.")
    ap.add_argument("--results-dir", type=str, required=True,
                    help="Directory containing bench_result_*.json files")
    ap.add_argument("--output-dir", type=str, default="plots",
                    help="Where to save figures (default: plots)")
    ap.add_argument("--num-gpus", type=str, default="",
                    help='Filter NUM_GPUS (e.g. "2,4" or "2..8:2"); empty = all')
    ap.add_argument("--ep-degree", type=str, default="",
                    help='Filter EP_DEGREE (e.g. "1,2,4"); empty = all')
    ap.add_argument("--num-replicas", type=str, default="",
                    help='Filter NUM_REPLICAS lines to include (e.g. "0,1,2"); empty = all')
    ap.add_argument("--batch-size", type=str, default="",
                    help='Filter BATCH_SIZE (e.g. "256,512,1024" or "256..4096:256"); empty = all')
    args = ap.parse_args()

    filters = {
        "num_gpus": parse_list(args.num_gpus),
        "ep_degree": parse_list(args.ep_degree),
        "num_replicas": parse_list(args.num_replicas),
        "batch_size": parse_list(args.batch_size),
    }

    results = load_results(args.results_dir, filters)
    if not results:
        print("No matching files found.")
        return

    # Now groups are (g, ep) only; each plot shows lines for different replicas
    for group_key, rep_to_bsdata in results.items():
        plot_group(group_key, rep_to_bsdata, args.output_dir)

    print(f"Done. Plots written to: {args.output_dir}")

if __name__ == "__main__":
    main()
