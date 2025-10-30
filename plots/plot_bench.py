#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from collections import defaultdict
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt

# Filename pattern: bench_result_${NUM_GPUS}_${EP_DEGREE}_${NUM_REPLICAS}_${BATCH_SIZE}.json
def get_re_by_dataset_id(dataset_id):
    # Ensure the dataset_id is an integer
    assert isinstance(dataset_id, int), "dataset_id must be an integer"
    return re.compile(
        rf"^bench_result_(?P<num_gpus>\d+)_(?P<ep_degree>\d+)_(?P<num_replicas>\d+)_(?P<batch_size>\d+)_(?P<mem_bound_routing>\d+)_({dataset_id})\.json$"
    )

METRICS = [
    "output_throughput",
    "mean_ttft_ms", "p95_ttft_ms", "p99_ttft_ms", "p10_ttft_ms",
    "mean_itl_ms", "p95_itl_ms", "p99_itl_ms",
    "mean_tpot_ms", "p95_tpot_ms", "p99_tpot_ms", "p10_tpot_ms"
]

dataset_id2name = {
    0 : 'likaixin/InstructCoder', # code humaneval
    1 : 'AI-MO/NuminaMath-1.5', # math gsm8k
    2 : 'Aeala/ShareGPT_Vicuna_unfiltered', # chat gpqa
}

def metric_to_ylabel(metric):
    if 'ms' in metric:
        return ' '.join(metric.split('_')[:-1]) + ' (ms)'
    elif metric == 'output_throughput':
        return 'Throughput (tokens/s)'
    else:
        raise RuntimeError('unsupported metric')

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

def load_results(results_dir, filters, dataset_id):
    """
    Scan results_dir for matching files and load JSON.

    Group by (num_gpus, ep_degree), and within each group
    store a mapping: num_replicas -> {batch_size -> data}.
    """
    # results[(g, ep)][rep][batch_size] = data
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    missing_metrics = set()

    for path in Path(results_dir).glob("bench_result_*.json"):
        m = get_re_by_dataset_id(dataset_id).match(path.name)
        if not m:
            continue

        num_gpus = int(m.group("num_gpus"))
        ep_degree = int(m.group("ep_degree"))
        num_replicas = int(m.group("num_replicas"))
        batch_size = int(m.group("batch_size"))
        mem_bound_routing_enabled = int(m.group("mem_bound_routing"))

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

        results[(num_gpus, ep_degree)][num_replicas][batch_size][mem_bound_routing_enabled] = data

    if missing_metrics:
        print("Warning: some files lacked metrics:", ", ".join(sorted(missing_metrics)))
        
    return results

def plot_group(group_key, rep_to_bsdata, outdir, dataset_name):
    """
    For a specific (num_gpus, ep_degree) group, plot each metric vs BATCH_SIZE.
    Each NUM_REPLICAS value becomes a separate bar within each BATCH_SIZE group.
    Saves both PNG and PDF.
    """
    num_gpus, ep_degree = group_key
    if not rep_to_bsdata:
        return

    # All batch sizes across replicas (union)
    all_batch_sizes = sorted({bs for d in rep_to_bsdata.values() for bs in d.keys()})
    if not all_batch_sizes:
        return

    reps = sorted(rep_to_bsdata.keys())
    x = np.arange(len(all_batch_sizes), dtype=float)

    total_width = 0.8
    n_rep = max(1, len(reps))
    bar_w = total_width / n_rep
    # center bars around tick
    offsets = (-total_width / 2) + (np.arange(n_rep) + 0.5) * bar_w

    for metric in METRICS:
        plt.figure(figsize=(6, 5))

        heights_mem_bound = []
        heights_eplb = []
        x = []

        for i, rep in enumerate(reps):
            bs_to_data = rep_to_bsdata.get(rep, {})
            # heights = []
            # print(bs_to_data.values())
            for bs in all_batch_sizes:
                for routing in bs_to_data[bs].keys():
                    v = bs_to_data.get(bs, {}).get(routing, {}).get(metric, float("nan"))
                    if routing:
                        heights_mem_bound.append(v)
                    if ((not routing) or rep == 0) and not (not routing and rep == 0):
                        heights_eplb.append(v)
            # heights = np.array(heights, dtype=float)
            x.append(rep)
            # Draw bars; NaNs will be skipped by matplotlib
            # plt.bar(x + offsets[i], heights, width=bar_w, label=f"NUM_REPLICAS={rep}")
        print(x, heights_mem_bound, heights_mem_bound)
        plt.plot(x, heights_mem_bound, label=f"mem-bound routing")
        plt.plot(x, heights_eplb, label=f"eplb routing")

        # plt.xlabel("BATCH_SIZE")
        plt.ylabel(metric_to_ylabel(metric))
        plt.ylim(0, max(heights_eplb) * 1.2)
        plt.title(dataset_name)
        plt.xticks(x, x, rotation=0)
        plt.xlabel('# Replicated Replicate Experts (128 Total)')
        plt.grid(True, axis="y", linestyle="--", alpha=0.4)
        plt.legend(frameon=False)
        plt.tight_layout()

        base = Path(outdir) / f"{metric}_g{num_gpus}_ep{ep_degree}"
        base.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{base}.pdf")
        plt.close()

def main():
    ap = argparse.ArgumentParser(description="Plot vLLM benchmark JSONs vs BATCH_SIZE, one line per NUM_REPLICAS.")
    ap.add_argument("--results-dir", type=str, default='../results',
                    help="Directory containing bench_result_*.json files")
    ap.add_argument("--output-dir", type=str, default=".",
                    help="Where to save figures (default: plots)")
    ap.add_argument("--num-gpus", type=str, default="8",
                    help='Filter NUM_GPUS (e.g. "2,4" or "2..8:2"); empty = all')
    ap.add_argument("--ep-degree", type=str, default="8",
                    help='Filter EP_DEGREE (e.g. "1,2,4"); empty = all')
    ap.add_argument("--num-replicas", type=str, default="0,16,32,48,64",
                    help='Filter NUM_REPLICAS lines to include (e.g. "0,1,2"); empty = all')
    ap.add_argument("--batch-size", type=str, default="16",
                    help='Filter BATCH_SIZE (e.g. "256,512,1024" or "256..4096:256"); empty = all')
    ap.add_argument("--dataset-id", type=str, default="0,1,2",
                    help='dataset ids')
    args = ap.parse_args()

    filters = {
        "num_gpus": parse_list(args.num_gpus),
        "ep_degree": parse_list(args.ep_degree),
        "num_replicas": parse_list(args.num_replicas),
        "batch_size": parse_list(args.batch_size),
        "dataset_id": parse_list(args.dataset_id),
    }

    for id in filters["dataset_id"]:
        results = load_results(args.results_dir, filters, id)
        if not results:
            print("No matching files found.")
            return

        # Now groups are (g, ep) only; each plot shows lines for different replicas
        for group_key, rep_to_bsdata in results.items():
            plot_group(group_key, rep_to_bsdata, args.output_dir, dataset_id2name[id])

    print(f"Done. Plots written to: {args.output_dir}")

if __name__ == "__main__":
    main()
