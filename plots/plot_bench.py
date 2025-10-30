#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from collections import defaultdict
from pathlib import Path
import numpy as np
from utils import *

import matplotlib.pyplot as plt

# Filename pattern: bench_result_${NUM_GPUS}_${EP_DEGREE}_${NUM_REPLICAS}_${BATCH_SIZE}.json
def get_re_by_dataset_id_batch_size(dataset_id, batch_size):
    # Ensure the dataset_id is an integer
    assert isinstance(dataset_id, int), "dataset_id must be an integer"
    return re.compile(
        rf"^bench_result_(?P<num_gpus>\d+)_(?P<ep_degree>\d+)_(?P<num_replicas>\d+)_({batch_size})_(?P<routing_id>\d+)_({dataset_id})\.json$"
    )

METRICS = [
    "total_token_throughput",
    "mean_ttft_ms",
    # "p95_ttft_ms", "p99_ttft_ms", "p10_ttft_ms",
    # "mean_itl_ms", "p95_itl_ms", "p99_itl_ms",
    "mean_tpot_ms",
    # "p95_tpot_ms", "p99_tpot_ms", "p10_tpot_ms"
]

dataset_id2name = {
    0 : 'likaixin/InstructCoder', # code humaneval
    1 : 'AI-MO/NuminaMath-1.5', # math gsm8k
    2 : 'Aeala/ShareGPT_Vicuna_unfiltered', # chat gpqa
}

routing_id2name = {
    1 : 'Mem. Bound Aware', # code humaneval
    0 : 'EPLB', # math gsm8k
}

def metric_to_ylabel(metric):
    if 'ms' in metric:
        res = ' '.join(metric.split('_')[:-1]) + ' (ms)'
        if 'mean' in res:
            res = res.replace('mean', 'Mean')
        if 'ttft' in res:
            res = res.replace('ttft', 'TTFT')
        elif 'tpot' in res:
            res = res.replace('tpot', 'TPOT')
        return res
    elif metric == 'total_token_throughput':
        return 'Throughput (Tokens/s)'
    else:
        raise RuntimeError('unsupported metric')

def metric_to_title(metric):
    if 'ttft' in metric:
        return 'Prefill Latency'
    elif 'tpot' in metric:
        return 'Decode Latency'
    elif metric == 'total_token_throughput':
        return 'Total Throughput'
    else:
        return ''

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

    for batch_size in filters["batch_size"]:
        for path in Path(results_dir).glob("bench_result_*.json"):
            m = get_re_by_dataset_id_batch_size(dataset_id, batch_size).match(path.name)
            if not m:
                continue

            num_gpus = int(m.group("num_gpus"))
            ep_degree = int(m.group("ep_degree"))
            num_replicas = int(m.group("num_replicas"))
            # batch_size = int(m.group("batch_size"))
            routing_id = int(m.group("routing_id"))

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

            results[(num_gpus, ep_degree)][num_replicas][batch_size][routing_id] = data

    if missing_metrics:
        print("Warning: some files lacked metrics:", ", ".join(sorted(missing_metrics)))
        
    return results

def plot_group(group_key, rep_to_bsdata, outdir, dataset_name, routing_ids):
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

    # Paper style + consistent colors for all lines in these figures
    set_paper_style()
    apply_color_cycle(len(all_batch_sizes) * len(routing_ids), "tableau10")

    for metric in METRICS:
        fig = plt.figure(figsize=(3.5, 3.5))
        ax = plt.gca()

        # clean axes
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)

        # integer x axis from your `reps`
        x_vals = np.array(sorted(reps), dtype=float)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xticks(x_vals)
        ax.set_xticklabels([str(int(xx)) for xx in x_vals])

        max_h = 0.0
        series_idx = 0

        for batch_size in all_batch_sizes:
            for routing_id in routing_ids:
                # collect y values across reps
                heights = []
                for rep in x_vals.astype(int):
                    bs_to_data = rep_to_bsdata.get(rep, {})
                    key_rid = 1 if (rep == 0 and routing_id == 0) else routing_id
                    v = bs_to_data.get(batch_size, {}).get(key_rid, {}).get(metric, float("nan"))
                    heights.append(v)

                arr = np.asarray(heights, dtype=float)
                if np.all(np.isnan(arr)):
                    continue

                marker = MARKERS[series_idx % len(MARKERS)]
                series_idx += 1
                ax.plot(
                    x_vals, arr,
                    marker=marker, linewidth=2.2, markersize=5.5,
                    label=f"{routing_id2name[routing_id]}, batch={batch_size}",
                )
                if np.any(np.isfinite(arr)):
                    max_h = max(max_h, np.nanmax(arr))

        ax.set_ylabel(metric_to_ylabel(metric))
        ax.set_xlabel("# Replicated Experts (128 Total)")
        ax.set_title(metric_to_title(metric))
        if metric == 'total_token_throughput':
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
        if max_h > 0:
            ax.set_ylim(0, max_h * 1.15)

        # if series_idx > 0:
        #     ax.legend(frameon=False, ncol=2, handlelength=2.2, columnspacing=1.0)

        fig.tight_layout()

        bs_tag = ",".join(map(str, sorted(all_batch_sizes)))
        base = Path(outdir) / f"{metric}_g{num_gpus}_ep{ep_degree}_bs{bs_tag}_{dataset_name.replace('/', '_')}"
        base.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(f"{base}.pdf", transparent=True)
        # fig.savefig(f"{base}.png", transparent=True)
        plt.close(fig)

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
    ap.add_argument("--batch-size", type=str, default="32",
                    help='Filter BATCH_SIZE (e.g. "256,512,1024" or "256..4096:256"); empty = all')
    ap.add_argument("--routing-id", type=str, default="0",
                    help='Filter Routing')
    ap.add_argument("--dataset-id", type=str, default="0",
                    help='dataset ids')
    args = ap.parse_args()

    filters = {
        "num_gpus": parse_list(args.num_gpus),
        "ep_degree": parse_list(args.ep_degree),
        "num_replicas": parse_list(args.num_replicas),
        "batch_size": parse_list(args.batch_size),
        "routing_id": parse_list(args.routing_id),
        "dataset_id": parse_list(args.dataset_id),
    }

    for did in filters["dataset_id"]:
        results = load_results(args.results_dir, filters, did)
        if not results:
            print("No matching files found.")
            return

        # Now groups are (g, ep) only; each plot shows lines for different replicas
        for group_key, rep_to_bsdata in results.items():
            plot_group(group_key, rep_to_bsdata, args.output_dir, dataset_id2name[did], filters["routing_id"])

    print(f"Done. Plots written to: {args.output_dir}")

if __name__ == "__main__":
    main()
