# plot_topk_moe_stats_singlefile.py
import argparse
import glob
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple
import itertools

import torch
import matplotlib.pyplot as plt


FNAME_RANK_RE = re.compile(r"\.ep(\d+)\.pt$")


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

def prefix_for_run(results_dir: str, num_gpus: int, ep_degree: int, num_replicas: int, batch_size: int) -> str:
    # Adjust here if your real prefix differs
    return f"{results_dir}/topk_tensor_{num_gpus}_{ep_degree}_{num_replicas}_{batch_size}"


def find_rank_files(prefix: str) -> Dict[int, str]:
    """
    Return {ep_rank: filepath} for all matching .pt files for this run prefix.
    Accepts files like: <prefix>.ep0.pt, <prefix>.ep1.pt, ...
    """
    files = glob.glob(prefix + ".ep*.pt")
    out: Dict[int, str] = {}
    for f in files:
        m = FNAME_RANK_RE.search(f)
        if m:
            rank = int(m.group(1))
            out[rank] = f
    return dict(sorted(out.items()))


def load_batches_from_rank_file(path: str) -> List[torch.Tensor]:
    """
    File contains a list of tensors (one per batch). Each tensor is [M, topk] of expert IDs.
    """
    data = torch.load(path, map_location="cpu")
    # fixme. dummy fix to jump over dummy tokens and tokens before first rebalance
    # print(len(data))
    while torch.equal(data[0][0], data[0][1]):
        data = data[1:]
    # print(len(data))
    size_dict = defaultdict(int)
    for tpk in data:
        size_dict[tpk.shape[0]] += 1
    print(size_dict)
    # Be tolerant if someone saved a single tensor (older dump): wrap it
    # due to vLLM dummy all-gather + reduce-scatter MoE all2all impl, each rank holds
    # the global (all-GPU) topk logits. So technically we only need data in rank 0
    if isinstance(data, torch.Tensor):
        return [data]
    return data


def estimate_stride(per_rank_batches: List[List[torch.Tensor]], ep_degree: int, num_replicas: int) -> int:
    """
    Estimate experts_per_rank (stride) under linear mapping after replication:
      ep_rank = (expert_id // stride) % ep_degree
      replica = (expert_id // stride) // ep_degree
    """
    max_id = -1
    for batches in per_rank_batches:
        for t in batches:
            if t is None or t.numel() == 0:
                continue
            mx = int(t.max().item())
            if mx > max_id:
                max_id = mx
    if max_id < 0:
        return 1
    total_experts_est = max_id + 1
    denom = max(1, ep_degree * num_replicas)
    stride = (total_experts_est + denom - 1) // denom
    return max(1, stride)


def ep_of_id(expert_id: int, stride: int, ep_degree: int) -> int:
    return (expert_id // stride) % ep_degree


def replica_of_id(expert_id: int, stride: int, ep_degree: int) -> int:
    return (expert_id // stride) // ep_degree


def compute_batch_metrics(
    per_rank_batches: List[List[torch.Tensor]],
    ep_degree: int,
    num_replicas: int,
    stride: int,
) -> Tuple[float, float]:
    """
    Returns:
      (avg_max_activated_replicas_over_batches,
       avg_max_tokens_on_a_rank_over_batches)
    """
    if not per_rank_batches:
        return float("nan"), float("nan")
    num_batches = min(len(b) for b in per_rank_batches)
    if num_batches == 0:
        return float("nan"), float("nan")

    max_activated_replicas_vals: List[int] = []
    max_tokens_on_rank_vals: List[int] = []

    for bidx in range(num_batches):
        # Collect this batch across ranks
        batch_ts = [per_rank_batches[r][bidx] for r in range(len(per_rank_batches))]

        replicas_per_rank: List[int] = []
        tokens_per_rank: List[int] = []

        for r, t in enumerate(batch_ts):
            ids = t.view(-1).tolist() if t is not None and t.numel() else []
            replica_set = set()
            token_count_on_rank = 0
            for eid in ids:
                if ep_of_id(eid, stride, ep_degree) == r:
                    token_count_on_rank += 1
                    replica_set.add(replica_of_id(eid, stride, ep_degree))
            replicas_per_rank.append(len(replica_set))
            tokens_per_rank.append(token_count_on_rank)

        max_activated_replicas_vals.append(max(replicas_per_rank))
        max_tokens_on_rank_vals.append(max(tokens_per_rank))

    avg_max_reps = sum(max_activated_replicas_vals) / len(max_activated_replicas_vals)
    avg_max_tokens = sum(max_tokens_on_rank_vals) / len(max_tokens_on_rank_vals)
    return avg_max_reps, avg_max_tokens


def main():
    ap = argparse.ArgumentParser(description="Plot MoE stats from single-file-per-rank .pt dumps")
    ap = argparse.ArgumentParser(description="Plot vLLM benchmark JSONs vs BATCH_SIZE.")
    ap.add_argument("--results-dir", type=str, required=True,
                    help="Directory containing bench_result_*.json files")
    ap.add_argument("--output-dir", type=str, default="plots",
                    help="Where to save figures (default: plots)")
    ap.add_argument("--num-gpus", type=str, default="",
                    help='Filter NUM_GPUS (e.g. "2,4" or "2..8:2"); empty = all')
    ap.add_argument("--ep-degree", type=str, default="",
                    help='Filter EP_DEGREE (e.g. "1,2,4"); empty = all')
    ap.add_argument("--num-replicas", type=str, default="",
                    help='Filter NUM_REPLICAS (e.g. "0,1,2"); empty = all')
    ap.add_argument("--batch-size", type=str, default="",
                    help='Filter BATCH_SIZE (e.g. "256,512,1024" or "256..4096:256"); empty = all')
    args = ap.parse_args()

    filters = {
        "num_gpus": parse_list(args.num_gpus),
        "ep_degree": parse_list(args.ep_degree),
        "num_replicas": parse_list(args.num_replicas),
        "batch_size": parse_list(args.batch_size),
    }

    # groups[(NUM_GPUS, EP_DEGREE)][NUM_REPLICAS][BATCH_SIZE] = (avg_max_activated_replicas, avg_max_tokens)
    groups: Dict[Tuple[int, int], Dict[int, Dict[int, Tuple[float, float]]]] = defaultdict(lambda: defaultdict(dict))
    
    for ng, ep, reps in itertools.product(filters["num_gpus"], filters["ep_degree"], filters["num_replicas"]):

        for bs in filters["batch_size"]:
            prefix = prefix_for_run(args.results_dir, ng, ep, reps, bs)
            rank_files = {0: find_rank_files(prefix)[0]}
            if not rank_files:
                continue

            # Load all ranks present
            per_rank_batches: List[List[torch.Tensor]] = []
            # Build in order of ep_rank (keys already sorted)
            for _, path in rank_files.items():
                per_rank_batches.append(load_batches_from_rank_file(path))

            # Estimate mapping stride and compute metrics
            stride = estimate_stride(per_rank_batches, ep_degree=ep, num_replicas=reps)
            avg_max_replica, avg_max_tokens = compute_batch_metrics(per_rank_batches, ep, reps, stride)
            groups[(ng, ep)][reps][bs] = (avg_max_replica, avg_max_tokens)

    # Plot per (NUM_GPUS, EP_DEGREE) group
    for (ng, ep) in sorted(groups.keys()):
        replica_values = sorted(groups[(ng, ep)].keys())
        all_bs = sorted({b for r in replica_values for b in groups[(ng, ep)][r].keys()})

        # Plot 1: avg max activated replicas vs BATCH_SIZE
        plt.figure()
        for r in replica_values:
            x, y = [], []
            for bs in all_bs:
                if bs in groups[(ng, ep)][r]:
                    x.append(bs)
                    y.append(groups[(ng, ep)][r][bs][0])
            if x:
                plt.plot(x, y, marker="o", label=f"NUM_REPLICAS={r}")
        plt.xlabel("BATCH_SIZE")
        plt.ylabel("Avg. max activated replicas (per-batch, max over EP ranks)")
        plt.title(f"Activated replicas vs BATCH_SIZE (NUM_GPUS={ng}, EP_DEGREE={ep})")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        out1 = os.path.join(args.output_dir, f"activated_replicas_NUMGPUS{ng}_EP{ep}.png")
        plt.savefig(out1, bbox_inches="tight")
        plt.close()

        # Plot 2: avg max tokens on a rank vs BATCH_SIZE
        plt.figure()
        for r in replica_values:
            x, y = [], []
            for bs in all_bs:
                if bs in groups[(ng, ep)][r]:
                    x.append(bs)
                    y.append(groups[(ng, ep)][r][bs][1])
            if x:
                plt.plot(x, y, marker="o", label=f"NUM_REPLICAS={r}")
        plt.xlabel("BATCH_SIZE")
        plt.ylabel("Avg. max tokens on a rank (per-batch)")
        plt.title(f"Max tokens per rank vs BATCH_SIZE (NUM_GPUS={ng}, EP_DEGREE={ep})")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        out2 = os.path.join(args.output_dir, f"max_tokens_NUMGPUS{ng}_EP{ep}.png")
        plt.savefig(out2, bbox_inches="tight")
        plt.close()

        print(f"Wrote:\n  {out1}\n  {out2}")


if __name__ == "__main__":
    main()
