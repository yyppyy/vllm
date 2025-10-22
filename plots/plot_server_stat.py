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
SERVER_NAME_RE = re.compile(
    r"^server_(?P<num_gpus>\d+)_(?P<ep_degree>\d+)_(?P<num_replicas>\d+)_(?P<batch_size>\d+)_0_0\.log$"
)

AVG_LINE_RE = re.compile(
    r"\[AVG last\s+(?P<x>\d+)\s+iters\]\s+batch size\s+(?P<y>\d+)\s+(?:→|->)\s+(?P<z>\d+(?:\.\d+)?)s\s+\(total\s+(?P<w>\d+(?:\.\d+)?)s\)"
)

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

def load_results(results_dir, filters, file_glob="*.log"):
    """
    Scan results_dir for matching text logs and parse '[AVG last ...]' lines.

    - filename encodes (num_gpus, ep_degree, num_replicas, batch_size)
    - for each requested inner batch size y in `inner_batch_sizes`,
      take the *last* occurrence in the file and store z (avg runtime, seconds)
      into results[(g, ep)][rep][batch_size][y] = z

    `filters` supports the same keys as before to include/exclude by:
      num_gpus, ep_degree, num_replicas, batch_size (each value is a set or None)

    `inner_batch_sizes`: Iterable[int] of y-values to extract from logs.
    """
    # results[(g, ep)][rep][batch_size][y] = z
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    wanted_y = filters["actual_batch_size"]

    for path in Path(results_dir).glob(file_glob):
        m = SERVER_NAME_RE.match(path.name)
        if not m:
            # Try a broader glob (e.g., .txt) if not matched by default
            continue

        num_gpus     = int(m.group("num_gpus"))
        ep_degree    = int(m.group("ep_degree"))
        num_replicas = int(m.group("num_replicas"))
        batch_size   = int(m.group("batch_size"))

        # Apply filters
        if filters.get("num_gpus")      not in (None,) and num_gpus     not in filters["num_gpus"]:      continue
        if filters.get("ep_degree")     not in (None,) and ep_degree    not in filters["ep_degree"]:     continue
        if filters.get("num_replicas")  not in (None,) and num_replicas not in filters["num_replicas"]:  continue
        if filters.get("batch_size")    not in (None,) and batch_size   not in filters["batch_size"]:    continue

        try:
            text = path.read_text(errors="ignore")
        except Exception as e:
            print(f"Failed to read {path}: {e}")
            continue

        # Keep the *last* z for each requested y found in this file
        last_z_by_y = {}
        for mm in AVG_LINE_RE.finditer(text):
            y = int(mm.group("y"))
            if y in wanted_y:
                z = float(mm.group("z"))
                last_z_by_y[y] = z  # overwrite → ends up with the last appearance

        if not last_z_by_y:
            # Nothing matched for the requested inner batch sizes
            continue

        # Store
        dest = results[(num_gpus, ep_degree)][num_replicas][batch_size]
        for y, z in last_z_by_y.items():
            dest[y] = z

    return results

def plot_group(group_key, rep_to_bsdata, outdir, y_cut, gap_ratio=0.04):
    """
    Broken y-axis grouped bar chart:
      - Draw identical bars on BOTH axes.
      - BOTTOM axis shows 0..y_cut (tall bars appear truncated).
      - TOP axis shows values > y_cut (by setting limits above the cut).
    """
    num_gpus, ep_degree = group_key
    if not rep_to_bsdata:
        return

    reps = sorted(rep_to_bsdata.keys())

    # Assert a single, shared outer batch size
    bs_keys_all = {list(rep_to_bsdata[r].keys())[0] for r in reps}
    assert all(len(rep_to_bsdata[r]) == 1 for r in reps), "Expected exactly one batch_size per replica."
    assert len(bs_keys_all) == 1, f"All replicas must share one batch_size; got {bs_keys_all}"
    outer_batch_size = int(next(iter(bs_keys_all)))

    # X ticks: union of inner y's
    all_y = sorted({y for r in reps for y in rep_to_bsdata[r][outer_batch_size].keys()})
    if not all_y:
        return
    x = np.arange(len(all_y), dtype=float)

    # Grouped bar layout
    total_width = 0.8
    n_rep = max(1, len(reps))
    bar_w = total_width / n_rep
    offsets = (-total_width / 2) + (np.arange(n_rep) + 0.5) * bar_w

    # Heights per replica
    heights_by_rep = []
    for r in reps:
        y_to_z = rep_to_bsdata[r][outer_batch_size]
        heights_by_rep.append(np.array([float(y_to_z.get(y, np.nan)) for y in all_y], dtype=float))

    # All numeric values
    all_vals = np.concatenate([h[~np.isnan(h)] for h in heights_by_rep if np.any(~np.isnan(h))], dtype=float)
    if all_vals.size == 0:
        return

    has_top = np.any(all_vals > y_cut)
    if has_top:
        top_min = np.nanmin(all_vals[all_vals > y_cut])
        top_max = np.nanmax(all_vals[all_vals > y_cut])

    # Subplots in (TOP, BOTTOM) order
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, sharex=True, figsize=(5.5, 6),
        gridspec_kw={"height_ratios": [1, 2]}
    )
    fig.subplots_adjust(hspace=0.05)

    # Draw the SAME bars on both axes (no masking)
    for i, r in enumerate(reps):
        h = heights_by_rep[i]
        ax_bottom.bar(x + offsets[i], h, width=bar_w, label=f"NUM_REPLICAS={r}")
        if has_top:
            ax_top.bar(x + offsets[i], h, width=bar_w, label=f"NUM_REPLICAS={r}")

    # Axis limits: bottom clips at y_cut; top shows the high range
    bottom_max_visible = max(y_cut, np.nanmax(np.where(all_vals <= y_cut, all_vals, np.nan)))
    if np.isnan(bottom_max_visible):  # all bars above cut
        bottom_max_visible = y_cut
    ax_bottom.set_ylim(0, bottom_max_visible * 1.05)

    if has_top:
        ax_top.set_ylim(top_min * 0.95, top_max * 1.05)
    else:
        ax_top.set_visible(False)

    # Broken-axis diagonals
    if has_top:
        d = gap_ratio
        kw_top = dict(transform=ax_top.transAxes, color='k', clip_on=False)
        ax_top.plot((-d, +d), (-d, +d), **kw_top)                   # top-left
        ax_top.plot((1 - d, 1 + d), (-d, +d), **kw_top)             # top-right
        kw_bot = dict(transform=ax_bottom.transAxes, color='k', clip_on=False)
        ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kw_bot)          # bottom-left
        ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kw_bot)    # bottom-right

    # Labels & styling
    ax_bottom.set_xlabel("Per-GPU batch size")
    ax_bottom.set_ylabel("batch computation time (s)")
    # fig.suptitle(
    #     f"Avg runtime vs inner batch size y (broken y-axis at {y_cut})\n"
    #     f"NUM_GPUS={num_gpus}, EP_DEGREE={ep_degree}, BATCH_SIZE={outer_batch_size}"
    # )
    ax_bottom.set_xticks(x, all_y)
    ax_bottom.grid(True, axis="y", linestyle="--", alpha=0.4)
    if has_top:
        ax_top.grid(True, axis="y", linestyle="--", alpha=0.4)

    # Single legend (dedup handles)
    handles, labels = ax_bottom.get_legend_handles_labels()
    seen, h_dedup, l_dedup = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l); h_dedup.append(h); l_dedup.append(l)
    (ax_top if has_top else ax_bottom).legend(h_dedup, l_dedup, title="Replicas", frameon=False, loc="best")

    base = Path(outdir) / f"avg_runtime_y_broken_g{num_gpus}_ep{ep_degree}_bs{outer_batch_size}_cut{y_cut}"
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{base}.pdf")
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
    ap.add_argument("--num-replicas", type=str, default="0,32,64",
                    help='Filter NUM_REPLICAS lines to include (e.g. "0,1,2"); empty = all')
    ap.add_argument("--batch-size", type=str, default="16",
                    help='Filter BATCH_SIZE (e.g. "256,512,1024" or "256..4096:256"); empty = all')
    ap.add_argument("--actual-batch-size", type=str, default="16,4096",
                    help='Filter BATCH_SIZE (e.g. "256,512,1024" or "256..4096:256"); empty = all')
    args = ap.parse_args()

    filters = {
        "num_gpus": parse_list(args.num_gpus),
        "ep_degree": parse_list(args.ep_degree),
        "num_replicas": parse_list(args.num_replicas),
        "batch_size": parse_list(args.batch_size),
        "actual_batch_size": parse_list(args.actual_batch_size)
    }

    results = load_results(args.results_dir, filters)
    if not results:
        print("No matching files found.")
        return

    # Now groups are (g, ep) only; each plot shows lines for different replicas
    for group_key, rep_to_bsdata in results.items():
        plot_group(group_key, rep_to_bsdata, args.output_dir, 0.015)

    print(f"Done. Plots written to: {args.output_dir}")

if __name__ == "__main__":
    main()
