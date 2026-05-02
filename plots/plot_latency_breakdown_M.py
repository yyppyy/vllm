#!/usr/bin/env python3
"""Per-(rank, layer, batch) MoE latency breakdown — mean across the
selected M (per-replay batch) range, grouped by replication ratio with
EP and METRO bars side by side.

Input: `server_breakdown.log` files emitted by `bench_breakdown.sh`.
Each line has the form

    Breakdown seq=N rank=R layer=L M=M attention_ns=X gating_ns=Y
              routing_ns=Z dispatch_ns=A expert_ns=B combine_ns=C

For each (model, dataset) pair the script writes one figure that
mirrors `plots/latency_breakdown.pdf`:

  * y-axis = replication ratio (1.0 + NUM_REPLICAS / num_experts);
  * for every replication ratio, **EP** and **METRO** stacked bars
    are drawn side by side (hatch distinguishes them, color marks
    category);
  * EP vs METRO is decided by the run's `MEM_BOUND_ROUTING_THRES`
    (10th positional arg / `_threshold` in the dirname): `>0` -> METRO,
    `0` -> EP. Same convention as `plot_throughput_latency.py`.

Records from every batch-size config that share the same
(num_replicas, system) are merged together.
"""
import argparse
import gzip
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from style import (HATCHES, STANDARD_PANEL_HEIGHT, STANDARD_PANEL_WIDTH,
                   apply_style, palette, save_fig, style_axes)

apply_style()

RESULTS_DIR = Path("results/vllm_results_final")
OUTPUT_DIR = Path("plots")

DATASET_NAMES = {
    0: "InstructCoder",
    1: "Edit_5k_char",
    2: "ShareGPT",
}

# Models with no `-{topk}-{num_experts}` suffix in the model name need
# a hardcoded num_experts so we can compute replication *ratio*.
KNOWN_NUM_EXPERTS = {
    "Qwen3-30B-A3B": 128,
    "ERNIE-4.5-21B-A3B-PT": 64,
}

CATEGORIES = [
    "attention", "gating", "routing", "dispatch", "expert", "combine",
]
LEGEND_LABELS = [
    "Attention", "Gating", "Routing", "Dispatch", "Expert", "Combine",
]

# System order maps to HATCHES[idx]: '' for vllm-EP, '///' for
# vllm-METRO. These strings are used both as legend labels and as
# bucket keys (returned by `system_label`).
SYSTEMS = ["vllm-EP", "vllm-METRO"]

BREAKDOWN_RE = re.compile(
    r"Breakdown seq=(?P<seq>-?\d+) rank=(?P<rank>-?\d+) "
    r"layer=(?P<layer>-?\d+) M=(?P<M>\d+) "
    r"attention_ns=(?P<attention>\d+) gating_ns=(?P<gating>\d+) "
    r"routing_ns=(?P<routing>\d+) dispatch_ns=(?P<dispatch>\d+) "
    r"expert_ns=(?P<expert>\d+) combine_ns=(?P<combine>\d+)"
)


def parse_dirname(dirname: str) -> dict | None:
    """Same `RUN_HASH` schema as `bench_breakdown.sh`. Group layout
    matches `plot_throughput_latency.py.parse_dirname`."""
    for backend in ("allgather_reducescatter", "dispatch_combine"):
        m = re.match(
            r"^(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_"
            + re.escape(backend)
            + r"_(\d+)_(\d+)_(\d+)_(.+?)(?:_g(\d+))?$",
            dirname,
        )
        if m:
            return {
                "use_ep":       int(m.group(3)),
                "num_replicas": int(m.group(4)),
                "batch":        int(m.group(5)),
                "routing":      int(m.group(6)),
                "backend":      backend,
                "dataset":      int(m.group(7)),
                "threshold":    int(m.group(9)),
                "model":        m.group(10),
            }
    return None


def model_num_experts(model_name: str) -> int | None:
    """Extract num_experts from `...-{topk}-{num_experts}`; otherwise
    fall back to `KNOWN_NUM_EXPERTS`."""
    m = re.search(r"-(\d+)-(\d+)$", model_name or "")
    if m:
        return int(m.group(2))
    return KNOWN_NUM_EXPERTS.get(model_name)


def system_label(cfg: dict) -> str:
    """Match `plot_throughput_latency.py`: `threshold > 0` is METRO,
    everything else is plain EP. Returned strings double as legend
    labels — keep them in sync with `SYSTEMS`."""
    return "vllm-METRO" if cfg["threshold"] > 0 else "vllm-EP"


def replication_ratio(num_replicas: int, num_experts: int | None
                       ) -> float | None:
    """`(num_experts + num_replicas) / num_experts`. Returns `None`
    when `num_experts` is unknown."""
    if num_experts is None or num_experts <= 0:
        return None
    return (num_experts + num_replicas) / num_experts


def _read_text(path: Path) -> str:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", errors="replace") as f:
            return f.read()
    return path.read_text(errors="replace")


def parse_breakdown_log(path: Path):
    try:
        text = _read_text(path)
    except OSError:
        return
    for m in BREAKDOWN_RE.finditer(text):
        yield {
            "M": int(m["M"]),
            "rank": int(m["rank"]),
            "layer": int(m["layer"]),
            "attention": int(m["attention"]),
            "gating": int(m["gating"]),
            "routing": int(m["routing"]),
            "dispatch": int(m["dispatch"]),
            "expert": int(m["expert"]),
            "combine": int(m["combine"]),
        }


def aggregate_means(records, m_lo: int, m_hi: int):
    """Merge every record in [m_lo, m_hi] into a single per-category
    mean (us)."""
    sums = {c: 0.0 for c in CATEGORIES}
    n = 0
    for r in records:
        if not (m_lo <= r["M"] <= m_hi):
            continue
        n += 1
        for c in CATEGORIES:
            sums[c] += r[c]
    if n == 0:
        return None, 0
    return {c: sums[c] / n / 1e3 for c in CATEGORIES}, n


def _ratio_label(ratio: float) -> str:
    """`1.5x`-style label, integer-clean when the ratio is whole.

    Uses 6 decimal places before stripping trailing zeros so clean
    rationals like 1.125 (= 1 + 16/128) and 1.375 (= 1 + 48/128) are
    rendered exactly instead of being rounded to 1.13 / 1.38.
    """
    if abs(ratio - round(ratio)) < 1e-6:
        return f"{int(round(ratio))}.0x"
    s = f"{ratio:.6f}".rstrip("0").rstrip(".")
    return f"{s}x"


def plot_breakdown(per_bucket, model, dataset, out_dir,
                   x_max: float | None = None):
    """One figure per (model, dataset). Bars are placed at integer
    y positions per replication ratio; within each ratio EP and METRO
    sit side by side, colored by category and hatched by system.

    `per_bucket` is `{(ratio, system): {category: mean_us}}`. When
    `x_max` is None the axis auto-sizes to ~1.18× the widest bar."""
    ratios = sorted({r for (r, _) in per_bucket.keys()})
    if not ratios:
        return None

    # 2x the standard panel width; ~15 % taller than STANDARD to
    # leave room for the two stacked legends without crowding the
    # bars.
    fig, ax = plt.subplots(
        figsize=(2 * STANDARD_PANEL_WIDTH, 1.15 * STANDARD_PANEL_HEIGHT))
    # Standard left-gutter y-axis label (needs ~0.12 fraction of the
    # 2x-wide figure width to fit the rotated text + tick labels);
    # two-line legend stack snug against the axes top.
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.18, top=0.84)

    colors = palette(len(CATEGORIES), name="tableau10")
    color_map = {c: colors[i] for i, c in enumerate(CATEGORIES)}

    y = np.arange(len(ratios), dtype=float)
    # Tighter bars leave a vertical gap between adjacent rows so we
    # can route leader-line annotations for small segments through it.
    bar_height = 0.30 if len(SYSTEMS) == 2 else 0.5

    bar_totals: dict[tuple[float, str], float] = {}
    for ratio in ratios:
        for sysname in SYSTEMS:
            vals = per_bucket.get((ratio, sysname), {})
            bar_totals[(ratio, sysname)] = sum(vals.values())
    max_total = max(bar_totals.values()) if bar_totals else 1.0

    for j, sysname in enumerate(SYSTEMS):
        # Stagger systems vertically around each y-tick: EP below
        # (j=0 -> -0.5*bar_height), METRO above (j=1 -> +0.5*bar_height).
        y_offset = (j - (len(SYSTEMS) - 1) / 2.0) * bar_height
        y_pos = y + y_offset

        present_mask = np.array(
            [(ratio, sysname) in per_bucket for ratio in ratios])
        left = np.zeros(len(ratios), dtype=float)
        for c, lg in zip(CATEGORIES, LEGEND_LABELS):
            vals = np.array(
                [per_bucket.get((ratio, sysname), {}).get(c, 0.0)
                 for ratio in ratios],
                dtype=float)
            ax.barh(
                y_pos, vals, bar_height, left=left,
                color=color_map[c], edgecolor="black", linewidth=0.6,
                hatch=HATCHES[j % len(HATCHES)],
                label=lg if j == 0 else None,
            )
            # Always draw the in-place label for any non-zero
            # segment, even when the segment is narrower than the
            # text. Overflow into adjacent segments is acceptable —
            # the segment color anchors which value belongs to which
            # category visually.
            for k, v in enumerate(vals):
                if not present_mask[k] or v <= 0:
                    continue
                ax.text(left[k] + v / 2.0, y_pos[k], f"{v:.0f}",
                        va="center", ha="center",
                        fontsize=plt.rcParams["legend.fontsize"] - 1,
                        color="white", fontweight="bold")
            left += vals

        # Total at right end.
        for k, ratio in enumerate(ratios):
            if not present_mask[k]:
                continue
            total = bar_totals[(ratio, sysname)]
            ax.text(total + max_total * 0.005, y_pos[k],
                    f"{total:.0f}",
                    va="center", ha="left",
                    fontsize=plt.rcParams["legend.fontsize"] - 1,
                    fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels([_ratio_label(r) for r in ratios])
    # Extend y_lim slightly so the inline y-axis label has clear
    # space above the topmost bar.
    ax.set_ylim(-0.5, len(ratios) - 0.4)

    x_axis_max = x_max if x_max is not None else max_total * 1.18
    style_axes(ax,
               x_label="Mean per-layer latency (us)",
               y_label="Replication Ratio",
               x_lim=(0.0, x_axis_max))

    # Two legends stacked above the axes: categories (color) on the
    # top row, systems (hatch) on the bottom row. Side-by-side on the
    # same row is too wide for a 6-category figure and the EP/METRO
    # box would overlap the categories on the right.
    cat_handles, cat_labels = ax.get_legend_handles_labels()
    sys_handles = [
        plt.Rectangle((0, 0), 1, 1,
                       facecolor="white", edgecolor="black",
                       hatch=HATCHES[i % len(HATCHES)], linewidth=1)
        for i in range(len(SYSTEMS))
    ]
    # Push the legend stack up against the figure top edge (figure
    # top sits at ~y_axes=1.24 with the current subplots_adjust),
    # which moves the white space from above the legends into the
    # band between the bottom legend and the axes top.
    leg1 = ax.legend(cat_handles, cat_labels,
                     ncols=len(CATEGORIES),
                     loc="upper center",
                     bbox_to_anchor=(0.5, 1.22),
                     frameon=False)
    ax.add_artist(leg1)
    ax.legend(sys_handles, SYSTEMS,
              ncols=len(SYSTEMS),
              loc="upper center",
              bbox_to_anchor=(0.5, 1.12),
              frameon=False)

    ds_name = DATASET_NAMES.get(dataset, f"dataset{dataset}")
    safe_model = model.replace("/", "_")
    out = out_dir / f"latency_breakdown_M_{safe_model}_{ds_name}.pdf"
    save_fig(fig, out)
    plt.close(fig)
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                   help="Directory of `RUN_HASH` subdirs containing "
                        "`server_breakdown.log` (or .log.gz).")
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--m-range", type=str, default="24-32",
                   help="Inclusive M range, e.g. '24-32' (default).")
    p.add_argument("--min-records", type=int, default=10,
                   help="Drop (replication, system) buckets with "
                        "fewer than this many records (default: 10).")
    p.add_argument("--x-max", type=float, default=700.0,
                   help="Upper bound of the x-axis in microseconds "
                        "(default: 700). Set to 0 to auto-scale.")
    args = p.parse_args()

    try:
        m_lo, m_hi = (int(s) for s in args.m_range.split("-"))
    except ValueError:
        print(f"--m-range must be 'lo-hi' (got {args.m_range!r})",
              file=sys.stderr)
        return 2
    if m_lo > m_hi:
        m_lo, m_hi = m_hi, m_lo

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Three-level grouping:
    #   key1 = (model, dataset) -> figure
    #   key2 = (replication_ratio, system)  -> bar
    #   value = list of records (any batch-size config)
    grouped: dict[tuple[str, int],
                  dict[tuple[float, str], list[dict]]] = defaultdict(
                      lambda: defaultdict(list))
    seen = 0
    skipped_unknown_model: set[str] = set()
    for d in sorted(args.results_dir.iterdir()):
        if not d.is_dir():
            continue
        cfg = parse_dirname(d.name)
        if cfg is None:
            continue
        log = d / "server_breakdown.log"
        if not log.exists():
            log = d / "server_breakdown.log.gz"
            if not log.exists():
                continue
        ne = model_num_experts(cfg["model"])
        if ne is None:
            skipped_unknown_model.add(cfg["model"])
            continue
        ratio = replication_ratio(cfg["num_replicas"], ne)
        sysname = system_label(cfg)
        seen += 1
        key = (cfg["model"], cfg["dataset"])
        for rec in parse_breakdown_log(log):
            grouped[key][(ratio, sysname)].append(rec)
    if not grouped:
        print(f"No Breakdown records under {args.results_dir}",
              file=sys.stderr)
        if skipped_unknown_model:
            print(f"  (skipped models with unknown num_experts: "
                  f"{sorted(skipped_unknown_model)}; add to "
                  f"KNOWN_NUM_EXPERTS or include the topk/experts "
                  f"suffix in the model name)", file=sys.stderr)
        return 1
    print(f"Parsed {seen} log files into {len(grouped)} "
          f"(model, dataset) groups; M range [{m_lo}, {m_hi}]; "
          f"all batch sizes merged")

    for (model, dataset), buckets in sorted(grouped.items()):
        per_bucket = {}
        bucket_counts = {}
        for k, recs in buckets.items():
            means, n = aggregate_means(recs, m_lo, m_hi)
            if means is not None and n >= args.min_records:
                per_bucket[k] = means
                bucket_counts[k] = n
        if not per_bucket:
            print(f"  {model} / "
                  f"{DATASET_NAMES.get(dataset, dataset)}: "
                  f"no bucket with >= {args.min_records} records "
                  f"in M=[{m_lo}, {m_hi}]")
            continue
        x_max = args.x_max if args.x_max > 0 else None
        out = plot_breakdown(per_bucket, model, dataset,
                              args.output_dir, x_max=x_max)
        ds_name = DATASET_NAMES.get(dataset, f"dataset{dataset}")
        summary = ", ".join(
            f"{_ratio_label(r)} {s}={bucket_counts[(r, s)]}"
            for (r, s) in sorted(per_bucket.keys()))
        print(f"  {model} / {ds_name}: [{summary}] -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
