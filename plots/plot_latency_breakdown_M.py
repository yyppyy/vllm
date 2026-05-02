#!/usr/bin/env python3
"""Per-(rank, layer, batch) MoE latency breakdown — mean across the
selected M (per-replay batch) range.

Input: `server_breakdown.log` files emitted by `bench_breakdown.sh`.
Each line has the form

    Breakdown seq=N rank=R layer=L M=M attention_ns=X gating_ns=Y
              routing_ns=Z dispatch_ns=A expert_ns=B combine_ns=C

For each (model, dataset) pair, the script averages the six categories
across every record whose `M` falls in `--m-range` (default 24..32
inclusive) and draws one horizontal stacked bar per M, mirroring
`plots/latency_breakdown.pdf` but using the project-standard `style.py`
helpers.
"""
import argparse
import gzip
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from style import (apply_style, paper_figure, palette, save_fig,
                   style_axes)

apply_style()

RESULTS_DIR = Path("results/vllm_results_final")
OUTPUT_DIR = Path("plots")

DATASET_NAMES = {
    0: "InstructCoder",
    1: "Edit_5k_char",
    2: "ShareGPT",
}

CATEGORIES = [
    "attention", "gating", "routing", "dispatch", "expert", "combine",
]
LEGEND_LABELS = [
    "Attention", "Gating", "Routing", "Dispatch", "Expert", "Combine",
]

BREAKDOWN_RE = re.compile(
    r"Breakdown seq=(?P<seq>-?\d+) rank=(?P<rank>-?\d+) "
    r"layer=(?P<layer>-?\d+) M=(?P<M>\d+) "
    r"attention_ns=(?P<attention>\d+) gating_ns=(?P<gating>\d+) "
    r"routing_ns=(?P<routing>\d+) dispatch_ns=(?P<dispatch>\d+) "
    r"expert_ns=(?P<expert>\d+) combine_ns=(?P<combine>\d+)"
)


def parse_dirname(dirname: str) -> dict | None:
    """Same `RUN_HASH` schema as `bench_breakdown.sh` /
    `bench_tok_cnt.sh`. Returns `None` for unrelated directories."""
    for backend in ("allgather_reducescatter", "dispatch_combine"):
        m = re.match(
            r"^(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_"
            + re.escape(backend)
            + r"_(\d+)_(\d+)_(\d+)_(.+?)(?:_g(\d+))?$",
            dirname,
        )
        if m:
            return {
                "model": m.group(10),
                "dataset": int(m.group(7)),
                "batch": int(m.group(5)),
            }
    return None


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
    """Return ({M: {category: mean_us}}, {M: count}) for every M in
    [m_lo, m_hi] that has at least one record."""
    sums: dict[int, dict[str, float]] = defaultdict(
        lambda: {c: 0.0 for c in CATEGORIES})
    counts: dict[int, int] = defaultdict(int)
    for r in records:
        M = r["M"]
        if not (m_lo <= M <= m_hi):
            continue
        counts[M] += 1
        for c in CATEGORIES:
            sums[M][c] += r[c]
    means = {}
    for M, n in counts.items():
        means[M] = {c: sums[M][c] / n / 1e3 for c in CATEGORIES}  # us
    return means, dict(counts)


def plot_breakdown(means, model, dataset, out_dir):
    """One horizontal stacked bar per M; one segment per category."""
    Ms = sorted(means.keys())
    if not Ms:
        return None

    fig, ax = paper_figure()
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.20, top=0.85)

    colors = palette(len(CATEGORIES), name="tableau10")
    color_map = {c: colors[i] for i, c in enumerate(CATEGORIES)}

    y = np.arange(len(Ms), dtype=float)
    bar_height = 0.6

    # Annotation rule: only label segments that occupy at least 6% of
    # the widest bar so the slim ones (gating / routing) don't get a
    # number jammed inside.
    bar_totals = [sum(means[M].values()) for M in Ms]
    max_total = max(bar_totals) if bar_totals else 1.0
    label_min = max_total * 0.06

    left = np.zeros(len(Ms), dtype=float)
    for c, lg in zip(CATEGORIES, LEGEND_LABELS):
        vals = np.array([means[M][c] for M in Ms], dtype=float)
        ax.barh(y, vals, bar_height, left=left,
                color=color_map[c], edgecolor="black", linewidth=0.6,
                label=lg)
        for k, v in enumerate(vals):
            if v < label_min:
                continue
            ax.text(left[k] + v / 2.0, y[k], f"{v:.0f}",
                    va="center", ha="center",
                    fontsize=plt.rcParams["legend.fontsize"] - 1,
                    color="white", fontweight="bold")
        left += vals

    # Total at the right end of each bar.
    for k, total in enumerate(bar_totals):
        ax.text(total + max_total * 0.01, y[k],
                f"{total:.0f}",
                va="center", ha="left",
                fontsize=plt.rcParams["legend.fontsize"] - 1,
                fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels([str(M) for M in Ms])
    style_axes(ax,
               x_label="Mean per-layer latency (us)",
               y_label="M (per-replay batch)",
               x_lim=(0.0, max_total * 1.18))

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.20),
              ncol=3, frameon=False)

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
    p.add_argument("--log", type=Path, default=None,
                   help="Optional: a single breakdown log file. When "
                        "set, --results-dir is ignored and the figure "
                        "is named 'latency_breakdown_M_<stem>.pdf'.")
    p.add_argument("--min-records", type=int, default=10,
                   help="Drop M values with fewer than this many "
                        "records (default: 10).")
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

    if args.log is not None:
        records = list(parse_breakdown_log(args.log))
        if not records:
            print(f"No Breakdown records in {args.log}", file=sys.stderr)
            return 1
        means, counts = aggregate_means(records, m_lo, m_hi)
        means = {M: v for M, v in means.items()
                 if counts[M] >= args.min_records}
        if not means:
            print(f"No M in [{m_lo}, {m_hi}] with "
                  f">= {args.min_records} records in {args.log}",
                  file=sys.stderr)
            return 1
        stem = args.log.name.replace(".log.gz", "").replace(".log", "")
        out = args.output_dir / f"latency_breakdown_M_{stem}.pdf"
        # Reuse plot_breakdown by giving it placeholder model/dataset
        # and then renaming the file.
        tmp = plot_breakdown(means, model=stem, dataset=-1,
                             out_dir=args.output_dir)
        if tmp is not None and tmp != out:
            tmp.rename(out)
        if out.exists():
            print(f"wrote {out}  (M={sorted(means.keys())}, "
                  f"records={[counts[M] for M in sorted(means)]})")
        return 0

    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    seen = 0
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
        seen += 1
        key = (cfg["model"], cfg["dataset"])
        for rec in parse_breakdown_log(log):
            grouped[key].append(rec)
    if not grouped:
        print(f"No Breakdown records under {args.results_dir}",
              file=sys.stderr)
        return 1
    print(f"Parsed {seen} log files into {len(grouped)} "
          f"(model, dataset) groups; M range [{m_lo}, {m_hi}]")

    for (model, dataset), records in sorted(grouped.items()):
        means, counts = aggregate_means(records, m_lo, m_hi)
        means = {M: v for M, v in means.items()
                 if counts[M] >= args.min_records}
        if not means:
            print(f"  {model} / "
                  f"{DATASET_NAMES.get(dataset, dataset)}: "
                  f"no M in range with >= {args.min_records} records")
            continue
        out = plot_breakdown(means, model, dataset, args.output_dir)
        ds_name = DATASET_NAMES.get(dataset, f"dataset{dataset}")
        Ms = sorted(means.keys())
        n_each = [counts[M] for M in Ms]
        print(f"  {model} / {ds_name}: "
              f"M={Ms}, records={n_each} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
