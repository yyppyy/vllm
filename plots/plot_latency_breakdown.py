#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from utils import (
    set_paper_style,
    get_palette,
)

# ---------------------------------------------
# config
# ---------------------------------------------
CSV_PATH = Path("../results/latency_breakdown.csv")
OUT_PATH = Path("latency_breakdown.pdf")

components = ["topk", "routing", "all2all", "ffn", "attention"]

# hatches to distinguish routing_id
ROUTING_HATCHES = {
    0: "",
    1: "//",
    2: "xx",   # in case you add more later
}

def main():
    # make it pretty
    set_paper_style()

    df = pd.read_csv(CSV_PATH)
    # keep ordering nice
    df = df.sort_values(["replication_id", "routing_id"])

    replications = df["replication_id"].unique()
    routing_ids = sorted(df["routing_id"].unique())

    # colors: one per component, consistent
    comp_colors = get_palette(len(components), name="tableau10")
    comp_color_map = {comp: comp_colors[i] for i, comp in enumerate(components)}

    x = np.arange(len(replications), dtype=float)
    bar_width = 0.38 if len(routing_ids) == 2 else 0.8 / max(len(routing_ids), 1)

    fig, ax = plt.subplots(figsize=(3.5, 4))

    for j, rid in enumerate(routing_ids):
        # shift for grouped bars
        x_pos = x + (j - (len(routing_ids) - 1) / 2.0) * bar_width

        sub = df[df["routing_id"] == rid].set_index("replication_id")
        bottom = np.zeros(len(replications), dtype=float)

        for comp in components:
            vals = [val * 1e3 for val in sub.loc[replications, comp].values]
            ax.bar(
                x_pos,
                vals,
                bar_width,
                bottom=bottom,
                color=comp_color_map[comp],
                edgecolor="black",
                linewidth=1,
                hatch=ROUTING_HATCHES.get(rid, ""),
                label=comp if j == 0 else None,  # components in legend only once
            )
            bottom += vals

    # legends
    # component legend (colors)
    comp_handles, comp_labels = ax.get_legend_handles_labels()

    # routing legend (hatches)
    routing_handles = []
    routing_labels = []
    for rid in routing_ids:
        patch = plt.Rectangle(
            (0, 0), 1, 1,
            facecolor="white",
            edgecolor="black",
            hatch=ROUTING_HATCHES.get(rid, ""),
            linewidth=1,
        )
        routing_handles.append(patch)
        routing_labels.append("vLLM-EPLB" if rid == 0 else 'vLLM-NAME')

    # place legends to the right
    leg1 = ax.legend(
        comp_handles,
        comp_labels,
        # title="Component",
        # loc="upper left",
        ncols=3,
        loc="upper center",
        bbox_to_anchor=(0.45, 1.34),
        frameon=False,
    )
    ax.add_artist(leg1)
    ax.legend(
        routing_handles,
        routing_labels,
        # title="Series",
        # loc="lower left",
        ncols=2,
        loc="upper center",
        bbox_to_anchor=(0.34, 1.15),
        frameon=False,
    )
    
    # plt.subplots_adjust(top=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in replications])
    ax.set_xlabel("Replication Ratio")
    ax.set_ylabel("time (ms)")
    # ax.set_title("Per-replication stacked breakdown")

    ax.grid(axis="y")
    ax.margins(x=0.03)

    fig.tight_layout()
    plt.subplots_adjust(top=0.65)  # leave space at top
    fig.savefig(OUT_PATH, format="pdf")
    print(f"saved to {OUT_PATH.resolve()}")

if __name__ == "__main__":
    main()
