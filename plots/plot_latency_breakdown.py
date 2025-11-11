#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from utils import (
    set_paper_style,
    get_palette,
    HATCHES
)

num_layers = 48

# ---------------------------------------------
# config
# ---------------------------------------------
CSV_PATH = Path("../results/latency_breakdown.csv")
OUT_PATH = Path("latency_breakdown.pdf")

components = ["topk", "routing_lock", "all2all", "ffn", "attention"]
legend_components = ["Top-k", "Routing", "All2All / AllGather", "FFN", "Attention"]

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

    fig, ax = plt.subplots(figsize=(9, 3.5))

    # y positions for each replication group
    y = np.arange(len(replications), dtype=float)

    for j, rid in enumerate(routing_ids):
        # shift for grouped bars, but vertically now
        y_pos = y + (j - (len(routing_ids) - 1) / 2.0) * bar_width

        sub = df[df["routing_id"] == rid].set_index("replication_id")
        left = np.zeros(len(replications), dtype=float)

        for comp, lg in zip(components, legend_components):
            vals = [val * 1e6 / num_layers for val in sub.loc[replications, comp].values]

            bar_container = ax.barh(
                y_pos,
                vals,
                bar_width,
                left=left,
                color=comp_color_map[comp],
                edgecolor="black",
                linewidth=1,
                hatch=HATCHES[rid%len(HATCHES)],
                label=lg if j == 0 else None,  # components in legend only once
            )
            
            # annotate each segment
            for k, v in enumerate(vals):
                if v == 0:
                    continue
                x_text = left[k] + v / 2.0     # middle of this stacked segment
                y_text = y_pos[k]
                ax.text(
                    x_text,
                    y_text,
                    f"{v:.0f}",                # format however you like
                    va="center",
                    ha="center",
                    fontsize=10,
                    fontweight='bold'
                )
            
            left += vals

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
            hatch=HATCHES[rid%len(HATCHES)],
            linewidth=1,
        )
        routing_handles.append(patch)
        routing_labels.append("vLLM-EPLB" if rid == 0 else "vLLM-NAME")

    # place legends above
    leg1 = ax.legend(
        comp_handles,
        comp_labels,
        ncols=5,
        loc="upper center",
        bbox_to_anchor=(0.32, 1.13),
        frameon=False,
    )
    ax.add_artist(leg1)
    ax.legend(
        routing_handles,
        routing_labels,
        ncols=2,
        loc="upper center",
        bbox_to_anchor=(0.85, 1.13),
        frameon=False,
    )

    # y ticks correspond to replications
    ax.set_yticks(y)
    ax.set_yticklabels([str(r) for r in replications])
    ax.set_ylabel("Replication Ratio")

    ax.set_xlabel("Time (us)")

    # ax.grid(axis="x", linestyle="--", alpha=0.4)
    # ax.margins(y=0.03)

    plt.subplots_adjust(top=0.9, bottom=0.15, left=0.1, right=0.99)  # leave space for legends
    # fig.tight_layout()
    fig.savefig(OUT_PATH, format="pdf")
    print(f"saved to {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
