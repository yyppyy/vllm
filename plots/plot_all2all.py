#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path

from style import (apply_style, paper_figure, save_fig, palette,
                   style_axes, style_legend)

CSV_PATH = Path("../results/all2all.csv")
OUT_PATH = Path("all2all_8gpus.pdf")

def main():
    apply_style()

    df = pd.read_csv(CSV_PATH)

    # only GPUs = 8
    df = df[df["GPUs"] == 8].copy()

    # keep relevant columns
    df = df[["batch", "allgather", "runtime(ms)"]]

    # sort to get stable order
    df = df.sort_values(["batch", "allgather"])

    batches = df["batch"].unique()
    allgathers = sorted(df["allgather"].unique())

    x = np.arange(len(batches), dtype=float)
    bar_width = 0.4 if len(allgathers) == 2 else 0.8 / max(len(allgathers), 1)

    # colors per allgather
    colors = palette(len(allgathers), name="okabe_ito")
    allgather_color = {ag: colors[i] for i, ag in enumerate(allgathers)}

    fig, ax = paper_figure(width="single", height=2.6)

    for j, ag in enumerate(allgathers):
        sub = df[df["allgather"] == ag].set_index("batch")
        x_pos = x + (j - (len(allgathers) - 1) / 2.0) * bar_width
        vals = sub.loc[batches, "runtime(ms)"].values
        vals = [v * 1e3 for v in vals]  # to us

        ax.bar(
            x_pos,
            vals,
            bar_width,
            label=f"allgather" if ag else "all2all",
            color=allgather_color[ag],
            edgecolor="black",
            linewidth=1,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in batches])
    style_axes(ax, x_label="Batch Size", y_label="Time (us)",
               y_zero=True)

    style_legend(ax, loc="upper center",
                 bbox_to_anchor=(0.5, 1.15),
                 ncol=len(allgathers))

    save_fig(fig, OUT_PATH)
    print(f"saved to {OUT_PATH.resolve()}")

if __name__ == "__main__":
    main()
