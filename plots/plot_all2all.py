#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from utils import set_paper_style, get_palette

CSV_PATH = Path("../results/all2all.csv")
OUT_PATH = Path("all2all_8gpus.pdf")

def main():
    set_paper_style()

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
    colors = get_palette(len(allgathers), name="okabe_ito")
    allgather_color = {ag: colors[i] for i, ag in enumerate(allgathers)}

    fig, ax = plt.subplots(figsize=(3.5, 4))

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
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Time (us)")
    # ax.set_title("Allgather runtime (GPUs=8)")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # put legend on top
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=len(allgathers),
        frameon=False,
    )
    plt.subplots_adjust(top=0.78)

    fig.tight_layout()
    fig.savefig(OUT_PATH, format="pdf")
    print(f"saved to {OUT_PATH.resolve()}")

if __name__ == "__main__":
    main()
