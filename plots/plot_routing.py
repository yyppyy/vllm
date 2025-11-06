#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from utils import set_paper_style, get_palette

CSV_PATH = Path("../results") / "routing_solver.csv"
OUT_TIME = Path("routing_solver_time.pdf")
OUT_EXPERTS = Path("routing_solver_experts.pdf")

algo_to_legend = {
    'cpu_max_flow': 'CPU Optimal',
    'gpu_max_flow': 'GPU Optimal',
    'gpu_greedy_lock': 'NAME',
}

def main():
    set_paper_style()
    
    df = pd.read_csv(CSV_PATH)

    # keep one batch size
    df = df[df["batch_per_chip"] == 16].copy()

    selected_algos = ["cpu_max_flow", "gpu_max_flow"]
    df = df[df["algo"].isin(selected_algos)]

    df = df[["algo", "density_factor", "avg_time_ms", "avg_copy_ms", "avg_experts"]]
    df = df.sort_values(["density_factor", "algo"])

    density_vals = df["density_factor"].unique()
    algos = df["algo"].unique()

    x = np.arange(len(density_vals), dtype=float)
    bar_width = 0.85 / max(len(algos), 1)

    # colors for algos (consistent across figures)
    algo_colors = get_palette(len(algo_to_legend), name="tableau10")
    algo_color_map = {algo: algo_colors[i] for i, algo in enumerate(algo_to_legend.keys())}

    # --------------------------------------------------
    # Figure 1: stacked avg_time_ms + avg_copy_ms
    # --------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(4, 4))

    for j, algo in enumerate(algos):
        sub = df[df["algo"] == algo].set_index("density_factor")
        x_pos = x + (j - (len(algos) - 1) / 2.0) * bar_width

        base = sub.loc[density_vals, "avg_time_ms"].values
        base = [b * 1e3 for b in base]  # to us
        extra = sub.loc[density_vals, "avg_copy_ms"].values
        extra = [e * 1e3 for e in extra]  # to us

        # bottom part
        ax1.bar(
            x_pos,
            base,
            bar_width,
            color=algo_color_map[algo],
            edgecolor="black",
            linewidth=1,
            label=algo_to_legend[algo]
        )
        if 'cpu' in algo:
            # top (stacked) part
            ax1.bar(
                x_pos,
                extra,
                bar_width,
                bottom=base,
                color=algo_color_map[algo],
                edgecolor="black",
                linewidth=1,
                alpha=0.45,  # slightly transparent to show it's the copy part
                label='GPU <-> CPU',  # legend per algo, once
            )

    ax1.set_xticks(x)
    ax1.set_xticklabels([str(v) for v in density_vals])
    ax1.set_xlabel("Replication Ratio")
    ax1.set_ylabel("Time (us)")
    # ax1.set_title("Routing solver time breakdown")

    # legend on top
    leg1 = ax1.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.20),
        ncol=len(algos),
        frameon=False,
        # title="algo",
    )
    # leave room for legend
    plt.subplots_adjust(top=0.75)

    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    fig1.tight_layout()
    fig1.savefig(OUT_TIME, format="pdf")
    print(f"saved {OUT_TIME}")

    # --------------------------------------------------
    # Figure 2: avg_experts (grouped bars)
    # --------------------------------------------------
    df = pd.read_csv(CSV_PATH)

    # keep one batch size
    df = df[df["batch_per_chip"] == 16].copy()

    selected_algos = ["cpu_max_flow", "gpu_greedy_lock"]
    df = df[df["algo"].isin(selected_algos)]

    df = df[["algo", "density_factor", "avg_time_ms", "avg_copy_ms", "avg_experts"]]
    df = df.sort_values(["density_factor", "algo"])

    density_vals = df["density_factor"].unique()
    algos = df["algo"].unique()

    fig2, ax2 = plt.subplots(figsize=(4, 4))

    for j, algo in enumerate(algos):
        sub = df[df["algo"] == algo].set_index("density_factor")
        x_pos = x + (j - (len(algos) - 1) / 2.0) * bar_width

        vals = sub.loc[density_vals, "avg_experts"].values

        ax2.bar(
            x_pos,
            vals,
            bar_width,
            color=algo_color_map[algo],
            edgecolor="black",
            linewidth=1,
            label=algo_to_legend[algo],
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels([str(v) for v in density_vals])
    ax2.set_xlabel("Replication Ratio")
    ax2.set_ylabel("Activated Experts")
    # ax2.set_title("Experts activated per density")

    leg2 = ax2.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.20),
        ncol=len(algos),
        frameon=False,
        # title="algo",
    )
    plt.subplots_adjust(top=0.78)

    ax2.grid(axis="y", linestyle="--", alpha=0.35)
    fig2.tight_layout()
    fig2.savefig(OUT_EXPERTS, format="pdf")
    print(f"saved {OUT_EXPERTS}")


if __name__ == "__main__":
    main()
