#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path

from style import (apply_style, paper_figure, save_fig, palette,
                   HATCHES, style_axes, style_legend, metro_palette)

CSV_PATH = Path("../results") / "routing_solver.csv"
OUT_TIME = Path("routing_solver_time.pdf")
OUT_EXPERTS = Path("routing_solver_experts.pdf")

algo_to_legend = {
    'cpu_max_flow': 'CPU Optimal',
    'gpu_max_flow': 'GPU Optimal',
    'gpu_greedy_lock': 'METRO',
    'eplb': 'EPLB',
}

algo_to_legend2 = {
    'cpu_max_flow': 'Optimal',
    'gpu_max_flow': 'Optimal',
    'gpu_greedy_lock': 'METRO',
    'eplb': 'EPLB',
}

model_to_legend = {
    'qwen.yaml': 'Qwen3-30B',
    'deepseek-v3.yaml': 'DeepSeek-V3',
}

dataset_to_legend = {
    'humaneval': 'HumanEval',
    'gsm8k': 'GSM8K',
}

vllm_eplb_ffn_time_us = [281.3125, 297.7916666666667, 310.7916666666667, 335.7083333333333]

def main():
    apply_style()
    
    df = pd.read_csv(CSV_PATH)

    # keep one batch size
    df = df[df["batch_per_chip"] == 32].copy()

    selected_algos = ["cpu_max_flow", "gpu_max_flow"]
    selected_algos2 = ["eplb", "cpu_max_flow", "gpu_greedy_lock"]
    m = 'qwen.yaml'
    d = 'humaneval'
    df = df[df["algo"].isin(selected_algos) & (df["model_config"] == m) & (df["dataset"] == d)]

    df = df[["algo", "density_factor", "avg_time_ms", "avg_copy_ms", "avg_experts"]]
    df = df.sort_values(["density_factor", "algo"])

    density_vals = df["density_factor"].unique()
    algos = df["algo"].unique()
    
    x = np.arange(len(density_vals), dtype=float)
    bar_width = 0.7 / max(len(algos), 1)

    # METRO is the only warm series; baselines (CPU Optimal, GPU
    # Optimal, EPLB) get cold colors so the eye snaps to METRO at
    # first glance.
    metro_c, cold = metro_palette(len(algo_to_legend) - 1)
    algo_color_map: dict[str, str] = {}
    cold_iter = iter(cold)
    for algo in algo_to_legend:
        if algo == 'gpu_greedy_lock':       # METRO
            algo_color_map[algo] = metro_c
        else:
            algo_color_map[algo] = next(cold_iter)

    # --------------------------------------------------
    # Figure 1: stacked avg_time_ms + avg_copy_ms
    # --------------------------------------------------
    fig1, ax1 = paper_figure(width="single", height=2.4)

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
            hatch=HATCHES[j],
            color=algo_color_map[algo],
            edgecolor="black",
            linewidth=1,
            label=algo_to_legend[algo]
        )
        print(base)
        if 'cpu' in algo:
            # top (stacked) part
            ax1.bar(
                x_pos,
                extra,
                bar_width,
                hatch=HATCHES[2],
                bottom=base,
                color=algo_color_map[algo],
                edgecolor="black",
                linewidth=1,
                alpha=0.45,  # slightly transparent to show it's the copy part
                label='GPU <-> CPU',  # legend per algo, once
            )
            print(extra)

    ax1.bar(
        x,
        vllm_eplb_ffn_time_us,
        bar_width * len(algos),
        facecolor=(0, 0, 1, 0.0),
        edgecolor="black", # keep frame
        linewidth=1.0,
        label="FFN",
    )
    
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{v}x' for v in density_vals], rotation=30)
    style_axes(ax1, x_label="Replication Ratio", y_label="Time (us)")

    style_legend(ax1, loc="upper center",
                 bbox_to_anchor=(0.5, 1.26),
                 ncol=len(algos))
    fig1.subplots_adjust(top=0.85, bottom=0.22, left=0.16, right=0.99)

    save_fig(fig1, OUT_TIME, tight=False)
    print(f"saved {OUT_TIME}")

    # --------------------------------------------------
    # Figure 2: avg_experts (grouped bars)
    # --------------------------------------------------
    df = pd.read_csv(CSV_PATH)

    # keep one batch size
    df = df[df["batch_per_chip"] == 32].copy()

    df = df[df["algo"].isin(selected_algos2)]

    df = df[["algo", "density_factor", "avg_time_ms", "avg_copy_ms", "avg_experts", "model_config", "dataset"]]
    df = df.sort_values(["density_factor", "algo"])

    density_vals = df["density_factor"].unique()
    algos = df["algo"].unique()
    algos = selected_algos2
    bar_width = 0.65 / max(len(algos), 1)

    models = ('deepseek-v3.yaml', 'qwen.yaml')
    datasets = ('humaneval', 'gsm8k')
    mds = [(m, d) for m in models for d in datasets]
    # create subplots with shared y so they all use the same scale
    fig2, ax2s = paper_figure(width="double", n_axes=len(mds),
                               height=2.4, sharey=True)

    # if len(mds) == 1, make ax2s iterable
    if not isinstance(ax2s, (list, np.ndarray)):
        ax2s = [ax2s]

    max_y = 0.0

    for ax2, md in zip(ax2s, mds):
        for j, algo in enumerate(algos):
            sub = (
                df[
                    (df["algo"] == algo)
                    & (df["model_config"] == md[0])
                    & (df["dataset"] == md[1])
                ]
                .set_index("density_factor")
            )

            x_pos = x + (j - (len(algos) - 1) / 2.0) * bar_width
            vals = sub.loc[density_vals, "avg_experts"].values

            ax2.bar(
                x_pos,
                vals,
                bar_width,
                hatch=HATCHES[j%len(HATCHES)],
                color=algo_color_map[algo],
                edgecolor="black",
                linewidth=1,
                label=algo_to_legend2[algo],
            )
            print(vals)

            # track global max for unified y
            if len(vals) > 0:
                max_y = max(max_y, float(np.max(vals)))

        ax2.set_xticks(x)
        ax2.set_xticklabels([f'{v}x' for v in density_vals], rotation=30)
        ax2.grid(axis="y", linestyle="--", alpha=0.35)
        ax2.set_title(f"{model_to_legend[md[0]]}\n{dataset_to_legend[md[1]]}")

    # apply the unified y-limit to all axes
    for i, ax2 in enumerate(ax2s):
        ax2.set_ylim(0, max_y * 1.05)
        if i == 0:
            ax2.set_ylabel("Activated Experts")
        else:
            ax2.set_ylabel("")  # no ylabel on others

    # one shared legend at the top, using handles from the first axis
    handles, labels = ax2s[0].get_legend_handles_labels()
    fig2.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(algos),
        frameon=False,
    )

    # one shared xlabel
    fig2.supxlabel("Replication Ratio")

    # tighten layout, remove horizontal gaps
    fig2.subplots_adjust(top=0.8, bottom=0.2, left=0.07, right=0.995, wspace=0.0)

    save_fig(fig2, OUT_EXPERTS, tight=False)
    print(f"saved {OUT_EXPERTS}")



if __name__ == "__main__":
    main()
