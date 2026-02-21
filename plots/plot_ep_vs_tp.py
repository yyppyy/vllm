#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
from pathlib import Path

from utils import set_paper_style, get_palette

# Base directory containing the results
# Layout: {BASE_DIR}/8_8_{IS_EP}_0_{BATCH}_0_{COMM_BACKEND}_0_0/bench_result.json
BASE_DIR = Path("../results/vllm_results_dev")
OUTPUT_FILE = Path(__file__).parent / "ep_vs_tp.pdf"

NUM_GPU = 8
PARALLEL_DEGREE = 8
REDUNDANT_EXPERTS = 0
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]
ROUTING_ALGO_ID = 0
DATASET_ID = 0


def load_data(is_ep, comm_backend):
    """Load data from the new directory structure."""
    data_points = []

    for batch_size in BATCH_SIZES:
        dir_name = (f"{NUM_GPU}_{PARALLEL_DEGREE}_{is_ep}"
                    f"_{REDUNDANT_EXPERTS}_{batch_size}"
                    f"_{ROUTING_ALGO_ID}_{comm_backend}"
                    f"_{DATASET_ID}_{0}")
        json_file = BASE_DIR / dir_name / "bench_result.json"

        if not json_file.exists():
            print(f"Warning: {json_file} not found, skipping")
            continue

        with open(json_file, 'r') as f:
            data = json.load(f)

        data_points.append({
            'batch_size': batch_size,
            'mean_itl_ms': data['mean_itl_ms'],
            'mean_ttft_ms': data['mean_ttft_ms'],
            'total_token_throughput': data['total_token_throughput'],
        })

    # Sort by batch size for consistent line plotting
    data_points.sort(key=lambda x: x['batch_size'])
    return data_points

def main():
    set_paper_style()

    # Load data from both directories
    ep_data = load_data(is_ep=1, comm_backend="dispatch_combine")
    tp_data = load_data(is_ep=0, comm_backend="allgather_reducescatter")

    print(f"Expert Parallel data points: {len(ep_data)}")
    print(f"Tensor Parallel data points: {len(tp_data)}")

    # Create the plot with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Get colors from palette
    colors = get_palette(2, name="tableau10")

    # ===== Plot 1: ITL vs Throughput =====
    # Plot Expert Parallel series
    if ep_data:
        ep_throughputs = [d['total_token_throughput'] for d in ep_data]
        ep_itls = [d['mean_itl_ms'] for d in ep_data]
        ep_batch_sizes = [d['batch_size'] for d in ep_data]

        ax1.plot(ep_itls, ep_throughputs, '-o', color=colors[0],
                linewidth=2, markersize=8, label='Expert Parallel')

        # Annotate each point with batch size
        for throughput, itl, batch_size in zip(ep_throughputs, ep_itls, ep_batch_sizes):
            ax1.annotate(f'{batch_size}',
                       xy=(itl, throughput),
                       xytext=(5, 5),
                       textcoords='offset points',
                       fontsize=8,
                       alpha=0.8)

    # Plot Tensor Parallel series
    if tp_data:
        tp_throughputs = [d['total_token_throughput'] for d in tp_data]
        tp_itls = [d['mean_itl_ms'] for d in tp_data]
        tp_batch_sizes = [d['batch_size'] for d in tp_data]

        ax1.plot(tp_itls, tp_throughputs, '-s', color=colors[1],
                linewidth=2, markersize=8, label='Tensor Parallel')

        # Annotate each point with batch size
        for throughput, itl, batch_size in zip(tp_throughputs, tp_itls, tp_batch_sizes):
            ax1.annotate(f'{batch_size}',
                       xy=(itl, throughput),
                       xytext=(5, -10),
                       textcoords='offset points',
                       fontsize=8,
                       alpha=0.8)

    # Configure plot 1
    ax1.set_xlabel('Mean ITL (ms)')
    ax1.set_ylabel('Total Token Throughput (tokens/s)')
    ax1.legend(loc='best', frameon=False)
    ax1.grid(True, linestyle='--', alpha=0.35)
    ax1.set_title('ITL vs Throughput')

    # ===== Plot 2: TTFT vs Throughput =====
    # Plot Expert Parallel series
    if ep_data:
        ep_throughputs = [d['total_token_throughput'] for d in ep_data]
        ep_ttfts = [d['mean_ttft_ms'] for d in ep_data]
        ep_batch_sizes = [d['batch_size'] for d in ep_data]

        ax2.plot(ep_ttfts, ep_throughputs, '-o', color=colors[0],
                linewidth=2, markersize=8, label='Expert Parallel')

        # Annotate each point with batch size
        for throughput, ttft, batch_size in zip(ep_throughputs, ep_ttfts, ep_batch_sizes):
            ax2.annotate(f'{batch_size}',
                       xy=(ttft, throughput),
                       xytext=(5, 5),
                       textcoords='offset points',
                       fontsize=8,
                       alpha=0.8)

    # Plot Tensor Parallel series
    if tp_data:
        tp_throughputs = [d['total_token_throughput'] for d in tp_data]
        tp_ttfts = [d['mean_ttft_ms'] for d in tp_data]
        tp_batch_sizes = [d['batch_size'] for d in tp_data]

        ax2.plot(tp_ttfts, tp_throughputs, '-s', color=colors[1],
                linewidth=2, markersize=8, label='Tensor Parallel')

        # Annotate each point with batch size
        for throughput, ttft, batch_size in zip(tp_throughputs, tp_ttfts, tp_batch_sizes):
            ax2.annotate(f'{batch_size}',
                       xy=(ttft, throughput),
                       xytext=(5, -10),
                       textcoords='offset points',
                       fontsize=8,
                       alpha=0.8)

    # Configure plot 2
    ax2.set_xlabel('Mean TTFT (ms)')
    ax2.set_ylabel('Total Token Throughput (tokens/s)')
    ax2.legend(loc='best', frameon=False)
    ax2.grid(True, linestyle='--', alpha=0.35)
    ax2.set_title('TTFT vs Throughput')

    # Adjust layout
    fig.tight_layout()

    # Save the figure
    fig.savefig(OUTPUT_FILE, format='pdf', bbox_inches='tight')
    print(f"Saved plot to {OUTPUT_FILE}")

    # Print summary
    print("\nExpert Parallel:")
    for d in ep_data:
        print(f"  Batch size {d['batch_size']}: throughput={d['total_token_throughput']:.2f}, itl={d['mean_itl_ms']:.2f}, ttft={d['mean_ttft_ms']:.2f}")

    print("\nTensor Parallel:")
    for d in tp_data:
        print(f"  Batch size {d['batch_size']}: throughput={d['total_token_throughput']:.2f}, itl={d['mean_itl_ms']:.2f}, ttft={d['mean_ttft_ms']:.2f}")

if __name__ == "__main__":
    main()
