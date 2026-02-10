#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
from pathlib import Path
import re

from utils import set_paper_style, get_palette

# Directories containing the results
EXPERT_PARALLEL_DIR = Path("../results/vllm_results_likaixin_InstructCoder1/results")
TENSOR_PARALLEL_DIR = Path("../results/vllm_results_likaixin_InstructCoder_tp/results")
OUTPUT_FILE = Path(__file__).parent / "ep_vs_tp.pdf"

# File pattern: bench_result_{num_gpu}_{parallel_degree}_{redundant_experts}_{batch_size_per_gpu}_{routing_algo_id}_{dataset_id}.json
# Filter criteria
NUM_GPU = 8
PARALLEL_DEGREE = 8
REDUNDANT_EXPERTS = 0
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]
ROUTING_ALGO_ID = 0
DATASET_ID = 0

def parse_filename(filename):
    """Parse the filename to extract parameters."""
    pattern = r'bench_result_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)\.json'
    match = re.match(pattern, filename)
    if match:
        return {
            'num_gpu': int(match.group(1)),
            'parallel_degree': int(match.group(2)),
            'redundant_experts': int(match.group(3)),
            'batch_size_per_gpu': int(match.group(4)),
            'routing_algo_id': int(match.group(5)),
            'dataset_id': int(match.group(6)),
        }
    return None

def load_data(directory):
    """Load data from JSON files matching the criteria."""
    data_points = []

    for json_file in directory.glob("bench_result_*.json"):
        params = parse_filename(json_file.name)

        # Check if file matches our criteria
        if params and \
           params['num_gpu'] == NUM_GPU and \
           params['parallel_degree'] == PARALLEL_DEGREE and \
           params['redundant_experts'] == REDUNDANT_EXPERTS and \
           params['batch_size_per_gpu'] in BATCH_SIZES and \
           params['routing_algo_id'] == ROUTING_ALGO_ID and \
           params['dataset_id'] == DATASET_ID:

            # Load the JSON file
            with open(json_file, 'r') as f:
                data = json.load(f)

            data_points.append({
                'batch_size': params['batch_size_per_gpu'],
                'mean_tpot_ms': data['mean_tpot_ms'],
                'mean_ttft_ms': data['mean_ttft_ms'],
                'total_token_throughput': data['total_token_throughput'],
            })

    # Sort by batch size for consistent line plotting
    data_points.sort(key=lambda x: x['batch_size'])
    return data_points

def main():
    set_paper_style()

    # Load data from both directories
    ep_data = load_data(EXPERT_PARALLEL_DIR)
    tp_data = load_data(TENSOR_PARALLEL_DIR)

    print(f"Expert Parallel data points: {len(ep_data)}")
    print(f"Tensor Parallel data points: {len(tp_data)}")

    # Create the plot with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Get colors from palette
    colors = get_palette(2, name="tableau10")

    # ===== Plot 1: TPOT vs Throughput =====
    # Plot Expert Parallel series
    if ep_data:
        ep_throughputs = [d['total_token_throughput'] for d in ep_data]
        ep_tpots = [d['mean_tpot_ms'] for d in ep_data]
        ep_batch_sizes = [d['batch_size'] for d in ep_data]

        ax1.plot(ep_tpots, ep_throughputs, '-o', color=colors[0],
                linewidth=2, markersize=8, label='Expert Parallel')

        # Annotate each point with batch size
        for throughput, tpot, batch_size in zip(ep_throughputs, ep_tpots, ep_batch_sizes):
            ax1.annotate(f'{batch_size}',
                       xy=(tpot, throughput),
                       xytext=(5, 5),
                       textcoords='offset points',
                       fontsize=8,
                       alpha=0.8)

    # Plot Tensor Parallel series
    if tp_data:
        tp_throughputs = [d['total_token_throughput'] for d in tp_data]
        tp_tpots = [d['mean_tpot_ms'] for d in tp_data]
        tp_batch_sizes = [d['batch_size'] for d in tp_data]

        ax1.plot(tp_tpots, tp_throughputs, '-s', color=colors[1],
                linewidth=2, markersize=8, label='Tensor Parallel')

        # Annotate each point with batch size
        for throughput, tpot, batch_size in zip(tp_throughputs, tp_tpots, tp_batch_sizes):
            ax1.annotate(f'{batch_size}',
                       xy=(tpot, throughput),
                       xytext=(5, -10),
                       textcoords='offset points',
                       fontsize=8,
                       alpha=0.8)

    # Configure plot 1
    ax1.set_xlabel('Mean TPOT (ms)')
    ax1.set_ylabel('Total Token Throughput (tokens/s)')
    ax1.legend(loc='best', frameon=False)
    ax1.grid(True, linestyle='--', alpha=0.35)
    ax1.set_title('TPOT vs Throughput')

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
        print(f"  Batch size {d['batch_size']}: throughput={d['total_token_throughput']:.2f}, tpot={d['mean_tpot_ms']:.2f}, ttft={d['mean_ttft_ms']:.2f}")

    print("\nTensor Parallel:")
    for d in tp_data:
        print(f"  Batch size {d['batch_size']}: throughput={d['total_token_throughput']:.2f}, tpot={d['mean_tpot_ms']:.2f}, ttft={d['mean_ttft_ms']:.2f}")

if __name__ == "__main__":
    main()
