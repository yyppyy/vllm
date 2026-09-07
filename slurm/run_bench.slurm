#!/bin/bash
# Generic runner:  sbatch slurm/run_bench.slurm <bench_script.sh> <13 positional args...>
#   e.g. sbatch slurm/run_bench.slurm bench_serve.sh 8 8 1 64 32 2 dispatch_combine 0 \
#            likaixin/InstructCoder 0 256 Qwen3-30B-A3B-8-128 1
#SBATCH --job-name=metro-bench
#SBATCH --partition=gpu-wentzlaff
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --gres=gpu:8
#SBATCH --mem=900G
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/bench-%j.out
#SBATCH --error=slurm/logs/bench-%j.out

set -uo pipefail
cd /scratch/gpfs/WENTZLAF/hm2595/metro_moe/vllm
module purge; module load cudatoolkit/12.8
export PATH="$HOME/.local/bin:$PATH"
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
# Compute nodes have no route to huggingface.co: run fully offline
# against the pre-fetched ./models and ./datasets trees.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false

SCRIPT="$1"; shift
echo "=== $SCRIPT $* ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
bash "./$SCRIPT" "$@"
echo "=== BENCH EXIT $? ==="
