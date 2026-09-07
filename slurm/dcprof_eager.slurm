#!/bin/bash
#SBATCH --job-name=repro
#SBATCH --partition=gpu-wentzlaff
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=56 --gres=gpu:8 --mem=900G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/dceager-%j.out
#SBATCH --error=slurm/logs/dceager-%j.out
set -uo pipefail
cd /scratch/gpfs/WENTZLAF/hm2595/metro_moe/vllm
module purge; module load cudatoolkit/12.8
export PATH="$HOME/.local/bin:$PATH"
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false
export VLLM_DC_PROFILE=50          # bench_serve.sh does not unset this
OUT=results/dceager; mkdir -p "$OUT"
# eager 模式: DC printer 是 host 侧的, CUDA graph replay 时 python 不执行。
# 用独立副本 bench_serve_eager.sh, 不修改 bench_serve.sh (否则会污染后续 job)。
for TH in 0 256; do
  for SPEC in "64 Qwen3-30B-A3B-8-128" "32 ERNIE-4.5-21B-A3B-PT"; do
    set -- $SPEC; REP=$1; MODEL=$2
    HASH="8_8_1_${REP}_32_2_dispatch_combine_0_0_${TH}_${MODEL}_g1"
    [ -d "$OUT/$HASH" ] && { echo ">>> skip $HASH"; continue; }
    echo ">>> eager DC_PROFILE: $MODEL rep=$REP thres=$TH  $(date +%H:%M:%S)"
    rm -rf "results/$HASH"
    bash ./bench_serve_eager.sh 8 8 1 $REP 32 2 dispatch_combine 0 likaixin/InstructCoder 0 $TH $MODEL 1
    [ -d "results/$HASH" ] && mv "results/$HASH" "$OUT/$HASH"
  done
done
echo "=== DC EAGER DONE ==="
