#!/bin/bash
# Variance study: repeat the same bench_serve config N times per arm and
# archive each run separately (bench_serve.sh otherwise overwrites in place).
#SBATCH --job-name=mrepeat
#SBATCH --partition=gpu-wentzlaff
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --gres=gpu:8
#SBATCH --mem=900G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/repeat-%j.out
#SBATCH --error=slurm/logs/repeat-%j.out

set -uo pipefail
cd /scratch/gpfs/WENTZLAF/hm2595/metro_moe/vllm
module purge; module load cudatoolkit/12.8
export PATH="$HOME/.local/bin:$PATH"
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false

MODEL=Qwen3-30B-A3B-8-128
ARGS_PRE="8 8 1 64 32 2 dispatch_combine 0 likaixin/InstructCoder 0"
REPS=3
OUT=results/repeat
mkdir -p "$OUT"

nvidia-smi --query-gpu=index,name --format=csv,noheader
echo "=== node $(hostname) | $REPS reps per arm ==="

for i in $(seq 1 $REPS); do
  for THRES in 0 256; do
    TAG=$([ "$THRES" = "0" ] && echo EPLB || echo METRO)
    HASH="8_8_1_64_32_2_dispatch_combine_0_0_${THRES}_${MODEL}_g1"
    DEST="$OUT/${TAG}_run${i}"
    if [ -d "$DEST" ]; then echo ">>> skip $TAG run$i (exists)"; continue; fi
    echo ">>> $TAG run $i/$REPS  $(date +%H:%M:%S)"
    bash ./bench_serve.sh $ARGS_PRE $THRES $MODEL 1
    if [ -d "results/$HASH" ]; then
      mkdir -p "$DEST"; cp results/"$HASH"/bench_result*.json "$DEST"/ 2>/dev/null
      echo ">>> archived -> $DEST"
    else
      echo ">>> WARNING: results/$HASH missing after $TAG run $i"
    fi
  done
done
echo "=== REPEAT DONE ==="
