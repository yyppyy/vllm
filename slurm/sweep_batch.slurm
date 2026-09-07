#!/bin/bash
# Reproduce a batch of configs.  Usage:
#   sbatch slurm/sweep_batch.slurm <list.txt> <out_subdir>
# Each line: <bench_script> <13 positional args>
# The bench scripts always write results/<RUN_HASH>/; we clear that first and
# MOVE it out afterwards, so repeated rounds never overwrite each other.
#SBATCH --job-name=repro
#SBATCH --partition=gpu-wentzlaff
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --gres=gpu:8
#SBATCH --mem=900G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/repro-%j.out
#SBATCH --error=slurm/logs/repro-%j.out

set -uo pipefail
cd /scratch/gpfs/WENTZLAF/hm2595/metro_moe/vllm
module purge; module load cudatoolkit/12.8
export PATH="$HOME/.local/bin:$PATH"
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false
[ "${DC_PROFILE:-0}" = "1" ] && export VLLM_DC_PROFILE=1

LIST="$1"; OUTDIR="results/${2:-repro}"
mkdir -p "$OUTDIR"
TOTAL=$(grep -cve '^\s*$' "$LIST")
echo "=== $(hostname) | $TOTAL configs | $LIST -> $OUTDIR | DC_PROFILE=${DC_PROFILE:-0} ==="

i=0
while read -r script args; do
  [ -z "$script" ] && continue
  i=$((i+1))
  set -- $args
  HASH="${1}_${2}_${3}_${4}_${5}_${6}_${7}_${8}_${10}_${11}_${12}_g${13}"
  # serve 与 breakdown 共用同一个 RUN_HASH, 必须按产物类型判断是否已完成,
  # 否则先跑的那个会让后跑的被误跳过。
  case "$script" in
    bench_serve.sh)          MARK="bench_result" ;;
    bench_breakdown.sh)      MARK="server_breakdown.log" ;;
    bench_exp_vs_latency.sh) MARK="server_explat.log" ;;
    bench_tok_cnt.sh)        MARK="server_tokcnt.log" ;;
    *)                       MARK="bench_result" ;;
  esac
  if compgen -G "$OUTDIR/$HASH/$MARK*" > /dev/null 2>&1; then
    echo ">>> [$i/$TOTAL] skip (done): $HASH [$MARK]"; continue
  fi
  # dataset id 3 = ShareGPT longest-N (prefill-heavy replacement for Edit5kChar)
  if [ "$8" = "3" ]; then export VLLM_SHAREGPT_LONGEST=1; else unset VLLM_SHAREGPT_LONGEST; fi
  echo ">>> [$i/$TOTAL] $script $args  SHAREGPT_LONGEST=${VLLM_SHAREGPT_LONGEST:-0}  $(date +%H:%M:%S)"
  rm -rf "results/$HASH"                 # never inherit a previous round
  bash "./$script" $args
  if [ -d "results/$HASH" ]; then
    mkdir -p "$OUTDIR"
    mv "results/$HASH" "$OUTDIR/$HASH"   # move, not copy
    echo ">>> archived -> $OUTDIR/$HASH"
  else
    echo ">>> WARNING: results/$HASH missing after run"
  fi
done < "$LIST"
echo "=== SWEEP DONE ($LIST -> $OUTDIR) ==="
