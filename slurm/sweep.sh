#!/usr/bin/env bash
# Emit (and optionally submit) the METRO-vs-EPLB sweep.
#   bash slurm/sweep.sh <bench_script> [submit]
# Positional bench args:
#   NUM_GPUS EP USE_EP NUM_REPLICAS BATCH ROUTING BACKEND DATASET DATASET_NAME
#   USE_PROFILER THRES MODEL EPLB_GROUPS
set -uo pipefail
BENCH="${1:-bench_serve.sh}"
SUBMIT="${2:-dry}"

declare -A DS_ID=( ["likaixin/InstructCoder"]=0 ["vdaita/edit_5k_char"]=1 )

emit() {  # model  n_experts  replicas  batch  dataset  thres
  local model=$1 rep=$3 bs=$4 ds=$5 thres=$6
  local dsid=${DS_ID[$ds]}
  echo "sbatch slurm/run_bench.slurm $BENCH 8 8 1 $rep $bs 2 dispatch_combine $dsid $ds 0 $thres $model 1"
}

for ds in "likaixin/InstructCoder" "vdaita/edit_5k_char"; do
  for bs in 4 8 16 32 64; do
    # ERNIE: 64 logical experts -> 1.5x = 32 redundant, 2.0x = 64
    for rep in 0 32 64; do
      [[ $rep -eq 0 ]] && { emit ERNIE-4.5-21B-A3B-PT 64 0 $bs "$ds" 0; continue; }
      emit ERNIE-4.5-21B-A3B-PT 64 $rep $bs "$ds" 0     # EPLB routing
      emit ERNIE-4.5-21B-A3B-PT 64 $rep $bs "$ds" 256   # METRO routing
    done
    # Qwen3: 128 logical experts -> 1.5x = 64 redundant, 2.0x = 128
    for rep in 0 64 128; do
      [[ $rep -eq 0 ]] && { emit Qwen3-30B-A3B-8-128 128 0 $bs "$ds" 0; continue; }
      emit Qwen3-30B-A3B-8-128 128 $rep $bs "$ds" 0
      emit Qwen3-30B-A3B-8-128 128 $rep $bs "$ds" 256
    done
  done
done | if [[ "$SUBMIT" == "submit" ]]; then bash -x; else cat; fi
