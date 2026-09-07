#!/bin/bash

NUM_GPUS=$1
EP_DEGREE=$2
USE_EP=$3
NUM_REPLICAS=$4
BATCH_SIZE=$5
MEM_BOUND_ROUTING=$6
ALLTOALL_BACKEND=$7
DATASET=$8
DATASET_NAME=$9
USE_PROFILER=${10}
MEM_BOUND_ROUTING_THRES=${11}
MODEL_NAME=${12:-Qwen3-30B-A3B}
EPLB_NUM_GROUPS=${13:-1}
MODEL_DIR=./models/${MODEL_NAME}

# If MODEL_NAME matches Qwen3-30B-A3B-{topk}-{num_experts},
# use the base model and patch config at runtime.
if [[ "$MODEL_NAME" =~ ^Qwen3-30B-A3B-([0-9]+)-([0-9]+)$ ]]; then
  QWEN_TOPK="${BASH_REMATCH[1]}"
  QWEN_NUM_EXPERTS="${BASH_REMATCH[2]}"
  MODEL_DIR="./models/Qwen3-30B-A3B"
  python3 -c "
import json
cfg = json.load(open('${MODEL_DIR}/config.json'))
# Preserve original checkpoint expert count (idempotent).
cfg['_checkpoint_num_experts'] = cfg.get(
    '_checkpoint_num_experts', cfg.get('num_experts'))
cfg['num_experts_per_tok'] = ${QWEN_TOPK}
cfg['num_experts'] = ${QWEN_NUM_EXPERTS}
json.dump(cfg, open('${MODEL_DIR}/config.json', 'w'), indent=2)
print(f'Patched Qwen3 config: topk=${QWEN_TOPK}, num_experts=${QWEN_NUM_EXPERTS}, ckpt_experts={cfg[\"_checkpoint_num_experts\"]}')
"
fi

RES_DIR=./results
RUN_HASH=${NUM_GPUS}_${EP_DEGREE}_${USE_EP}_${NUM_REPLICAS}_${BATCH_SIZE}_${MEM_BOUND_ROUTING}_${ALLTOALL_BACKEND}_${DATASET}_${USE_PROFILER}_${MEM_BOUND_ROUTING_THRES}_${MODEL_NAME}_g${EPLB_NUM_GROUPS}
mkdir -p "$RES_DIR"/"$RUN_HASH"

PORT=$(python3 -c 'import socket as s; sock=s.socket(); sock.bind(("",0)); print(sock.getsockname()[1]); sock.close()')

source .venv/bin/activate

# 1) Clear anything you set earlier
unset NCCL_NET NCCL_SOCKET_IFNAME NCCL_NET_PLUGIN
# 2) Prefer pure intra-node P2P (NVLink) and disable external net plugins
export NCCL_NET_PLUGIN=none
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_P2P_LEVEL=NVL
unset NCCL_PROTO
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=INIT,GRAPH
MAX_TOKEN_PER_BATCH=8192
MAX_REQ_PER_BATCH=$BATCH_SIZE
NUM_PROMPTS=$((BATCH_SIZE * NUM_GPUS))

# Patch ROUTING_MODE_THRESHOLD directly in source instead of
# setting env var, which conflicts with nsys profiling.
DCPF_PY="vllm/model_executor/layers/fused_moe/dispatch_combine_prepare_finalize.py"
sed -i "s|\"VLLM_ROUTING_MODE_THRESHOLD\", \"[^\"]*\"|\"VLLM_ROUTING_MODE_THRESHOLD\", \"${MEM_BOUND_ROUTING_THRES}\"|" "$DCPF_PY"
unset VLLM_ROUTING_MODE_THRESHOLD
# Patch VLLM_PREFILL_ROUTING_MODE: MEM_BOUND_ROUTING=1 → mode 1 (LPT),
# MEM_BOUND_ROUTING=2 → mode 2 (round-robin),
# MEM_BOUND_ROUTING=3 → mode 3 (minimize per-token rank fanout).
if (( MEM_BOUND_ROUTING == 1 )); then
  sed -i 's|"VLLM_PREFILL_ROUTING_MODE", "[^"]*"|"VLLM_PREFILL_ROUTING_MODE", "1"|' "$DCPF_PY"
elif (( MEM_BOUND_ROUTING == 2 )); then
  sed -i 's|"VLLM_PREFILL_ROUTING_MODE", "[^"]*"|"VLLM_PREFILL_ROUTING_MODE", "2"|' "$DCPF_PY"
elif (( MEM_BOUND_ROUTING == 3 )); then
  sed -i 's|"VLLM_PREFILL_ROUTING_MODE", "[^"]*"|"VLLM_PREFILL_ROUTING_MODE", "3"|' "$DCPF_PY"
elif (( MEM_BOUND_ROUTING != 0 )); then
  echo "ERROR: MEM_BOUND_ROUTING must be 0, 1, 2, or 3 (got $MEM_BOUND_ROUTING)" >&2
  exit 1
fi
# unset VLLM_PREFILL_ROUTING_MODE
SCHED_PY="vllm/v1/core/sched/scheduler.py"
sed -i 's|"VLLM_PREFILL_BEFORE_DECODE", "[^"]*"|"VLLM_PREFILL_BEFORE_DECODE", "0"|' "$SCHED_PY"
unset VLLM_PREFILL_BEFORE_DECODE

export VLLM_ZIPFIAN_ROUTING=1
export VLLM_EPLB_NUM_GROUPS=${EPLB_NUM_GROUPS}
# export VLLM_DC_PROFILE=1 # time breakdown debug
# export VLLM_DC_EXPERT_PROFILE_M=256 # threshold

# export VLLM_MOE_LOAD_PROFILE_INTERVAL=1 # print expert activation / token distribution
# export VLLM_ROUTING_DEBUG=1 # print routing decisions

if (( NUM_GPUS <= 2 )); then
  GPU_MEM_UTIL="0.9"
else
  GPU_MEM_UTIL="0.75"
fi

args=(
  serve "$MODEL_DIR"
  --port "$PORT"
  --data-parallel-size "$EP_DEGREE"
  --tensor-parallel-size 1
  --max-num-seqs $MAX_REQ_PER_BATCH
  --no-enable-chunked-prefill
  --compilation-config "{\"level\": 3, \"cudagraph_capture_sizes\": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]}"
  --max-model-len 8192
  --max-num-batched-tokens $MAX_TOKEN_PER_BATCH
  --expert-placement-strategy linear
  --gpu-memory-utilization "$GPU_MEM_UTIL"
)
  # --enforce-eager

if (( USE_EP > 0 )); then
  args+=( --enable-expert-parallel )
  args+=( --enable-eplb )
  EPLB_STEP=$((2 * NUM_PROMPTS))
  # When grouping is active, disable runtime rearrange — the existing
  # rebalance algorithm is not group-aware and would clobber the layout.
  if (( EPLB_NUM_GROUPS > 1 )); then
    MAX_REARR=0
  else
    MAX_REARR=1
  fi
  args+=( --eplb-config "{\"window_size\":${EPLB_STEP},\"step_interval\":${EPLB_STEP},\"num_redundant_experts\":${NUM_REPLICAS},\"max_rearrangements\":${MAX_REARR}}" )
  if (( MEM_BOUND_ROUTING > 0 )); then
    args+=( --mem-bound-aware-routing greedy )
  fi
  # args+=( --all2all-backend $ALLTOALL_BACKEND)
else
  args+=( --no-enable-expert-parallel )
fi

unset VLLM_TORCH_PROFILER_DIR
unset TOPK_DUMP_PREFIX
# HACK: patch the default VLLM_ALL2ALL_BACKEND in envs.py instead of
# setting the env var, which conflicts with nsys profiling.
ENVS_PY="vllm/envs.py"
sed -i "s|env_with_choices(\"VLLM_ALL2ALL_BACKEND\", \"[^\"]*\"|env_with_choices(\"VLLM_ALL2ALL_BACKEND\", \"${ALLTOALL_BACKEND}\"|" "$ENVS_PY"
unset VLLM_ALL2ALL_BACKEND

if (( USE_PROFILER > 0 )); then
  # nsys is run via sudo so it can read GPU PMU counters
  # (--gpu-metrics-set / --gpu-metrics-device). Without root,
  # NVreg_RestrictProfilingToAdminUsers=1 makes nsys abort with
  # ERR_NVGPUCTRPERM. sudo's secure_path drops the venv, so resolve
  # vllm's absolute path BEFORE sudo and pass it through.
  VLLM_BIN=$(command -v vllm)
  setsid sudo -E env "PATH=$PATH" "VIRTUAL_ENV=$VIRTUAL_ENV" \
    nsys profile \
    --trace-fork-before-exec=true \
    --sample=process-tree \
    --cuda-graph-trace=node \
    --gpu-metrics-set=ga100 \
    --gpu-metrics-device=all \
    --delay 30 \
    --duration 6000 \
    --output="$RES_DIR"/"$RUN_HASH"/profile \
    -- \
    "$VLLM_BIN" "${args[@]}" >"$RES_DIR/$RUN_HASH/server.log" 2>&1 &
  NSYS_PID=$!
  SESSION_PID=$NSYS_PID     # setsid => session leader PID == NSYS_PID
else
  setsid vllm "${args[@]}" >"$RES_DIR/$RUN_HASH/server.log" 2>&1 &
  SERVER_PID=$!
  SESSION_PID=$SERVER_PID
  NSYS_PID=""
fi

################ client #################

WARMUP_PROMPTS=$((1 * NUM_PROMPTS))

INPUT_LEN=512
OUTPUT_LEN=512

# Warmup run: EPLB rebalances during these requests (results discarded)
warmup_args=(
    --model "$MODEL_DIR"
    --backend vllm
    --save-result
    --result-filename /dev/null
    --percentile-metrics ttft,tpot,itl,e2el
    --metric-percentiles 10,20,30,40,50,95,99
    --ready-check-timeout-sec 2400
    --port "$PORT"
    --num-prompts $WARMUP_PROMPTS
    --max-concurrency $NUM_PROMPTS
)
if [[ "$DATASET_NAME" == "random" ]]; then
  warmup_args+=( --dataset-name random --random-input-len $INPUT_LEN --random-output-len $OUTPUT_LEN )
elif [[ "$DATASET_NAME" == "sharegpt" ]]; then
  warmup_args+=( --dataset-name sharegpt --dataset-path ./datasets/ShareGPT_V3_unfiltered_cleaned_split.json --sharegpt-output-len $OUTPUT_LEN )
else
  warmup_args+=( --dataset-name hf --dataset-path "$DATASET_NAME" --hf-output-len $OUTPUT_LEN )
fi

echo "=== Warmup: sending $WARMUP_PROMPTS requests ==="
vllm bench serve "${warmup_args[@]}"

# ---------------------------------------------------------------------
# v2: block until the EPLB rearrange has actually finished.
# The original script only *assumed* warmup absorbed it. Measured on
# della-l06g12: a rearrange takes 13.3 s while the measured window is
# ~14 s, so when it spills over it inflates duration by ~7 s and wrecks
# throughput / TTFT (2 of 6 repeat runs were hit; throughput CV 21%,
# p99 TTFT CV 147%). Matching "Rearranged experts in" excludes the
# engine-startup "(profile)" pass, which has a different message.
# ---------------------------------------------------------------------
if (( USE_EP > 0 )); then
  echo "=== Waiting for EPLB rearrange to settle ==="
  SRV_LOG="$RES_DIR/$RUN_HASH/server.log"
  REARR_OK=0
  for _ in $(seq 1 240); do
    if grep -qE "Rearranged experts in [0-9.]+ seconds" "$SRV_LOG" 2>/dev/null; then
      echo "  rearrange observed; quiescing 10s"; sleep 10; REARR_OK=1; break
    fi
    sleep 1
  done
  (( REARR_OK == 0 )) && echo "  WARNING: no rearrange seen in 240s; measuring anyway"
fi

# Real benchmark run (EPLB already rebalanced, no interference).
# Run multiple sequential clients to gather >=TARGET_TOTAL_PROMPTS total
# prompts for stable percentiles. Each client uses the same NUM_PROMPTS as
# both --num-prompts and --max-concurrency, so the in-flight load matches
# the configured batch capacity. The plotting script pools per-prompt
# latencies across these per-client files.
TARGET_TOTAL_PROMPTS=128
if (( NUM_PROMPTS >= TARGET_TOTAL_PROMPTS )); then
  NUM_CLIENT_RUNS=1
else
  NUM_CLIENT_RUNS=$(( (TARGET_TOTAL_PROMPTS + NUM_PROMPTS - 1) / NUM_PROMPTS ))
fi

# Wipe any stale bench_result*.json from a previous run at the same hash.
rm -f "$RES_DIR"/"$RUN_HASH"/bench_result*.json

for ((CLIENT_IDX=0; CLIENT_IDX<NUM_CLIENT_RUNS; CLIENT_IDX++)); do
  cli_args=(
      --model "$MODEL_DIR"
      --backend vllm
      --save-result
      --save-detailed
      --result-filename "$RES_DIR"/"$RUN_HASH"/bench_result_${CLIENT_IDX}.json
      --percentile-metrics ttft,tpot,itl,e2el
      --metric-percentiles 10,20,30,40,50,95,99
      --ready-check-timeout-sec 2400
      --port "$PORT"
      --num-prompts $NUM_PROMPTS
      --max-concurrency $NUM_PROMPTS
  )
  if [[ "$DATASET_NAME" == "random" ]]; then
    cli_args+=( --dataset-name random --random-input-len $INPUT_LEN --random-output-len $OUTPUT_LEN )
  elif [[ "$DATASET_NAME" == "sharegpt" ]]; then
    cli_args+=( --dataset-name sharegpt --dataset-path ./datasets/ShareGPT_V3_unfiltered_cleaned_split.json --sharegpt-output-len $OUTPUT_LEN )
  else
    cli_args+=( --dataset-name hf --dataset-path "$DATASET_NAME" --hf-output-len $OUTPUT_LEN )
  fi

  echo "=== Bench client $((CLIENT_IDX+1))/$NUM_CLIENT_RUNS: sending $NUM_PROMPTS requests (seed=$CLIENT_IDX) ==="
  vllm bench serve "${cli_args[@]}"
done


############## kill server & collect profile and logs ##############
if [[ -n "${NSYS_PID:-}" ]]; then
  # Profiler path: signal ONLY nsys (not the process group).
  # nsys must stay alive to collect profiling data from its
  # traced children. Killing the group would terminate vllm
  # workers before nsys can read their profiling buffers,
  # causing "Collecting data..." to hang forever.
  kill -INT "$NSYS_PID" 2>/dev/null || true

  # Give nsys generous time to collect + write (up to 10min).
  # Large profiles with many traced processes can take a while.
  for _ in {1..6000}; do
    kill -0 "$NSYS_PID" 2>/dev/null || break
    sleep 0.1
  done

  # If nsys is still alive, escalate to TERM (avoids SIGKILL
  # which corrupts the output file).
  if kill -0 "$NSYS_PID" 2>/dev/null; then
    kill -TERM "$NSYS_PID" 2>/dev/null || true
    for _ in {1..100}; do
      kill -0 "$NSYS_PID" 2>/dev/null || break
      sleep 0.1
    done
  fi

  wait "$NSYS_PID" 2>/dev/null || true

  # Now kill any remaining vllm workers that nsys left behind.
  # SESSION_PID == NSYS_PID (setsid leader), so the process
  # group contains both nsys (now dead) and vllm children.
  kill -TERM -- "-$SESSION_PID" 2>/dev/null || true
  sleep 1
  kill -KILL -- "-$SESSION_PID" 2>/dev/null || true
else
  # Non-profiler path: kill the session group
  kill -INT  -- "-$SESSION_PID" 2>/dev/null || true

  for _ in {1..200}; do
    kill -0 "$SESSION_PID" 2>/dev/null || break
    sleep 0.1
  done

  if kill -0 "$SESSION_PID" 2>/dev/null; then
    kill -TERM -- "-$SESSION_PID" 2>/dev/null || true
    for _ in {1..100}; do
      kill -0 "$SESSION_PID" 2>/dev/null || break
      sleep 0.1
    done
  fi

  if kill -0 "$SESSION_PID" 2>/dev/null; then
    kill -KILL -- "-$SESSION_PID" 2>/dev/null || true
  fi

  wait "$SESSION_PID" 2>/dev/null || true
fi