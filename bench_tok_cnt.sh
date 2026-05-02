#!/bin/bash
#
# Token-count profile companion to bench_exp_vs_latency.sh.
#
# Same 13 positional args as bench_exp_vs_latency.sh. Differences:
#   - Zipfian routing DISABLED (VLLM_ZIPFIAN_ROUTING=0).
#   - Profile output goes to results/$RUN_HASH/server_tokcnt.log
#     instead of server_explat.log.
#   - Bench client sends prefill requests at a Poisson rate matched
#     to the expected steady-state concurrency (Little's law,
#     option (a)) instead of blasting all NUM_PROMPTS at once.

NUM_GPUS=$1
EP_DEGREE=$2
USE_EP=$3
NUM_REPLICAS=$4
BATCH_SIZE=$5
MEM_BOUND_ROUTING=$6
ALLTOALL_BACKEND=$7
DATASET=$8
DATASET_NAME=$9
USE_PROFILER=${10}        # accepted for signature parity but ignored
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

unset NCCL_NET NCCL_SOCKET_IFNAME NCCL_NET_PLUGIN
export NCCL_NET_PLUGIN=none
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_P2P_LEVEL=NVL
unset NCCL_PROTO

MAX_TOKEN_PER_BATCH=8192
MAX_REQ_PER_BATCH=$BATCH_SIZE
NUM_PROMPTS=$((BATCH_SIZE * NUM_GPUS))

# Same source patches as bench_exp_vs_latency.sh.
DCPF_PY="vllm/model_executor/layers/fused_moe/dispatch_combine_prepare_finalize.py"
sed -i "s|\"VLLM_ROUTING_MODE_THRESHOLD\", \"[^\"]*\"|\"VLLM_ROUTING_MODE_THRESHOLD\", \"${MEM_BOUND_ROUTING_THRES}\"|" "$DCPF_PY"
unset VLLM_ROUTING_MODE_THRESHOLD
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
SCHED_PY="vllm/v1/core/sched/scheduler.py"
sed -i 's|"VLLM_PREFILL_BEFORE_DECODE", "[^"]*"|"VLLM_PREFILL_BEFORE_DECODE", "0"|' "$SCHED_PY"
unset VLLM_PREFILL_BEFORE_DECODE

# Differences from bench_exp_vs_latency.sh: zipfian routing OFF.
export VLLM_ZIPFIAN_ROUTING=0
export VLLM_EPLB_NUM_GROUPS=${EPLB_NUM_GROUPS}

# Per-(rank, layer, batch) profile gate. Output goes to
# server_tokcnt.log instead of server_explat.log so this run's data
# doesn't collide with bench_exp_vs_latency.sh in the same RUN_HASH.
READY_FILE="$RES_DIR/$RUN_HASH/tokcnt_ready"
TOKCNT_LOG="$RES_DIR/$RUN_HASH/server_tokcnt.log"
rm -f "$READY_FILE" "$TOKCNT_LOG"
export VLLM_EXP_LATENCY_PROFILE=1
export VLLM_EXP_LATENCY_READY_FILE="$READY_FILE"
export VLLM_EXP_LATENCY_LOG_PATH="$TOKCNT_LOG"

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
  --max-model-len 8192
  --max-num-batched-tokens $MAX_TOKEN_PER_BATCH
  --expert-placement-strategy linear
  --gpu-memory-utilization "$GPU_MEM_UTIL"
)

if (( USE_EP > 0 )); then
  args+=( --enable-expert-parallel )
  args+=( --enable-eplb )
  EPLB_STEP=$((2 * NUM_PROMPTS))
  if (( EPLB_NUM_GROUPS > 1 )); then
    MAX_REARR=0
  else
    MAX_REARR=1
  fi
  args+=( --eplb-config "{\"window_size\":${EPLB_STEP},\"step_interval\":${EPLB_STEP},\"num_redundant_experts\":${NUM_REPLICAS},\"max_rearrangements\":${MAX_REARR}}" )
  if (( MEM_BOUND_ROUTING > 0 )); then
    args+=( --mem-bound-aware-routing greedy )
  fi
else
  args+=( --no-enable-expert-parallel )
fi

unset VLLM_TORCH_PROFILER_DIR
unset TOPK_DUMP_PREFIX
ENVS_PY="vllm/envs.py"
sed -i "s|env_with_choices(\"VLLM_ALL2ALL_BACKEND\", \"[^\"]*\"|env_with_choices(\"VLLM_ALL2ALL_BACKEND\", \"${ALLTOALL_BACKEND}\"|" "$ENVS_PY"
unset VLLM_ALL2ALL_BACKEND

setsid vllm "${args[@]}" \
  >"$RES_DIR/$RUN_HASH/server_main.log" 2>&1 &
SERVER_PID=$!
SESSION_PID=$SERVER_PID

################ client #################

WARMUP_PROMPTS=$((1 * NUM_PROMPTS))

INPUT_LEN=512
OUTPUT_LEN=512

# Steady-state arrival rate (option (a)). Estimate request lifetime
# W (sojourn time) and apply Little's law: lambda = NUM_PROMPTS / W
# so the server runs at ~NUM_PROMPTS in flight on average.
#
# Lifetime ≈ INPUT_LEN / PREFILL_RATE_PER_REQ + OUTPUT_LEN / DECODE_RATE_PER_SEQ.
# Override either constant by exporting the variable before invocation.
PREFILL_RATE_PER_REQ=${PREFILL_RATE_PER_REQ:-5000}     # tok/s/req
DECODE_RATE_PER_SEQ=${DECODE_RATE_PER_SEQ:-40}         # tok/s/seq
W_SEC=$(python3 -c "print($INPUT_LEN / $PREFILL_RATE_PER_REQ + $OUTPUT_LEN / $DECODE_RATE_PER_SEQ)")
REQUEST_RATE=$(python3 -c "print(round($NUM_PROMPTS / $W_SEC, 2))")
BURSTINESS=1.0
echo "=== Steady-state arrival: lambda=$REQUEST_RATE req/s "\
"(NUM_PROMPTS=$NUM_PROMPTS, W=${W_SEC}s, prefill=${PREFILL_RATE_PER_REQ} tok/s, "\
"decode=${DECODE_RATE_PER_SEQ} tok/s/seq) ==="

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

echo "=== Warmup: sending $WARMUP_PROMPTS requests (no profile output) ==="
vllm bench serve "${warmup_args[@]}"

# Flip the gate. Subsequent (rank, layer, batch) emit one record into
# server_tokcnt.log.
echo "=== Warmup done; touching $READY_FILE to enable token-count output ==="
touch "$READY_FILE"

# Steady-state client run. Poisson arrivals at lambda=REQUEST_RATE,
# pure Poisson (burstiness=1.0).
cli_args=(
    --model "$MODEL_DIR"
    --backend vllm
    --seed 0
    --percentile-metrics ttft,tpot,itl,e2el
    --metric-percentiles 10,20,30,40,50,95,99
    --ready-check-timeout-sec 2400
    --port "$PORT"
    --num-prompts $NUM_PROMPTS
    --max-concurrency $NUM_PROMPTS
    --request-rate "$REQUEST_RATE"
    --burstiness "$BURSTINESS"
)
if [[ "$DATASET_NAME" == "random" ]]; then
  cli_args+=( --dataset-name random --random-input-len $INPUT_LEN --random-output-len $OUTPUT_LEN )
elif [[ "$DATASET_NAME" == "sharegpt" ]]; then
  cli_args+=( --dataset-name sharegpt --dataset-path ./datasets/ShareGPT_V3_unfiltered_cleaned_split.json --sharegpt-output-len $OUTPUT_LEN )
else
  cli_args+=( --dataset-name hf --dataset-path "$DATASET_NAME" --hf-output-len $OUTPUT_LEN )
fi

echo "=== Profile run: sending $NUM_PROMPTS Poisson(lambda=$REQUEST_RATE) requests (token-count output enabled) ==="
vllm bench serve "${cli_args[@]}"

# The in-process explat poller drains the pinned ringbuffer 30s after
# the last replay. Give it 35s of slack before tearing the server down.
echo "=== Bench done; sleeping 35s to let the explat poller drain ==="
sleep 35

if [[ -s "$TOKCNT_LOG" ]]; then
  echo "=== TokCnt log written: $TOKCNT_LOG ($(wc -l <"$TOKCNT_LOG") lines) ==="
else
  echo "WARN: TokCnt log $TOKCNT_LOG missing or empty"
fi

############## kill server ##############

rm -f "$READY_FILE"

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
