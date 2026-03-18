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
RES_DIR=./results
RUN_HASH=${NUM_GPUS}_${EP_DEGREE}_${USE_EP}_${NUM_REPLICAS}_${BATCH_SIZE}_${MEM_BOUND_ROUTING}_${ALLTOALL_BACKEND}_${DATASET}_${USE_PROFILER}
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
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=INIT,GRAPH
MAX_TOKEN_PER_BATCH=4096
MAX_REQ_PER_BATCH=$BATCH_SIZE
NUM_PROMPTS=$((BATCH_SIZE * NUM_GPUS))

# Patch ROUTING_MODE_THRESHOLD directly in source instead of
# setting env var, which conflicts with nsys profiling.
DCPF_PY="$(VLLM_LOGGING_LEVEL=ERROR python3 -c 'from vllm.model_executor.layers.fused_moe import dispatch_combine_prepare_finalize as m; print(m.__file__)')"
sed -i 's|"VLLM_ROUTING_MODE_THRESHOLD", "[^"]*"|"VLLM_ROUTING_MODE_THRESHOLD", "256"|' "$DCPF_PY"
unset VLLM_ROUTING_MODE_THRESHOLD
# export VLLM_DC_PROFILE=10 # time breakdown debug
# export VLLM_MOE_LOAD_PROFILE_INTERVAL=1 # print expert activation / token distribution

if (( NUM_GPUS <= 2 )); then
  GPU_MEM_UTIL="0.9"
else
  GPU_MEM_UTIL="0.75"
fi

args=(
  serve Qwen/Qwen3-30B-A3B
  --port "$PORT"
  --data-parallel-size "$EP_DEGREE"
  --tensor-parallel-size 1
  --max-num-seqs $MAX_REQ_PER_BATCH
  --no-enable-chunked-prefill
  --compilation-config "{\"level\": 3, \"cudagraph_capture_sizes\": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]}"
  --max-model-len 4096
  --max-num-batched-tokens $MAX_TOKEN_PER_BATCH
  --expert-placement-strategy linear
  --gpu-memory-utilization "$GPU_MEM_UTIL"
)
  # --enforce-eager

if (( USE_EP > 0 )); then
  args+=( --enable-expert-parallel )
  args+=( --enable-eplb )
  EPLB_STEP=$((4 * NUM_PROMPTS))
  args+=( --eplb-config "{\"window_size\":${EPLB_STEP},\"step_interval\":${EPLB_STEP},\"num_redundant_experts\":${NUM_REPLICAS},\"max_rearrangements\":1}" )
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
ENVS_PY="$(VLLM_LOGGING_LEVEL=ERROR python3 -c 'import vllm.envs; print(vllm.envs.__file__)')"
sed -i "s|env_with_choices(\"VLLM_ALL2ALL_BACKEND\", \"[^\"]*\"|env_with_choices(\"VLLM_ALL2ALL_BACKEND\", \"${ALLTOALL_BACKEND}\"|" "$ENVS_PY"
unset VLLM_ALL2ALL_BACKEND

if (( USE_PROFILER > 0 )); then
  setsid nsys profile \
    --trace-fork-before-exec=true \
    --sample=process-tree \
    --cuda-graph-trace=node \
    --delay 30 \
    --duration 6000 \
    --output="$RES_DIR"/"$RUN_HASH"/profile \
    -- \
    vllm "${args[@]}" >"$RES_DIR/$RUN_HASH/server.log" 2>&1 &
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
OUTPUT_LEN=128

# Warmup run: EPLB rebalances during these requests (results discarded)
warmup_args=(
    --model Qwen/Qwen3-30B-A3B
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
else
  warmup_args+=( --dataset-name hf --dataset-path "$DATASET_NAME" )
fi

echo "=== Warmup: sending $WARMUP_PROMPTS requests ==="
vllm bench serve "${warmup_args[@]}"

# Real benchmark run (EPLB already rebalanced, no interference)
cli_args=(
    --model Qwen/Qwen3-30B-A3B
    --backend vllm
    --save-result
    --result-filename "$RES_DIR"/"$RUN_HASH"/bench_result.json
    --percentile-metrics ttft,tpot,itl,e2el
    --metric-percentiles 10,20,30,40,50,95,99
    --ready-check-timeout-sec 2400
    --port "$PORT"
    --num-prompts $NUM_PROMPTS
    --max-concurrency $NUM_PROMPTS
)
if [[ "$DATASET_NAME" == "random" ]]; then
  cli_args+=( --dataset-name random --random-input-len $INPUT_LEN --random-output-len $OUTPUT_LEN )
else
  cli_args+=( --dataset-name hf --dataset-path "$DATASET_NAME" )
fi

# if (( USE_PROFILER > 0 )); then
#   cli_args+=( --profile )
# fi

echo "=== Benchmark: sending $NUM_PROMPTS requests ==="
vllm bench serve "${cli_args[@]}"


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