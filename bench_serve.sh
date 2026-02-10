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
)

if (( USE_EP > 0 )); then
  args+=( --enable-expert-parallel )
  args+=( --enable-eplb )
  args+=( --eplb-config "{\"window_size\":100,\"step_interval\":10000000,\"num_redundant_experts\":${NUM_REPLICAS}}" )
  if (( MEM_BOUND_ROUTING > 0 )); then
    args+=( --mem-bound-aware-routing greedy )
  fi
  # args+=( --all2all-backend $ALLTOALL_BACKEND)
else
  args+=( --no-enable-expert-parallel )
fi

unset VLLM_TORCH_PROFILER_DIR
unset TOPK_DUMP_PREFIX

if (( USE_PROFILER > 0 )); then
  nsys profile \
    --trace-fork-before-exec=true \
    --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi \
    --capture-range-end=repeat \
    --output="$RES_DIR"/"$RUN_HASH"/profile \
    VLLM_ALL2ALL_BACKEND=${ALLTOALL_BACKEND} vllm "${args[@]}" >"$RES_DIR/$RUN_HASH/server.log" 2>&1 &
  NSYS_PID=$!
  # Get the actual vllm PID (child of nsys)
  sleep 10  # Give nsys time to fork vllm
  SERVER_PID=$(pgrep -P "$NSYS_PID" | head -1)
  if [[ -z "$SERVER_PID" ]]; then
    SERVER_PID=$NSYS_PID  # Fallback if we can't find child
  fi
else
  vllm "${args[@]}" >"$RES_DIR/$RUN_HASH/server.log" 2>&1 &
  SERVER_PID=$!
  NSYS_PID=""
fi

# Ensure we always stop the server on exit (success or failure)
cleanup() {
    if [[ -n "$NSYS_PID" ]]; then
        # Signal nsys, it will handle stopping vllm
        kill -INT "$NSYS_PID" 2>/dev/null || true
        wait "$NSYS_PID" 2>/dev/null || true
    else
        kill -INT "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT


################ client #################

MAX_CONCURRENT_REQ=$((BATCH_SIZE * NUM_GPUS))

cli_args=(
    --model Qwen/Qwen3-30B-A3B
    --dataset-name hf
    --dataset-path $DATASET_NAME \
    --backend vllm
    --save-result
    --result-filename "$RES_DIR"/"$RUN_HASH"/bench_result.json
    --percentile-metrics ttft,tpot,itl,e2el
    --metric-percentiles 10,20,30,40,50,95,99
    --ready-check-timeout-sec 2400
    --port "$PORT"
    --num-prompts $MAX_CONCURRENT_REQ
    --max-concurrency $MAX_CONCURRENT_REQ
)

if (( USE_PROFILER > 0 )); then
  cli_args+=( --profile )
fi

vllm bench serve "${cli_args[@]}"
