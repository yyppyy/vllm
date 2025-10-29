#!/bin/bash

NUM_GPUS=$1
EP_DEGREE=$2
NUM_REPLICAS=$3
BATCH_SIZE=$4
MEM_BOUND_ROUTING=$5
DATASET=$6
RES_DIR=./results

RUN_HASH=${NUM_GPUS}_${EP_DEGREE}_${NUM_REPLICAS}_${BATCH_SIZE}_${MEM_BOUND_ROUTING}_${DATASET}

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
CS=$BATCH_SIZE
CR=$(( BATCH_SIZE > 16 ? 512 : BATCH_SIZE ))

args=(
  serve Qwen/Qwen3-30B-A3B
  --port "$PORT"
  --data-parallel-size "$EP_DEGREE"
  --tensor-parallel-size 1
  --enable-expert-parallel
  --max-num-seqs $CR
  --no-enable-chunked-prefill
  --compilation-config "{\"level\": 3, \"cudagraph_capture_sizes\": [1, 16, 256, 512, 4096]}"
  --max-model-len 4096
  --max-num-batched-tokens $CS
  --expert-placement-strategy linear
)

# if (( NUM_REPLICAS > 0 )); then
  args+=( --enable-eplb )
  args+=( --eplb-config "{\"window_size\":100,\"step_interval\":10000000,\"num_redundant_experts\":${NUM_REPLICAS}}" )
# fi

if (( BATCH_SIZE > 16 )); then
  args+=( --enforce-eager )
fi

if (( MEM_BOUND_ROUTING > 0 )); then
  args+=( --mem-bound-aware-routing greedy )
fi

unset VLLM_TORCH_PROFILER_DIR
unset TOPK_DUMP_PREFIX
vllm "${args[@]}" >"$RES_DIR/server_$RUN_HASH.log" 2>&1 &
SERVER_PID=$!

# Ensure we always stop the server on exit (success or failure)
cleanup() {
    # SIGINT lets vLLM shut down cleanly
    kill -INT "$SERVER_PID" 2>/dev/null || true
    # wait a bit; if it's still around, escalate to SIGTERM
    sleep 30
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    # final fallback after a short wait
    sleep 10
    kill -KILL "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

cli_args=(
    --model Qwen/Qwen3-30B-A3B
    --dataset-name hf
    --dataset-path likaixin/InstructCoder \
    --backend vllm
    --save-result
    --result-filename "$RES_DIR"/bench_result_"$RUN_HASH".json
    --percentile-metrics ttft,tpot,itl,e2el
    --metric-percentiles 10,20,30,40,50,95,99
    --ready-check-timeout-sec 240
    --port "$PORT"
    --num-prompts $CR
    --max-concurrency $CR
)

# decode
if (( BATCH_SIZE > 16 )); then
  cli_args+=( --hf-output-len 1 )
fi

vllm bench serve "${cli_args[@]}"
