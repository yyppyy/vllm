#!/bin/bash

NUM_GPUS=$1
EP_DEGREE=$2
NUM_REPLICAS=$3
BATCH_SIZE=$4
RES_DIR=./results

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

vllm serve Qwen/Qwen3-30B-A3B \
    --port $PORT \
    --tensor-parallel-size $EP_DEGREE \
    --enable-expert-parallel \
    --enable-eplb \
    --eplb-config "{\"window_size\":1000,\"step_interval\":3000,\"num_redundant_experts\":$NUM_REPLICAS}" \
    --max-num-batched-tokens $BATCH_SIZE \
    --enable-chunked-prefill \
    -O.level=3 \
    >$RES_DIR/server.log 2>&1 &
SERVER_PID=$!

# Ensure we always stop the server on exit (success or failure)
cleanup() {
    # SIGINT lets vLLM shut down cleanly
    kill -INT "$SERVER_PID" 2>/dev/null || true
    # wait a bit; if it's still around, escalate to SIGTERM
    sleep 10
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    # final fallback after a short wait
    sleep 10
    kill -KILL "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

vllm bench serve \
    --model Qwen/Qwen3-30B-A3B \
    --dataset-name hf \
    --dataset-path philschmid/mt-bench \
    --backend vllm \
    --save-result \
    --result-dir $RES_DIR \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,95,99 \
    --ready-check-timeout-sec 120 \
    --port $PORT
    # --hf-output-len use this to increase decode ratio?
