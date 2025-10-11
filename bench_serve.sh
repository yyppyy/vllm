#!/bin/bash

NUM_GPUS=$1
EP_DEGREE=$2
NUM_REPLICAS=$3
BATCH_SIZE=$4
RES_DIR=./results

PORT=$(python3 -c 'import socket as s; sock=s.socket(); sock.bind(("",0)); print(sock.getsockname()[1]); sock.close()')

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

vllm bench serve \
    --model Qwen/Qwen3-30B-A3B \
    --dataset-name hf \
    --dataset-path philschmid/mt-bench \
    --backend vllm \
    --save-result \
    --result-dir $RES_DIR \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,95,99 \
    --ready-check-timeout-sec 360 \
    --port $PORT
    # --hf-output-len use this to increase decode ratio?
