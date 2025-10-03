#!/bin/bash

RES_DIR=./results

PORT=$(python3 -c 'import socket as s; sock=s.socket(); sock.bind(("",0)); print(sock.getsockname()[1]); sock.close()')

vllm serve Qwen/Qwen3-30B-A3B --port $PORT >$RES_DIR/server.log 2>&1 &

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
