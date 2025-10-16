#!/bin/bash

# prefill_only=False, chunked_prefill=False
./bench_serve.sh 8 8 0 32 0 0
sleep 30

./bench_serve.sh 8 8 32 32 0 0
sleep 30

./bench_serve.sh 8 8 64 32 0 0
sleep 30

# prefill_only=True, chunked_prefill=False
./bench_serve.sh 8 8 0 32 1 0
sleep 30

./bench_serve.sh 8 8 32 32 1 0
sleep 30

./bench_serve.sh 8 8 64 32 1 0
sleep 30

# prefill_only=False, chunked_prefill=True
./bench_serve.sh 8 8 0 32 0 1
sleep 30

./bench_serve.sh 8 8 32 32 0 1
sleep 30

./bench_serve.sh 8 8 64 32 0 1
sleep 30