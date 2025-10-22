#!/bin/bash

# mem_bound_routing=False, chunked_prefill=False
./bench_serve.sh 8 8 0 16 0 0
sleep 30

./bench_serve.sh 8 8 32 16 0 0
sleep 30

./bench_serve.sh 8 8 64 16 0 0

# mem_bound_routing=True, chunked_prefill=False
./bench_serve.sh 8 8 0 16 1 0
sleep 30

./bench_serve.sh 8 8 32 16 1 0
sleep 30

./bench_serve.sh 8 8 64 16 1 0