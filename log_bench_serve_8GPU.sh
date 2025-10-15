#!/bin/bash

./log_bench_serve.sh 8 8 0 2
sleep 30
./log_bench_serve.sh 8 8 0 8
sleep 30
./log_bench_serve.sh 8 8 0 32
sleep 30
./log_bench_serve.sh 8 8 0 128
sleep 30

./log_bench_serve.sh 8 8 32 2
sleep 30
./log_bench_serve.sh 8 8 32 8
sleep 30
./log_bench_serve.sh 8 8 32 32
sleep 30
./log_bench_serve.sh 8 8 32 128
sleep 30

./log_bench_serve.sh 8 8 64 2
sleep 30
./log_bench_serve.sh 8 8 64 8
sleep 30
./log_bench_serve.sh 8 8 64 32
sleep 30
./log_bench_serve.sh 8 8 64 128
sleep 30