#!/bin/bash

./log_bench_serve.sh 2 2 0 8
sleep 30
./log_bench_serve.sh 2 2 0 32
sleep 30

./log_bench_serve.sh 2 2 8 8
sleep 30
./log_bench_serve.sh 2 2 8 32
sleep 30