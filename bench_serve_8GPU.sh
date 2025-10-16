#!/bin/bash

./bench_serve.sh 8 8 0 32
sleep 30

./bench_serve.sh 8 8 32 32
sleep 30

./bench_serve.sh 8 8 64 32