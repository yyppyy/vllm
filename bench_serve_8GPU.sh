#!/usr/bin/env bash
set -euo pipefail

# declare associative array
declare -A token_max

# Fill it: token_max["<dataset>,<replication>"]=<value>
token_max["0,0"]=AAA
token_max["0,16"]=BBB
token_max["0,32"]=CCC
token_max["0,64"]=DDD

token_max["1,0"]=EEE
token_max["1,16"]=FFF
token_max["1,32"]=GGG
token_max["1,64"]=HHH

token_max["2,0"]=III
token_max["2,16"]=JJJ
token_max["2,32"]=KKK
token_max["2,64"]=LLL

for dataset in 0 1 2; do
    for routing_scheme in 1 0; do
        for replication in 0 16 32 64; do

            decode_bs=16
            prefill_bs=${token_max["$dataset,$replication"]}

            # decode
            ./bench_serve.sh 8 8 "$replication" "$decode_bs" "$routing_scheme" "$dataset"

            # prefill
            ./bench_serve.sh 8 8 "$replication" "$prefill_bs" "$routing_scheme" "$dataset"

        done
    done
done
