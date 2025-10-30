#!/usr/bin/env bash
set -euo pipefail

# declare associative array
dataset_names=('likaixin/InstructCoder' 'AI-MO/NuminaMath-1.5' 'Aeala/ShareGPT_Vicuna_unfiltered')

for dataset in 0 1 2; do
    for routing_scheme in 1 0; do
        for replication in 0 16 32 48 64; do

            decode_bs=16

            # decode
            ./bench_serve.sh 8 8 "$replication" "$decode_bs" "$routing_scheme" "$dataset" "${dataset_names[$dataset]}"

        done
    done
done
