#!/usr/bin/env bash
set -euo pipefail

for dataset in 0 1 2; do
    for routing_scheme in 1 0; do
        for replication in 0 16 32 64; do

            decode_bs=16

            # decode
            ./bench_serve.sh 8 8 "$replication" "$decode_bs" "$routing_scheme" "$dataset"

        done
    done
done
