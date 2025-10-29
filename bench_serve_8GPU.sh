#!/usr/bin/env bash
set -euo pipefail

# token_max[0] = x, token_max[1] = y, token_max[2] = z
token_max=(x y z)

# loops:
#   dataset in (0 1 2) 0=humaneval; 1=gpqa; 2=gsm8k
#   routing_scheme in (1 0)
#   replication in (0 16 32 64)
for dataset in 0 1 2; do
    for routing_scheme in 1 0; do
        for replication in 0 16 32 64; do

            # decode run (batch size = 16)
            ./bench_serve.sh 8 8 "$replication" 16 "$routing_scheme" "$dataset"

            # prefill run (batch size = token_max[dataset])
            ./bench_serve.sh 8 8 "$replication" "${token_max[$dataset]}" "$routing_scheme" "$dataset"

        done
    done
done
