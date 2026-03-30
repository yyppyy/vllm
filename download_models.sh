#!/bin/bash
set -euo pipefail

MODEL_NAME="Qwen/Qwen3-30B-A3B"
LOCAL_DIR="./models/Qwen3-30B-A3B"

if [ -d "$LOCAL_DIR" ] && [ -f "$LOCAL_DIR/config.json" ]; then
    echo "Model already exists at $LOCAL_DIR, skipping download."
else
    echo "Downloading $MODEL_NAME to $LOCAL_DIR ..."
    pip install -q huggingface_hub
    huggingface-cli download "$MODEL_NAME" --local-dir "$LOCAL_DIR"
fi

# Patch num_experts_per_tok from 8 to 4
python3 -c "
import json, sys
cfg_path = '${LOCAL_DIR}/config.json'
with open(cfg_path, 'r') as f:
    cfg = json.load(f)
orig = cfg.get('num_experts_per_tok', '?')
cfg['num_experts_per_tok'] = 4
with open(cfg_path, 'w') as f:
    json.dump(cfg, f, indent=2)
print(f'Patched num_experts_per_tok: {orig} -> 4')
"

# echo "Done. Use '$LOCAL_DIR' as the model path in bench_serve.sh."
