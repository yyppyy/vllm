#!/bin/bash
set -euo pipefail

pip install -q huggingface_hub

download_model() {
    local model_name="$1"
    local local_dir="$2"
    if [ -d "$local_dir" ] && [ -f "$local_dir/config.json" ]; then
        echo "Model already exists at $local_dir, skipping download."
    else
        echo "Downloading $model_name to $local_dir ..."
        # Della's resolver hands out unreachable CloudFront edges for
        # huggingface.co; tools/net/hf_get.py re-resolves via 8.8.8.8.
        (cd tools/net && python3 hf_get.py "$model_name" \
            "$(cd "$OLDPWD" && pwd)/$local_dir")
    fi
}

patch_topk() {
    local local_dir="$1"
    local new_topk="$2"
    local key_name="${3:-num_experts_per_tok}"
    python3 -c "
import json, sys
cfg_path = '${local_dir}/config.json'
with open(cfg_path, 'r') as f:
    cfg = json.load(f)
orig = cfg.get('${key_name}', '?')
cfg['${key_name}'] = ${new_topk}
with open(cfg_path, 'w') as f:
    json.dump(cfg, f, indent=2)
print(f'Patched ${key_name}: {orig} -> ${new_topk}')
"
}

# Qwen3-30B-A3B
download_model "Qwen/Qwen3-30B-A3B" "./models/Qwen3-30B-A3B"
patch_topk "./models/Qwen3-30B-A3B" 4

# ERNIE-4.5-21B-A3B-PT
download_model "baidu/ERNIE-4.5-21B-A3B-PT" "./models/ERNIE-4.5-21B-A3B-PT"
# patch_topk "./models/ERNIE-4.5-21B-A3B-PT" 3 moe_k
# Make all 14 layers MoE (no dense layers)
patch_topk "./models/ERNIE-4.5-21B-A3B-PT" 27 num_hidden_layers
patch_topk "./models/ERNIE-4.5-21B-A3B-PT" 0 moe_layer_start_index
patch_topk "./models/ERNIE-4.5-21B-A3B-PT" 26 moe_layer_end_index
patch_topk "./models/ERNIE-4.5-21B-A3B-PT" 1 moe_layer_interval
