#!/bin/bash
set -euo pipefail

DATASET_DIR="./datasets"
mkdir -p "$DATASET_DIR"

# ShareGPT
SHAREGPT_FILE="$DATASET_DIR/ShareGPT_V3_unfiltered_cleaned_split.json"
if [ -f "$SHAREGPT_FILE" ]; then
    echo "ShareGPT already exists at $SHAREGPT_FILE, skipping."
else
    echo "Downloading ShareGPT dataset..."
    pip install -q huggingface_hub
    hf download anon8231489123/ShareGPT_Vicuna_unfiltered \
        --repo-type dataset \
        --include "ShareGPT_V3_unfiltered_cleaned_split.json" \
        --local-dir "$DATASET_DIR"
fi

echo "Done. Available datasets:"
ls -lh "$DATASET_DIR"/*.json 2>/dev/null || echo "  (none)"
