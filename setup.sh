#!/bin/bash

cd ~/project_pi_ak2579/yy594/vllm
git checkout v0.11.0

source .venv/bin/activate

module load CUDA/12.8.0
module load GCC/13.3.0

VLLM_USE_PRECOMPILED=1 uv pip install -U -e . --torch-backend=auto

python tools/generate_cmake_presets.py
cmake --preset release
cmake --build --preset release --target install
