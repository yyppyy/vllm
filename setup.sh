#!/usr/bin/env bash
set -euo pipefail

incremental=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --incremental) incremental=true; shift ;;
    -h|--help)
      echo "Usage: $0 [--incremental]"
      echo "  --incremental   Only run the final build+install step"
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

if "$incremental"; then
  source .venv/bin/activate
  python3 tools/generate_cmake_presets.py
  cmake --preset release
  cmake --build --preset release --target install
  exit 0
fi

git checkout v0.11.0-gcp

uv venv --python 3.12 --seed
source .venv/bin/activate

VLLM_USE_PRECOMPILED=1 uv pip install -U -e ".[bench]" --torch-backend=auto

python3 tools/generate_cmake_presets.py
python -m pip install cmake ninja
cmake --preset release
cmake --build --preset release --target install
