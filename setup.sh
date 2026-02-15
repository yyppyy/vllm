#!/usr/bin/env bash
set -euo pipefail

incremental=false
ep_kernels=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --incremental) incremental=true; shift ;;
    --ep-kernels) ep_kernels=true; shift ;;
    -h|--help)
      echo "Usage: $0 [--incremental] [--ep-kernels]"
      echo "  --incremental   Only run the final build+install step"
      echo "  --ep-kernels    Build and install pplx-kernels + DeepEP (editable, requires CUDA_HOME and TORCH_CUDA_ARCH_LIST)"
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

build_ep_kernels() {
  # Auto-detect CUDA_HOME if not already set
  if [[ -z "${CUDA_HOME:-}" ]]; then
    if command -v nvcc &>/dev/null; then
      CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
    elif [[ -d /usr/local/cuda ]]; then
      CUDA_HOME=/usr/local/cuda
    else
      echo "ERROR: CUDA_HOME is not set and could not be auto-detected." >&2
      exit 1
    fi
    export CUDA_HOME
    echo "==> Auto-detected CUDA_HOME=${CUDA_HOME}"
  fi
  # Auto-detect GPU arch if not already set
  if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)"
    export TORCH_CUDA_ARCH_LIST
    echo "==> Auto-detected TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
  fi
  SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
  EP_WORKSPACE="$SCRIPT_DIR/tools/ep_kernels/ep_kernels_workspace"
  if "$incremental"; then
    echo "==> Incremental rebuild of EP kernels..."
    for pkg in pplx-kernels DeepEP; do
      if [[ -d "$EP_WORKSPACE/$pkg" ]]; then
        echo "  -> Rebuilding $pkg"
        (cd "$EP_WORKSPACE/$pkg" && python setup.py build_ext --inplace)
      else
        echo "  -> $pkg not found at $EP_WORKSPACE/$pkg, skipping (run without --incremental first)"
      fi
    done
  else
    echo "==> Building EP kernels (pplx-kernels + DeepEP)..."
    bash "$SCRIPT_DIR/tools/ep_kernels/install_python_libraries.sh"
  fi
}

if "$incremental"; then
  source .venv/bin/activate
  cmake --build --preset release --target install
  if "$ep_kernels"; then build_ep_kernels; fi
  exit 0
fi

# git checkout v0.11.0-gcp

uv venv --python 3.12 --seed
source .venv/bin/activate

VLLM_USE_PRECOMPILED=1 uv pip install -U -e ".[bench]" --torch-backend=auto

python3 tools/generate_cmake_presets.py
python -m pip install cmake ninja
cmake --preset release
cmake --build --preset release --target install

if "$ep_kernels"; then build_ep_kernels; fi
