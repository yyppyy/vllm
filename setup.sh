#!/usr/bin/env bash
set -euo pipefail

incremental=false
ep_kernels=false
deepep=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --incremental) incremental=true; shift ;;
    --ep-kernels) ep_kernels=true; shift ;;
    --deepep) deepep=true; shift ;;
    -h|--help)
      echo "Usage: $0 [--incremental] [--ep-kernels] [--deepep]"
      echo "  --incremental   Only run the final build+install step"
      echo "  --ep-kernels    Build and install pplx-kernels (editable)"
      echo "  --deepep        Build and install DeepEP (editable)"
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
  EP_WORKSPACE="$SCRIPT_DIR/ep_kernels_workspace"
  if "$incremental"; then
    echo "==> Incremental rebuild of pplx-kernels..."
    if [[ -d "$EP_WORKSPACE/pplx-kernels" ]]; then
      (cd "$EP_WORKSPACE/pplx-kernels" && python setup.py build_ext --inplace)
    else
      echo "  -> pplx-kernels not found at $EP_WORKSPACE/pplx-kernels, skipping (run without --incremental first)"
    fi
  else
    echo "==> Building pplx-kernels..."
    bash "$SCRIPT_DIR/tools/ep_kernels/install_pplx.sh" "$EP_WORKSPACE"
  fi
}

build_deepep() {
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
  EP_WORKSPACE="$SCRIPT_DIR/ep_kernels_workspace"
  if "$incremental"; then
    echo "==> Incremental rebuild of DeepEP..."
    if [[ -d "$EP_WORKSPACE/DeepEP" ]]; then
      (cd "$EP_WORKSPACE/DeepEP" && python setup.py build_ext --inplace)
    else
      echo "  -> DeepEP not found at $EP_WORKSPACE/DeepEP, skipping (run without --incremental first)"
    fi
  else
    echo "==> Building DeepEP..."
    bash "$SCRIPT_DIR/tools/ep_kernels/install_deepep.sh" "$EP_WORKSPACE"
  fi
}

if "$incremental"; then
  source .venv/bin/activate
  cmake --build --preset release --target install
  if "$ep_kernels"; then build_ep_kernels; fi
  if "$deepep"; then build_deepep; fi
  exit 0
fi

# git checkout v0.11.0-gcp

# --- Toolchain pinning (Della) ---------------------------------------------
# The default GCC 11.5 cannot compile PyTorch 2.8's ATen/core/List_inl.h
# ("need 'typename' before ... dependent scope"); GCC 13 can. CMake also
# picks up /usr/local/cuda (13.3) by default, which mismatches the cu128
# torch wheel -- pin nvcc to 12.8 explicitly.
if [[ -d /opt/rh/gcc-toolset-13 ]]; then
  export PATH=/opt/rh/gcc-toolset-13/root/usr/bin:$PATH
  export CC=/opt/rh/gcc-toolset-13/root/usr/bin/gcc
  export CXX=/opt/rh/gcc-toolset-13/root/usr/bin/g++
fi
if [[ -d /usr/local/cuda-12.8 ]]; then
  export CUDA_HOME=/usr/local/cuda-12.8
  export CUDACXX=$CUDA_HOME/bin/nvcc
  export PATH=$CUDA_HOME/bin:$PATH
fi
# A100; set explicitly so configure does not need a visible GPU.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"

uv venv --python 3.12 --seed
source .venv/bin/activate

# Compute nodes have no route to the internet, so every download (wheels,
# models, datasets) must happen on a login node.
VLLM_USE_PRECOMPILED=1 uv pip install -U -e ".[bench]" --torch-backend=cu128

python3 tools/generate_cmake_presets.py
uv pip install cmake ninja
cmake --preset release \
  ${CUDACXX:+-DCMAKE_CUDA_COMPILER=$CUDACXX} \
  ${CC:+-DCMAKE_C_COMPILER=$CC} ${CXX:+-DCMAKE_CXX_COMPILER=$CXX}
cmake --build --preset release --target install

if "$ep_kernels"; then build_ep_kernels; fi
if "$deepep"; then build_deepep; fi
