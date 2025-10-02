#!/bin/bash

echo "=== Host Info ==="
hostname
date
echo

echo "=== Loading CUDA module ==="
module load CUDA/12.8.0
echo

echo "=== PATH and LD_LIBRARY_PATH ==="
echo "PATH=$PATH"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "CUDA_HOME=$CUDA_HOME"
echo

echo "=== NVIDIA Driver / CUDA Toolkit ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "nvidia-smi not found"
fi
echo

if command -v nvcc &> /dev/null; then
    nvcc --version
else
    echo "nvcc not found"
fi
echo

echo "=== Python & PyTorch (from venv) ==="
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "Activated Python venv: .venv"
    python -c "import sys; print('Python', sys.version)"
    python -c "import torch; print('Torch', torch.__version__); \
                print('CUDA available:', torch.cuda.is_available()); \
                print('Compiled CUDA:', getattr(torch.version, 'cuda', None)); \
                print('cuDNN:', torch.backends.cudnn.version() if torch.cuda.is_available() else None); \
                print('Visible GPUs:', torch.cuda.device_count()); \
                [print(f'  GPU{i}:', torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]" \
                2>/dev/null || echo "PyTorch not available in venv"
else
    echo ".venv not found in current directory"
fi
