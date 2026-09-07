#!/bin/bash
#SBATCH --job-name=metro-build
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/build-%j.out
#SBATCH --error=slurm/logs/build-%j.out

set -euo pipefail
cd /scratch/gpfs/WENTZLAF/hm2595/metro_moe/vllm
module purge; module load cudatoolkit/12.8
export PATH="$HOME/.local/bin:$PATH"
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS=32 CMAKE_BUILD_PARALLEL_LEVEL=32
export CCACHE_DIR=/scratch/gpfs/WENTZLAF/hm2595/.ccache
source .venv/bin/activate
nvcc --version | tail -2
python3 tools/generate_cmake_presets.py < /dev/null || true
cmake --preset release
cmake --build --preset release --target install
echo "=== BUILD OK ==="
python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda)"
python -c "import vllm._C; print('vllm._C ok')"
python -c "import torch; import vllm._C; print('dispatch_and_route:', hasattr(torch.ops._C,'dispatch_and_route'))"
