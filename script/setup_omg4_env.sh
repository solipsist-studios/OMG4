#!/bin/bash
# Create the `omg4` conda env on a modern toolchain (python 3.11, torch cu130)
# and build this repo's CUDA extensions into it.
#
# Overridable environment variables:
#   CONDA                 conda executable (default: conda from PATH)
#   TORCH_CUDA_ARCH_LIST  GPU compute capability to build for
#                         (default 12.0 = sm_120, e.g. RTX 5090; set to
#                         your GPU's capability, `nvidia-smi --query-gpu=compute_cap --format=csv`)
#   MAX_JOBS              parallel compile jobs (default: nproc)
set -x
CONDA=${CONDA:-conda}
REPO="$(cd "$(dirname "$0")/.." && pwd)"

$CONDA create -y -n omg4 python=3.11 || exit 1
PY="$($CONDA info --base)/envs/omg4/bin/python"
PIP="$PY -m pip"
echo "=== STEP conda env created ==="
$PIP install --index-url https://download.pytorch.org/whl/cu130 torch torchvision || exit 1
echo "=== STEP torch installed ==="
$PIP install numpy plyfile dahuffman lpips omegaconf imageio imageio-ffmpeg tqdm scikit-learn ninja || exit 1
echo "=== STEP basic deps installed ==="
$PIP install cupy-cuda13x || echo "=== WARN cupy-cuda13x failed ==="
$PIP install "cuml-cu13" || $PIP install "cuml-cu12" || echo "=== WARN cuml failed ==="
echo "=== STEP rapids attempted ==="
cd "$REPO" || exit 1
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"12.0"}
export MAX_JOBS=${MAX_JOBS:-$(nproc)}
$PIP install ./diff-gaussian-rasterization --no-build-isolation && echo "=== STEP rasterizer built ===" || echo "=== FAIL rasterizer ==="
$PIP install ./simple-knn --no-build-isolation && echo "=== STEP simple-knn built ===" || echo "=== WARN simple-knn failed ==="
$PIP install ./pointops2 --no-build-isolation && echo "=== STEP pointops2 built ===" || echo "=== WARN pointops2 failed ==="
$PY -c "
import torch; print('torch', torch.__version__, torch.cuda.is_available())
import diff_gaussian_rasterization; print('rasterizer ok')
try:
    import cuml; print('cuml ok')
except Exception as e: print('cuml MISSING:', e)
try:
    import cupy; print('cupy ok')
except Exception as e: print('cupy MISSING:', e)
"
echo "=== DONE ==="
