#!/bin/bash
# LL3DA inference/scoring env as a uv venv, ported to current GPUs: the README pins
# torch 1.13.1+cu116, which has no kernels for Blackwell (sm_120), so this uses
# torch 2.7.1+cu128, transformers 4.40.2 (README: >= 4.37), builds the pointnet2 ops and the
# cython box-IoU, and adds h5py/scipy/plyfile/trimesh (repo imports) and openai (Beacon3D's
# GPT-judged QA evaluator).
#   bash extra_eval/build_env.sh
# Run on a GPU node with internet access (compiles CUDA code). Paths come from env.sh:
#   LL3DA_ENV        where the venv goes                       (default .venv)
#   LL3DA_CUDA_HOME  CUDA 12.x toolkit, >= 12.8                 (default /usr/local/cuda)
#   TORCH_CUDA_ARCH_LIST  GPU archs to compile for             (default: the visible GPU's)
#   UV_CACHE_DIR / TMPDIR   point these at a disk with room if $HOME is full
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO="$PWD"
[ -f env.sh ] && source env.sh
unset CONDA_PREFIX CONDA_DEFAULT_ENV
UV="${UV:-uv}"
ENV="${LL3DA_ENV:-.venv}"
export CUDA_HOME="${LL3DA_CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS="${MAX_JOBS:-16}"
$UV python install 3.10
[ -x "$ENV/bin/python" ] || $UV venv "$ENV" -p 3.10 --seed
PY="$ENV/bin/python"
PIP="$UV pip install --python $PY"
$PIP torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
$PIP transformers==4.40.2 "numpy<2" scipy h5py cython plyfile "trimesh<4" "networkx<3" \
    matplotlib tensorboardX tqdm huggingface_hub openai
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$($PY -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability())')}"
echo "CUDA_HOME=$CUDA_HOME TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

# pointnet2 ops (installed as top-level package `pointnet2._ext`)
if ! $PY -c "import torch, pointnet2._ext" 2>/dev/null; then
    cd third_party/pointnet2
    $PY setup.py install
    rm -rf build dist *.egg-info
    cd "$REPO"
fi
# cython box intersection (utils/box_intersection*.so, in place)
if ! (cd "$REPO" && $PY -c "from utils.box_intersection import box_intersection") 2>/dev/null; then
    cd "$REPO/utils"
    $PY cython_compile.py build_ext --inplace
    rm -rf build
    cd "$REPO"
    git checkout -- utils/box_intersection.c 2>/dev/null || true
fi

PYTHONPATH=. $PY -c "
import torch, transformers, pointnet2._ext
import third_party.pointnet2.pointnet2_modules  # puts pointnet2/ on sys.path, as the detector does
from third_party.pointnet2.pointnet2_utils import furthest_point_sample
from utils.box_intersection import box_intersection
print('torch', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
print('transformers', transformers.__version__)
x = torch.rand(1, 1024, 3, device='cuda')
print('furthest_point_sample ok', furthest_point_sample(x, 16).shape)
"
echo ENV_DONE
