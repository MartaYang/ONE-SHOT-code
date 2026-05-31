#!/usr/bin/env bash
# ONE-SHOT environment installer.
#
# Builds the `oneshot` conda env with everything needed for the inference +
# preprocessing pipeline, including a self-contained ffmpeg / ffprobe inside
# the env (so video I/O does not depend on any host-side ffmpeg).
#
# Usage:
#     bash install.sh
#
# This script does NOT configure proxies or pip/conda mirrors — that is up
# to the user.  If you are behind a corporate proxy or need a regional
# mirror, configure them BEFORE running:
#   - proxy:    export http_proxy / https_proxy / no_proxy
#   - pip src:  export PIP_INDEX_URL / PIP_TRUSTED_HOST  (or write ~/.pip/pip.conf)
#   - conda src: edit ~/.condarc to point `channels:` at a conda-forge mirror
#                (the ffmpeg step below requires conda-forge to be reachable).
#
# Hard prerequisites:
#   - conda (any recent version) on PATH
#   - NVIDIA driver compatible with CUDA 12.1 (we install the cu121 PyTorch wheels)
#
# Pinned versions: Python 3.12 + torch 2.5.1+cu121 + pytorch3d 0.7.9.

set -euo pipefail

ENV_NAME=oneshot
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
P3D_WHEEL="${SCRIPT_DIR}/Preprocessing/third_party/wheels/pytorch3d-0.7.9-cp312-cp312-manylinux_2_31_x86_64.whl"

echo "[install.sh] creating conda env: ${ENV_NAME}"
conda create -n "${ENV_NAME}" python=3.12 -y

echo "[install.sh] installing ffmpeg (GPL build with libx264) into ${ENV_NAME}"
# We require a GPL-enabled ffmpeg (libx264 encoder) — skvideo / our video I/O
# helpers shell out to `ffmpeg -c:v libx264 ...`.  The default-channel ffmpeg
# is built with --disable-gpl and will fail with "Unknown encoder 'libx264'".
# The `*gpl*` build-string filter forces the GPL variant published on
# conda-forge.  We do NOT pass `-c conda-forge` here so that whatever mirror
# the user configured in ~/.condarc is honored (passing `-c conda-forge`
# bypasses mirror config and goes directly to conda.anaconda.org).
conda install -n "${ENV_NAME}" 'ffmpeg=*=*gpl*' -y

ENV_PIP="$(conda run -n ${ENV_NAME} python -c 'import sys,os; print(os.path.join(sys.prefix, "bin", "pip"))')"
ENV_PY="$(conda run -n ${ENV_NAME} python -c 'import sys,os; print(os.path.join(sys.prefix, "bin", "python"))')"
echo "[install.sh] resolved ENV_PIP=${ENV_PIP}"
echo "[install.sh] resolved ENV_PY=${ENV_PY}"

echo "[install.sh] installing PyTorch 2.5.1+cu121"
"${ENV_PIP}" install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121

echo "[install.sh] installing pytorch3d 0.7.9 (vendored wheel)"
"${ENV_PIP}" install "${P3D_WHEEL}"

echo "[install.sh] installing remaining requirements"
"${ENV_PIP}" install -r "${SCRIPT_DIR}/requirements.txt"

echo "[install.sh] smoke test"
"${ENV_PY}" - <<'PY'
import subprocess, torch
from pytorch3d.renderer import MeshRasterizer  # noqa: F401
import diffusers, transformers, skvideo.io, smplx, roma, trimesh, pyrender  # noqa: F401
import omegaconf, ftfy  # noqa: F401
from depth_anything_3.api import DepthAnything3  # noqa: F401
out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                     capture_output=True, text=True, check=True).stdout
assert "libx264" in out, "ffmpeg in this env was built without libx264 (GPL)"
print(f"OK  torch={torch.__version__} cuda={torch.cuda.is_available()}  ffmpeg+libx264 ok")
PY

echo
echo "[install.sh] done.  Activate with:  conda activate ${ENV_NAME}"
