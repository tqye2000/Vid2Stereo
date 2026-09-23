#!/usr/bin/env bash
# One-time environment setup for Vid2Stereo (run inside WSL2 / Ubuntu).
#
# Clones the official m2svid repo + submodules, creates the two conda
# environments, and downloads the pretrained weights.
#
# Prerequisites:
#   - conda / miniconda on PATH
#   - git, wget, unzip
#   - NVIDIA driver + CUDA-capable GPU visible in WSL2 (`nvidia-smi` works)
#
# Usage:
#   bash setup.sh
#
# Some steps are heavy (large downloads) and environment-specific; read the
# comments if a step needs manual attention.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
M2SVID_DIR="${HERE}/m2svid"
CONDA_DIR="${HOME}/miniforge3"

# --------------------------------------------------------------------------- #
# 0a. Corporate TLS (Zscaler) CA bundle — required behind the corporate proxy
# --------------------------------------------------------------------------- #
# This network intercepts TLS. Point every downloader at a CA bundle exported
# from the Windows trust store (contains the Zscaler roots). Generate it with:
#   powershell scripts/export-windows-ca.ps1   (writes corp-ca-bundle.pem)
CA_BUNDLE="${HOME}/corp-ca-bundle.pem"
if [[ ! -f "${CA_BUNDLE}" && -f "${HERE}/corp-ca-bundle.pem" ]]; then
  cp "${HERE}/corp-ca-bundle.pem" "${CA_BUNDLE}"
  sed -i 's/\r$//' "${CA_BUNDLE}"   # normalize CRLF -> LF
fi
if [[ -f "${CA_BUNDLE}" ]]; then
  echo "==> Using corporate CA bundle: ${CA_BUNDLE}"
  # NOTE: do NOT export SSL_CERT_FILE / REQUESTS_CA_BUNDLE here. conda's base
  # Python uses strict X509 verification and the Zscaler root has non-critical
  # basicConstraints, so those globals would force conda to fail. curl/git/pip
  # use lenient/dedicated settings below. The inference runtime sets
  # REQUESTS_CA_BUNDLE itself for the Python 3.10 envs (which are not strict).
  export CURL_CA_BUNDLE="${CA_BUNDLE}"
  export GIT_SSL_CAINFO="${CA_BUNDLE}"
  export PIP_CERT="${CA_BUNDLE}"
  export NODE_EXTRA_CA_CERTS="${CA_BUNDLE}"
  git config --global http.sslCAInfo "${CA_BUNDLE}" || true
else
  echo "WARNING: ${CA_BUNDLE} not found; downloads may fail behind Zscaler." >&2
fi

# --------------------------------------------------------------------------- #
# 0b. Bootstrap conda + ffmpeg (no sudo required)
# --------------------------------------------------------------------------- #
# This machine has no system conda and sudo needs a password, so install a
# user-space Miniforge (conda-forge based). Note: repo.anaconda.com is blocked
# by the corporate proxy (HTTP 403), so we avoid Anaconda's defaults channel.
if ! command -v conda >/dev/null 2>&1; then
  if [[ ! -x "${CONDA_DIR}/bin/conda" ]]; then
    echo "==> Installing Miniforge into ${CONDA_DIR} (user-space, no sudo)"
    tmp_installer="$(mktemp --suffix=.sh)"
    # Use curl (OpenSSL) rather than wget (GnuTLS) — verifies the Zscaler chain.
    curl -fSL -A 'Mozilla/5.0' --cacert "${CA_BUNDLE}" -o "${tmp_installer}" \
      https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash "${tmp_installer}" -b -p "${CONDA_DIR}"
    rm -f "${tmp_installer}"
  fi
fi

# Make conda usable in this non-interactive shell.
if [[ -z "${CONDA_EXE:-}" ]] && ! command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "${CONDA_DIR}/etc/profile.d/conda.sh"
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda config --set always_yes true >/dev/null 2>&1 || true
# conda.anaconda.org is blocked by the corporate proxy (HTTP 403), so use the
# prefix.dev conda-forge mirror (reachable) via channel_alias.
conda config --set channel_alias https://prefix.dev >/dev/null 2>&1 || true
if [[ -f "${CA_BUNDLE}" ]]; then
  # conda's (base) Python enables strict X509 verification and the corporate
  # Zscaler Root CA has non-critical basicConstraints (not RFC 5280 compliant),
  # so verification against the bundle fails. Disable it for the conda tool
  # only. All hosts are already TLS-intercepted by the corporate proxy; the
  # Python 3.10 envs below still verify normally via PIP_CERT / the bundle.
  conda config --set ssl_verify false >/dev/null 2>&1 || true
  pip config set global.cert "${CA_BUNDLE}" >/dev/null 2>&1 || true
fi

# ffmpeg + ffprobe (used by vid2stereo.py post-processing) via conda-forge.
if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  echo "==> Installing ffmpeg into conda base (conda-forge)"
  conda install -n base -y -c conda-forge ffmpeg
fi

# --------------------------------------------------------------------------- #
# 1. Clone m2svid + submodules
# --------------------------------------------------------------------------- #
if [[ ! -d "${M2SVID_DIR}/.git" ]]; then
  echo "==> Cloning m2svid"
  git clone --recurse-submodules \
    https://github.com/google-research/m2svid.git "${M2SVID_DIR}"
else
  echo "==> m2svid already cloned; updating submodules"
  git -C "${M2SVID_DIR}" submodule update --init --recursive
fi

cd "${M2SVID_DIR}"

# Hi3D-Official is required on PYTHONPATH (openclip model + training utils).
if [[ ! -d third_party/Hi3D-Official ]]; then
  echo "==> Cloning Hi3D-Official into third_party/"
  git clone https://github.com/yanghb22-fdu/Hi3D-Official.git \
    third_party/Hi3D-Official
fi

# --------------------------------------------------------------------------- #
# 2. Conda env: depthcrafter
# --------------------------------------------------------------------------- #
# The DepthCrafter submodule HEAD (post-v1.0.1 "refactor") targets python>=3.13
# / torch>=2.7 and pulls a numpy-2 / accelerate stack that breaks on the cuda
# 11.8 / torch 2.0.1 base m2svid uses. The tagged v1.0.1 release ships a
# requirements.txt pinned to exactly that torch-2.0.1 stack and matching code,
# and its depth .npz output is format-compatible with warping.py. So we pin the
# submodule to v1.0.1 and install its requirements.txt.
DC_DIR="third_party/DepthCrafter"
if [[ -d "${DC_DIR}/.git" || -f "${DC_DIR}/.git" ]]; then
  git -C "${DC_DIR}" checkout -q v1.0.1 2>/dev/null || true
fi

# Resumable: if the env exists and already imports the runtime deps, skip.
dc_ready=0
if conda env list | grep -qE '^depthcrafter[[:space:]]'; then
  if conda run -n depthcrafter python -c "import torch, fire, decord, diffusers" >/dev/null 2>&1; then
    dc_ready=1
    echo "==> conda env 'depthcrafter' already complete"
  fi
fi

if [[ "${dc_ready}" -eq 0 ]]; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if ! conda env list | grep -qE '^depthcrafter[[:space:]]'; then
    echo "==> Creating conda env 'depthcrafter'"
    conda create -y -n depthcrafter python=3.10
  else
    echo "==> conda env 'depthcrafter' exists; installing runtime deps"
  fi
  conda activate depthcrafter
  pip install torch==2.0.1 torchvision==0.15.2 \
    --index-url https://download.pytorch.org/whl/cu118
  # Install DepthCrafter v1.0.1's pinned runtime deps. xformers is optional (the
  # pipeline logs a warning and runs without it) and its 0.0.20 wheel is picky
  # about the exact torch build, so we skip it to keep torch 2.0.1 intact.
  if [[ -f "${DC_DIR}/requirements.txt" ]]; then
    grep -vE '^(torch==|torchvision==|xformers==)' "${DC_DIR}/requirements.txt" \
      > /tmp/dc-req.txt
    pip install -r /tmp/dc-req.txt
  fi
  conda deactivate
  # DepthCrafter's run.py writes a preview _depth.mp4 via mediapy, which needs
  # an ffmpeg binary on the env PATH (base's ffmpeg is not visible under
  # `conda run -n depthcrafter`). Install ffmpeg into the env itself.
  conda install -n depthcrafter -y -c conda-forge ffmpeg
fi

# --------------------------------------------------------------------------- #
# 3. Conda env: sgm (m2svid)
# --------------------------------------------------------------------------- #
# NOTE: the upstream environment.yml pip section is a frozen `pip freeze` dump
# and cannot be installed as-is:
#   * torch/vision/audio are pinned to +cu118 local versions that only exist on
#     the PyTorch index (no --extra-index-url is declared in the file),
#   * `clip==1.0` / `sdata==0.0.1` are git-installed packages whose frozen
#     versions collide with unrelated PyPI packages,
#   * `deepspeed==0.14.2` needs torch present at build time, and
#   * `transformers==4.47.0.dev0` (and any other `.devN` pin) is a git build
#     that does not exist on PyPI.
# For INFERENCE we don't need clip / sdata / deepspeed (verified: the sgm code
# only imports them from training-only modules), so we install the torch stack
# first (with the cu118 index) and then the remaining pins with those entries
# stripped out and any trailing `.devN` suffix removed.
#
# This block is resumable: if the env already has torch (e.g. a previous run
# died in the middle of the pip install) we keep it and just (re)run the pip
# step instead of wiping and rebuilding from scratch.
sgm_ready=0
if conda env list | grep -qE '^sgm[[:space:]]'; then
  if conda run -n sgm python -c "import torch, xformers" >/dev/null 2>&1; then
    sgm_ready=1
    echo "==> conda env 'sgm' already complete"
  fi
fi

if [[ "${sgm_ready}" -eq 0 ]]; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"

  if conda run -n sgm python -c "import torch" >/dev/null 2>&1; then
    echo "==> 'sgm' already has torch; resuming pip dependency install"
    conda activate sgm
    export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu118"
  else
    if conda env list | grep -qE '^sgm[[:space:]]'; then
      echo "==> Removing broken 'sgm' env (missing torch)"
      conda env remove -y -n sgm
    fi
    echo "==> Creating conda env 'sgm' (conda deps only, pip handled separately)"
    # Strip the pip section -> conda deps only.
    sed '/^[[:space:]]*-[[:space:]]*pip:/,$d' environment.yml > /tmp/sgm-conda.yml
    conda env create -f /tmp/sgm-conda.yml -n sgm

    conda activate sgm
    export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu118"

    echo "==> Installing torch stack (cu118) first"
    pip install \
      torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 \
      --extra-index-url https://download.pytorch.org/whl/cu118
  fi

  echo "==> Installing remaining pip deps (inference subset)"
  # Extract the pip requirements from environment.yml, dropping the entries that
  # are unresolvable / unneeded for inference and normalising `.devN` pins.
  awk '/^[[:space:]]*-[[:space:]]*pip:/{f=1;next} f&&/^[[:space:]]+-[[:space:]]/{sub(/^[[:space:]]+-[[:space:]]*/,"");print}' environment.yml \
    | grep -vE '^(clip==|sdata==|deepspeed==|tokenizers==|torch==|torchvision==|torchaudio==)' \
    | sed -E 's/\.dev[0-9]+$//' \
    > /tmp/sgm-req.txt
  pip install --extra-index-url https://download.pytorch.org/whl/cu118 -r /tmp/sgm-req.txt

  conda deactivate
  unset PIP_EXTRA_INDEX_URL
fi

# warping.py / inpaint_and_refine.py in the sgm env shell out to ffprobe/ffmpeg
# (via ffmpeg-python), which must be on the env PATH under `conda run -n sgm`.
if ! conda run -n sgm bash -lc 'command -v ffprobe >/dev/null 2>&1'; then
  echo "==> Installing ffmpeg into 'sgm' env (ffprobe for warping)"
  conda install -n sgm -y -c conda-forge ffmpeg
fi

# --------------------------------------------------------------------------- #
# 4. Weights
# --------------------------------------------------------------------------- #
mkdir -p ckpts

# 4a. OpenCLIP ViT-H/14 image encoder used by sgm's FrozenOpenCLIPImageEmbedder
#     (configs/m2svid.yaml -> version: "ckpts/open_clip_pytorch_model.bin").
#     This is the laion2b_s32b_b79k checkpoint, fetched straight from HuggingFace.
if [[ ! -f ckpts/open_clip_pytorch_model.bin ]]; then
  echo "==> Downloading open_clip_pytorch_model.bin (ViT-H/14, ~3.9 GB)"
  curl -fSL -C - --cacert "${CA_BUNDLE}" -o ckpts/open_clip_pytorch_model.bin \
    https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/open_clip_pytorch_model.bin
fi

# 4b. M2SVid weights (full-attention variant, 4.64 GB).
if [[ ! -f ckpts/m2svid_weights.pt ]]; then
  echo "==> Downloading m2svid_weights.pt (4.64 GB)"
  curl -fSL -C - --cacert "${CA_BUNDLE}" -o ckpts/m2svid_weights.pt \
    https://storage.googleapis.com/gresearch/m2svid/m2svid_weights.pt
fi

# 4c. (optional) no-full-attention variant
# curl -fSL -C - --cacert "${CA_BUNDLE}" -o ckpts/m2svid_no_full_atten_weights.pt \
#   https://storage.googleapis.com/gresearch/m2svid/m2svid_no_full_atten_weights.pt

echo ""
echo "==> Setup complete."
echo "    Smoke test the upstream pipeline:  (cd m2svid && bash inference.sh)"
echo "    Or run the wrapper:                python vid2stereo.py -i INPUT.mp4 -o out/"
