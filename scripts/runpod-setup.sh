#!/usr/bin/env bash
# One-shot installer for open-translate on a RunPod PyTorch pod.
# Target image: runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
#   (CUDA 12.8.1, torch 2.8 cu128, Ubuntu 24.04, Python 3.11)
# Works on RTX 4090 (Ada sm_89) and RTX 5090 (Blackwell sm_120).
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/AgentBlitz/open-translate.git}"
REPO_DIR="${REPO_DIR:-/workspace/open-translate}"
BRANCH="${BRANCH:-main}"

echo "==> open-translate RunPod setup"
echo "    repo: $REPO_URL"
echo "    dir:  $REPO_DIR"
echo "    branch: $BRANCH"

if [ ! -d "$REPO_DIR/.git" ]; then
  git clone --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
else
  echo "==> repo already present, pulling latest"
  git -C "$REPO_DIR" fetch origin "$BRANCH"
  git -C "$REPO_DIR" checkout "$BRANCH"
  git -C "$REPO_DIR" pull --ff-only
fi

cd "$REPO_DIR"

echo "==> installing python deps (keeping pod's torch 2.8 cu128)"
# Strip the torch pin from requirements.txt so we don't clobber the
# pre-installed cu128 build. Everything else installs normally.
TMP_REQ="$(mktemp)"
grep -v -E '^(torch|torchvision|torchaudio)([=<>!].*)?$' requirements.txt > "$TMP_REQ"

pip install --no-cache-dir \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -r "$TMP_REQ"
rm -f "$TMP_REQ"

echo "==> installing PaddleOCR stack (cu126 wheel runs on cu128 host)"
pip install --no-cache-dir \
  paddlepaddle-gpu==3.0.0 \
  -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
pip install --no-cache-dir \
  paddleocr==3.0.0 \
  pypdfium2==4.30.0 \
  python-docx==1.1.2 \
  pypdf==5.1.0 \
  python-multipart==0.0.20 \
  Pillow==11.0.0

echo "==> pre-warming PaddleOCR weights into /workspace/.cache/paddlex"
export PADDLE_PDX_CACHE_HOME="${PADDLE_PDX_CACHE_HOME:-/workspace/.cache/paddlex}"
mkdir -p "$PADDLE_PDX_CACHE_HOME"
python - <<'PY'
from paddleocr import PaddleOCR
PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=True,
    lang="en",
    device="cpu",  # warm download; real requests use gpu
)
print("paddleocr: weights cached")
PY

echo "==> verifying CUDA + GPU"
python - <<'PY'
import torch
print(f"torch       : {torch.__version__}")
print(f"cuda avail  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  gpu {i}    : {p.name}  sm_{p.major}{p.minor}  {p.total_memory/1e9:.1f} GB")
PY

echo
echo "==> launching server on port ${PORT:-8000}"
echo "    On RunPod, open the HTTP proxy URL for this port, then visit /ui/"
echo "    Example: https://<pod-id>-${PORT:-8000}.proxy.runpod.net/ui/"
echo

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export NLLB_MODEL_SIZE="${NLLB_MODEL_SIZE:-1.3B-distilled}"
export DTYPE="${DTYPE:-fp16}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"

exec bash ./start.sh
