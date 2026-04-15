# syntax=docker/dockerfile:1.6
#
# open-translate — self-contained GPU translation server
# Built for NVIDIA GPUs with driver >= 570 (Blackwell/Ada/Hopper).
# Runs: docker run --gpus all -p 8005:8005 ghcr.io/agentblitz/open-translate:latest

# ---- builder stage ----------------------------------------------------------
# devel base is used only at build time so any wheel that needs CUDA headers
# can find them. The final runtime stage below uses the smaller -runtime base.
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 AS builder
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        python3.11 python3-pip python3.11-venv \
        ca-certificates curl \
        build-essential \
        libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*
# libgl1 + libglib2.0-0: paddlex -> opencv-contrib-python imports cv2, which
# dlopens libGL.so.1 at module load. The minimal nvidia/cuda base images are
# headless and don't ship libGL. Without these two packages the PaddleOCR
# prewarm in the builder — and `_startup` in the runtime stage — both fail
# with `ImportError: libGL.so.1: cannot open shared object file`.

RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Core deps: torch 2.9.1+cu128 + transformers + fastapi + deepspeed + langdetect.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Document-translation + OCR stack.
#
# paddlepaddle-gpu==3.2.2 from the cu129 wheel index. The cu129 index no longer
# ships 3.0.0 (lowest is 3.1.0) and 3.2.2 is the version verified on the Vast
# RTX 5090 / sm_120 pod. Paddle bundles its own cu12.9 nvidia-* libs which
# overwrite torch's cu12.8 equivalents; torch still reports `cuda 12.8` and
# detects sm_120, so the overlap is cosmetic in practice.
#
# paddleocr is intentionally unpinned — resolves to 3.4.x which is all-wheel.
# Do NOT re-pin paddleocr==3.0.0: on Python 3.11/3.12 it transitively drags
# paddlex[ocr]==3.0.0, which tries to build an old pandas from source inside
# pip's isolated build env and fails with `ModuleNotFoundError: pkg_resources`.
# PaddleOCR 3.4.x exposes the same `.predict(np_array)` / `rec_texts` API that
# server.py uses, so no code changes are needed.
RUN pip install --no-cache-dir \
        paddlepaddle-gpu==3.2.2 \
        -i https://www.paddlepaddle.org.cn/packages/stable/cu129/ && \
    pip install --no-cache-dir \
        paddleocr \
        pypdfium2==4.30.0 \
        python-docx==1.1.2 \
        pypdf==5.1.0 \
        python-multipart==0.0.20 \
        Pillow==11.0.0

# Bake model weights directly into the image so `docker run` is fully offline
# after pull. Caches live under /opt/* (NOT /workspace/*) so a user-mounted
# volume at /workspace doesn't shadow them.
ENV HF_HOME=/opt/hf_cache \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PADDLE_PDX_CACHE_HOME=/opt/paddlex_cache
RUN mkdir -p /opt/hf_cache /opt/paddlex_cache

# Pre-download NLLB-200-distilled-1.3B (~3.3 GB on disk) into /opt/hf_cache.
# This is the default model the server loads at startup; baking it means first
# request is fast and no HuggingFace network access is required at boot.
RUN python - <<'PY'
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
mid = "facebook/nllb-200-distilled-1.3B"
AutoTokenizer.from_pretrained(mid)
AutoModelForSeq2SeqLM.from_pretrained(mid)
print("NLLB cached")
PY

# Pre-download PaddleOCR detection + recognition + textline-orientation weights
# (~500 MB) into /opt/paddlex_cache. Uses device=cpu because the builder stage
# has no GPU; weights are architecture-independent.
#
# This is the awkward step in the build. paddlepaddle-gpu's import-time init
# dlopens libcuda.so.1 to probe the driver, even when device=cpu is requested
# later. At `docker build` time there is no host NVIDIA driver bind-mounted so
# libcuda.so.1 is absent and the import raises. The combined workaround:
#
#   1. Symlink libcuda.so → libcuda.so.1 in the CUDA driver stub directory
#      that the -devel base image ships for exactly this case (stubs are
#      API-only; they let dlopen succeed without a real driver).
#   2. Point LD_LIBRARY_PATH at the stub dir for this RUN only (scoped env
#      doesn't leak into the runtime stage because we only copy /opt/venv
#      and /opt/*_cache out of the builder).
#   3. Set CUDA_VISIBLE_DEVICES="" so paddle skips any GPU enumeration.
#
# If all of the above still fails (paddle 3.x has been tightening its driver
# checks), the step is **non-fatal**: the image still ships, just without
# baked OCR weights, and the server downloads them on first container start
# (adds ~10-15s to initial boot, invisible to end users thereafter).
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH:-} \
    CUDA_VISIBLE_DEVICES="" \
    python - <<'PY'
import sys, traceback

# Non-fatal prewarm: on any failure, print the traceback as a warning and
# exit 0 so the build continues. The first container start will then
# download OCR weights (~10-15s) via server.py's _startup on the user's
# actual GPU, where libcuda.so.1 is bind-mounted by the NVIDIA runtime.

try:
    import paddle
    print(f"paddle {paddle.__version__}  compiled_with_cuda={paddle.is_compiled_with_cuda()}", flush=True)
except Exception:
    print("[WARN] paddle import failed in builder — skipping OCR prewarm. Image still ships.", flush=True)
    traceback.print_exc()
    sys.exit(0)

try:
    from paddleocr import PaddleOCR
    PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
        lang="en",
        device="cpu",
    )
    print("PaddleOCR cached into /opt/paddlex_cache", flush=True)
except Exception:
    print("[WARN] PaddleOCR prewarm failed in builder — skipping. Image still ships.", flush=True)
    traceback.print_exc()
    sys.exit(0)
PY


# ---- runtime stage ----------------------------------------------------------
# -runtime base (~2 GB) instead of -devel (~5 GB) since torch and paddle wheels
# ship their own bundled nvidia-* CUDA libs; nvcc and CUDA headers are not
# needed at runtime.
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04 AS runtime
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# Model + cache locations baked into the image layer.
ENV HF_HOME=/opt/hf_cache \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PADDLE_PDX_CACHE_HOME=/opt/paddlex_cache

# Runtime tunables (all env-overridable at `docker run` time).
ENV NLLB_MODEL_SIZE=1.3B-distilled \
    NLLB_MODEL_ID="" \
    TP_SIZE=auto \
    DTYPE=fp16 \
    MAX_BATCH_SIZE=32 \
    MAX_INPUT_LENGTH=1024 \
    MAX_DOC_BYTES=26214400 \
    MAX_PDF_PAGES=50 \
    OCR_DPI=200 \
    OCR_LANG=en \
    HOST=0.0.0.0 \
    PORT=8005 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0"

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        python3.11 \
        ca-certificates \
        curl \
        libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3
# libgl1 + libglib2.0-0: required at runtime too, because server.py's
# _startup imports `from paddleocr import PaddleOCR`, which transitively
# loads opencv -> cv2 -> libGL.so.1. Without these, the runtime container
# starts but `_ocr_engine` is left None and image / scanned-PDF requests
# return 503. Text + DOCX + text-layer PDF still work in that case, but
# the image would no longer be fully self-contained for OCR paths.

COPY --from=builder /opt/venv         /opt/venv
COPY --from=builder /opt/hf_cache     /opt/hf_cache
COPY --from=builder /opt/paddlex_cache /opt/paddlex_cache
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY server.py /app/server.py
COPY start.sh  /app/start.sh
COPY static    /app/static
RUN chmod +x /app/start.sh

EXPOSE 8005
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT:-8005}/health" || exit 1

CMD ["/app/start.sh"]
