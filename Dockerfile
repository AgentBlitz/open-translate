FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 AS builder
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    python3.11 python3-pip python3.11-venv \
    ca-certificates curl \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# venv for clean copy into runtime
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Document-translation + OCR stack.
# PaddlePaddle 3.0 ships cu126/cu129 wheels but not cu128. The cu126 wheel
# runs correctly on the CUDA 12.8.1 base thanks to CUDA minor-version forward
# compatibility (verified on RTX 5090 / Blackwell sm_120).
RUN pip install --no-cache-dir \
        paddlepaddle-gpu==3.0.0 \
        -i https://www.paddlepaddle.org.cn/packages/stable/cu126/ && \
    pip install --no-cache-dir \
        paddleocr==3.0.0 \
        pypdfium2==4.30.0 \
        python-docx==1.1.2 \
        pypdf==5.1.0 \
        python-multipart==0.0.20 \
        Pillow==11.0.0

# Pre-warm PaddleOCR weights into the image so first-request latency stays low.
ENV PADDLE_PDX_CACHE_HOME="/opt/paddlex_cache"
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=True, lang='en', device='cpu')"


FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 AS runtime
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Hugging Face cache
ENV HF_HOME="/workspace/.cache/huggingface/"
ENV HF_HUB_ENABLE_HF_TRANSFER="1"

# NLLB variables
ENV NLLB_MODEL_SIZE="1.3B-distilled" \
    NLLB_MODEL_ID="" \
    TP_SIZE="auto" \
    DTYPE="fp16" \
    MAX_BATCH_SIZE="32" \
    MAX_INPUT_LENGTH="10000" \
    MAX_DOC_BYTES="26214400" \
    MAX_PDF_PAGES="50" \
    OCR_DPI="200" \
    OCR_LANG="en" \
    PADDLE_PDX_CACHE_HOME="/workspace/.cache/paddlex" \
    PORT="8000" \
    HOST="0.0.0.0" \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0"

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    python3.11 \
    ca-certificates \
    curl && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/paddlex_cache /opt/paddlex_cache
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /
COPY server.py /server.py
COPY start.sh /start.sh
COPY static /static
RUN chmod +x /start.sh

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT:-8000}/health" || exit 1

CMD ["/start.sh"]
