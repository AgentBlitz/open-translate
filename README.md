# 🌍 `open-translate`

[![GHCR](https://img.shields.io/badge/ghcr.io-agentblitz%2Fopen--translate-2088FF?style=for-the-badge&logo=github)](https://github.com/AgentBlitz/open-translate/pkgs/container/open-translate)

> **A high-performance, self-hostable translation API compatible with Google Cloud Translate.**
> Built on Meta's **NLLB-200** and optimized with **DeepSpeed** for efficient GPU inference.

---

## ✨ Why?

This project provides a robust, private, and cost-effective alternative to commercial translation APIs.

* **💰 Cost Efficiency:** Run on your own GPU infrastructure. Ideal for high-volume translation tasks.
* **🔒 Data Privacy:** No external API calls mean your content never leaves your control.
* **🔄 Drop-in Compatibility:** Implements the standard `POST /language/translate/v2` API surface. Switch existing applications simply by changing the base URL.
* **🌍 Advanced Models:** Leverages Meta's [NLLB-200 (No Language Left Behind)](https://arxiv.org/abs/2207.04672), supporting 200+ languages.
* **🚀 High Performance:** Optimized for throughput with DeepSpeed and Tensor Parallelism, capable of handling heavy concurrent loads.

---

## ⚡ Drop-in Replacement

Designed to work with existing Google Cloud Translate client libraries and integrations.

**Before:**
`https://translation.googleapis.com/language/translate/v2`

**After:**
`http://localhost:8005/language/translate/v2`

---

## 🚀 Quick Start

### 🐳 Run with Docker

One command, no config, no volume mounts, no HuggingFace account required. The image ships with NLLB-200-distilled-1.3B (~3.3 GB) and PaddleOCR detection + recognition + textline-orientation weights (~500 MB) **pre-baked**, so `docker run` is fully offline after pull. First boot takes ~30 seconds (NLLB model load + DeepSpeed init on your GPU); subsequent restarts are instant. All OCR paths (image uploads, scanned PDFs) work out of the box — no volume mount needed for weight persistence.

```bash
docker run --gpus all -p 8005:8005 ghcr.io/agentblitz/open-translate:latest
```

Then open **http://localhost:8005/ui/**.

**Requirements:**

- NVIDIA GPU with ≥ 5 GB VRAM (5 GB for NLLB-1.3B-distilled in fp16 + 1–2 GB for PaddleOCR)
- NVIDIA Container Runtime installed on the host ([setup guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html))
- Host driver **≥ 570** for Blackwell (RTX 5090), or **≥ 550** for Ada (RTX 4090). The image is CUDA 12.8 and supports `sm_80`, `sm_86`, `sm_89`, `sm_90`, `sm_120`.

**Swap the model size** at run time by setting `NLLB_MODEL_SIZE` (note: only `1.3B-distilled` is baked into the image — any other size forces a HuggingFace download on first boot):

```bash
docker run --gpus all -p 8005:8005 \
  -e NLLB_MODEL_SIZE=3.3B \
  -v open-translate-hf:/opt/hf_cache \
  ghcr.io/agentblitz/open-translate:latest
```

The named volume `open-translate-hf` persists the downloaded weights across container restarts.

---

## 🖥️ Web UI

A minimal Google-Translate-style frontend is served directly by the API at:

```
http://localhost:8005/ui/
```

Two panels (source + target), language dropdowns populated from `/language/translate/v2/languages`, auto-detect, swap, and copy. No build step, no external CDN, no third-party requests.

### Privacy guarantees enforced in code

- **No logging of request or response text** anywhere in `server.py` — the only `print()` calls are startup model/OCR status lines.
- **No disk writes of user content.** Text translations are an in-memory GPU pass. Document uploads (`.docx`, `.pdf`, images) flow end-to-end through `io.BytesIO`: `python-docx` reads from a BytesIO, `pypdf` / `pypdfium2` rasterize PDFs from a BytesIO, images go straight into `PIL.Image.open(io.BytesIO(...))` → NumPy → `PaddleOCR.predict(arr)`, and the translated DOCX is written into another BytesIO and streamed back. There are no `tempfile` calls, no `open(..., "w")`, and no intermediate files under `/tmp`.
- **No translation cache.** There is no LRU / Redis / dict keyed on input text. Every request recomputes from scratch, and the only state that persists between requests is model weights.
- **Response filename is hardcoded** to `translated.docx`. Client-provided filenames are used only for MIME-type classification and never echoed back in headers or logs.
- **Access logs disabled** — `start.sh` launches uvicorn with `--no-access-log` so request lines never reach stdout (and therefore never reach Docker/journald log files).
- **`GET /language/translate/v2` and `GET /language/translate/v2/detect` have been removed.** Only POST is accepted. This guarantees the text never appears in a URL, which would otherwise be captured by proxies, browser history, and access logs.
- Every response carries `Cache-Control: no-store, no-cache, must-revalidate`, `Pragma: no-cache`, and `Referrer-Policy: no-referrer` via middleware, so browsers and intermediate proxies cannot cache translations to disk.
- UI uses `cache: 'no-store'`, `credentials: 'same-origin'`, and stores only language-code preferences (never text or filenames) in `localStorage`. Document downloads go via `URL.createObjectURL()` + immediate `revokeObjectURL()` — no OCR preview is ever rendered in the browser. `same-origin` credentials mode is required for deployments behind an auth-protecting reverse proxy (e.g. Vast.ai's caddy wrapper); the cookie is scoped to the server's own origin and never leaks to third parties.

### Operator responsibilities (the code cannot enforce these)

- **OS swap:** disable swap (`swapoff -a`) or use an encrypted swap device — otherwise memory pages holding in-flight text may be written to disk by the kernel.
- **Log drivers:** ensure your Docker/systemd log driver is not capturing stdout to an unexpected sink. With `--no-access-log`, nothing containing user text is printed, but startup output and tracebacks still are.
- **Reverse proxies:** do not front the service with a proxy that logs request bodies (nginx `access_log` with body capture, Cloudflare logging, etc.).

> **Honest caveat:** Python strings are immutable and garbage-collected; there is no reliable way to zeroize request text in memory before GC. "Irrecoverable deletion" here means **never written to non-volatile storage in the first place**, which is what the hardening above achieves.

---

## 🛠️ API Reference

Compatible with **Google Cloud Translation API v2**.

### Translate Text

**POST** `/language/translate/v2`

**Single Translation:**

```bash
curl -X POST "http://localhost:8005/language/translate/v2" \
  -H "Content-Type: application/json" \
  -d '{
    "q": "Hello world!",
    "target": "es"
  }'
```

**Batch Translation:**
Send arrays of strings to maximize GPU throughput.

```bash
curl -X POST "http://localhost:8005/language/translate/v2" \
  -H "Content-Type: application/json" \
  -d '{
    "q": ["Hello world!", "Self hosting rulez"],
    "target": "fr",
    "source": "en",
    "max_new_tokens": 128
  }'
```

### Language Detection

**POST** `/language/translate/v2/detect`

```bash
curl -X POST "http://localhost:8005/language/translate/v2/detect" \
  -H "Content-Type: application/json" \
  -d '{"q": "Hola mundo"}'
```

### Translate a Document or Image

**POST** `/language/translate/v2/document`

Upload a document or image and receive a translated `.docx` back. The whole pipeline runs in-memory on the GPU — no temp files, no disk writes of user content.

**Supported inputs:** `.docx`, `.pdf` (text-layer and scanned), `.jpg` / `.jpeg`, `.png`, `.webp`, `.bmp`, `.tif` / `.tiff`.

Scanned PDFs and images are OCR'd with PaddleOCR 3.x on the same GPU as NLLB; text-layer PDFs and `.docx` inputs skip OCR entirely.

```bash
curl -X POST "http://localhost:8005/language/translate/v2/document" \
  -F "file=@/path/to/input.pdf" \
  -F "target=es" \
  -F "source=en" \
  -o translated.docx
```

Limits (env-tunable): `MAX_DOC_BYTES` (default **25 MB**), `MAX_PDF_PAGES` (default **50**), `OCR_DPI` (default **200**), `OCR_LANG` (default **en**). The response filename is always `translated.docx` — the client-provided filename is used only for MIME dispatch and never echoed back, for privacy.

### List Supported Languages

**GET** `/language/translate/v2/languages`

```bash
curl "http://localhost:8005/language/translate/v2/languages"
```

---

## ⚙️ Configuration

| Variable | Default | Description |
| :--- | :--- | :--- |
| `NLLB_MODEL_SIZE` | `1.3B-distilled` | Model size: `600M`, `600M-distilled`, `1.3B`, `1.3B-distilled`, or `3.3B` |
| `NLLB_MODEL_ID` | *(None)* | HF model override |
| `TP_SIZE` | `auto` | Tensor Parallel size |
| `DTYPE` | `fp16` | `fp16`, `bf16`, or `fp32` |
| `MAX_BATCH_SIZE` | `32` | Max sentences processed in parallel |
| `HOST` | `0.0.0.0` | Bind host |
| `PORT` | `8005` | Bind port |
| `HF_HOME` | `/opt/hf_cache` | HuggingFace cache dir (weights baked in at this path) |
| `PADDLE_PDX_CACHE_HOME` | `/opt/paddlex_cache` | PaddleOCR cache dir (weights baked in at this path) |
| `MAX_DOC_BYTES` | `26214400` | Max upload size for document translation (25 MB) |
| `MAX_PDF_PAGES` | `50` | Max pages processed per PDF upload |
| `OCR_DPI` | `200` | Raster DPI for scanned PDFs before OCR |
| `OCR_LANG` | `en` | PaddleOCR language model |

---

## 🌐 Language Codes

We support standard **ISO 639-1** (e.g., `es`, `en`) and **BCP-47** (e.g., `zh-TW`, `pt-BR`) codes, automatically mapping them to NLLB's internal representation.

For a full list of over 200 supported languages and their codes, see **[LANGUAGES.md](./LANGUAGES.md)**.

---

## 💾 VRAM Requirement Guide

| Model Size      | FP16 / BF16 | FP32    |
|-----------------|-------------|---------|
| `600M` / `600M-distilled` | ~3 GB      | ~5 GB  |
| `1.3B` / `1.3B-distilled` | ~5 GB      | ~9 GB |
| `3.3B`           | ~9 GB     | ~15 GB |
