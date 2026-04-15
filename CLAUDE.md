# CLAUDE.md

Project context for Claude (and developers). Keep this file current when architecture, deployment, or gotchas change.

## What this is

A self-hostable translation API + Google-Translate-style web UI, compatible with the Google Cloud Translation v2 API surface. Runs Meta's NLLB-200 via HuggingFace Transformers + DeepSpeed kernel-inject on a single GPU. Built for **privacy-first deployments**: user text is never written to disk or logged on the server.

## Architecture

- **FastAPI** app in [server.py](server.py), single file.
- **NLLB-200** (default `facebook/nllb-200-distilled-1.3B`) loaded at startup via Transformers.
- **DeepSpeed** `init_inference` with `replace_with_kernel_inject=True` for GPU kernel acceleration, wrapped in a try/except that falls back to plain HF `model.to("cuda")` if kernel injection fails (important for Blackwell / sm_120 where DeepSpeed kernels may lack support).
- **PaddleOCR 3.0** loaded at startup alongside NLLB for document/image translation. Runs on the same GPU (shares VRAM with NLLB; ~1–2 GB extra). Guarded by its own `_OCR_LOCK` so OCR and NLLB generation can pipeline. Fallback: if `PaddleOCR` import/init raises, `_ocr_engine` is left `None` and the document endpoint rejects image/scanned-PDF inputs while still serving `.docx` and text-layer PDFs.
- **Static UI** at `/ui/` served via `StaticFiles` mount from [static/ui/](static/ui/) — plain HTML/CSS/JS, no build step, no CDN, no framework. Tabs: **Text** (existing two-panel translate) and **Documents** (drag-and-drop upload → translated `.docx` download).
- **Response middleware** sets `Cache-Control: no-store, no-cache, must-revalidate`, `Pragma: no-cache`, `Referrer-Policy: no-referrer` on every response.
- **Model access is thread-safe** via `_MODEL_LOCK`; batched inference up to `MAX_BATCH_SIZE` (default 32), max input length 1024 tokens.

### API surface (all POST — GET variants deliberately removed)

- `POST /language/translate/v2` — translate a string or batch. Body: `{"q": ["text"], "target": "es", "source": "en"?}`.
- `POST /language/translate/v2/detect` — language detection via `langdetect`.
- `POST /language/translate/v2/document` — multipart upload of a document or image; returns a translated `.docx`. Fields: `file` (required), `target` (required), `source` (optional). Supported inputs: `.docx`, `.pdf` (text-layer and scanned), `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.tif`, `.tiff`. Size cap `MAX_DOC_BYTES` (default 25 MB), page cap `MAX_PDF_PAGES` (default 50), OCR raster DPI `OCR_DPI` (default 200).
- `GET  /language/translate/v2/languages` — list supported languages (safe: no user text in URL).
- `GET  /health` — status, model id, CUDA/GPU info.
- `GET  /ui/` — web UI (only if `static/ui/` exists; mount is guarded).

## Privacy guarantees (enforced in code)

- **No logging of request/response text** anywhere in `server.py`.
- **No disk writes of user text** — inference is in-memory GPU only; HF and PaddleOCR caches hold **weights**, not inputs.
- **Document translation is fully in-memory** — uploads flow through `io.BytesIO`, PDFs rasterize via `pypdfium2` (no system libs, no temp files), images go straight to `PIL.Image` → NumPy → PaddleOCR. The endpoint never calls `tempfile`, never writes the upload to disk, and hardcodes the response filename to `translated.docx` (client-provided filenames are used only for MIME dispatch, never echoed).
- **GET translate/detect endpoints removed** so text can never appear in a URL (which would be captured by proxies, browser history, access logs).
- **uvicorn `--no-access-log`** in [start.sh](start.sh) — request lines never reach stdout / container log driver.
- **No-store / no-cache / no-referrer headers** on every response.
- **UI uses POST only**, `cache: 'no-store'`, `credentials: 'omit'`, stores only language codes in `localStorage` (never text or filenames). The Documents tab downloads via `URL.createObjectURL` + immediate `revokeObjectURL` — no intermediate OCR preview is ever rendered in the browser.

**Operator responsibilities not enforceable in code:** disable or encrypt OS swap; ensure the container log driver doesn't capture stdout somewhere unexpected; don't front with a reverse proxy that logs request bodies. See [README.md](README.md#privacy-guarantees-enforced-in-code) for the honest caveat about Python memory GC.

## Key files

| Path | Role |
|---|---|
| [server.py](server.py) | FastAPI app, routing, model lifecycle, middleware, UI mount |
| [static/ui/index.html](static/ui/index.html) | Two-panel UI layout |
| [static/ui/app.js](static/ui/app.js) | Debounced auto-translate, swap/copy/detect, `localStorage` for language codes only |
| [static/ui/style.css](static/ui/style.css) | Google-Translate-style two-card layout, responsive |
| [start.sh](start.sh) | Uvicorn launcher with `--no-access-log` |
| [Dockerfile](Dockerfile) | Multi-stage CUDA 12.8.1 build, cu128 torch wheels, `TORCH_CUDA_ARCH_LIST` includes 12.0 (Blackwell) |
| [requirements.txt](requirements.txt) | Pinned deps (torch 2.9.1, transformers, deepspeed 0.18.3) |
| [scripts/runpod-setup.sh](scripts/runpod-setup.sh) | One-shot installer for RunPod PyTorch pods |
| [scripts/runpod-launch.sh](scripts/runpod-launch.sh) | start / stop / restart / status for the server on RunPod |
| [.github/workflows/docker-publish.yml](.github/workflows/docker-publish.yml) | Builds and pushes to Docker Hub on every push to main |

## Local development

```bash
# Create venv (Python 3.11+)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./start.sh   # serves on 0.0.0.0:8000 by default
```

Open http://localhost:8000/ui/. Note that CPU-only inference works but is slow; GPU strongly recommended.

## Deploying on RunPod (source install, current primary path)

**Template:** `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (CUDA 12.8.1 + torch 2.8.0+cu128 + Ubuntu 24.04). Works for RTX 4090 (Ada sm_89) and RTX 5090 (Blackwell sm_120).

**Pod config at creation:**
- GPU: RTX 4090 / RTX 5090 / etc.
- **Volume: 30 GB mounted at `/workspace`** — critical, otherwise repo + venv + ~11 GB HF cache are lost on restart.
- Expose HTTP port **8000** (RunPod gives you a `https://<pod-id>-8000.proxy.runpod.net/ui/` URL).
- Environment: `HF_HOME=/workspace/.cache/huggingface`.

**First-time setup from local Mac:**

```bash
# 1. rsync the working tree (exclude secrets + local state)
rsync -az \
  --exclude='.env' --exclude='.env.*' \
  --exclude='.planning/' --exclude='__pycache__/' --exclude='*.pyc' \
  --exclude='.DS_Store' --exclude='.claude/' --exclude='.venv/' --exclude='venv/' \
  -e "ssh -p <SSH_PORT> -i ~/.ssh/id_ed25519" \
  ./ root@<POD_HOST>:/workspace/open-translate/

# 2. Create a venv that inherits the pod's pre-installed torch, install deps, launch
ssh -p <SSH_PORT> -i ~/.ssh/id_ed25519 root@<POD_HOST> 'bash -s' <<'REMOTE'
  cd /workspace/open-translate
  git config --global --add safe.directory /workspace/open-translate
  [ -d /workspace/venv ] || python3 -m venv --system-site-packages /workspace/venv
  source /workspace/venv/bin/activate
  # Strip torch pins so we keep the pod's 2.8.0+cu128 instead of forcing torch 2.9.1
  grep -v -E '^(torch|torchvision|torchaudio)([=<>!].*)?$' requirements.txt > /tmp/req.txt
  pip install --no-cache-dir --quiet \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    -r /tmp/req.txt
  bash scripts/runpod-launch.sh start
REMOTE
```

**Subsequent restarts** (after pod stop/start):

```bash
ssh -p <SSH_PORT> -i ~/.ssh/id_ed25519 root@<POD_HOST> \
  'bash /workspace/open-translate/scripts/runpod-launch.sh start'
```

`scripts/runpod-launch.sh` supports `start | stop | restart | status`, writes pid to `server.pid`, logs to `server.log`, polls `/health` before returning.

## Deploying on RunPod (Docker image path, recommended for future)

Once [docker-publish.yml](.github/workflows/docker-publish.yml) has produced an image, skip the rsync/install dance entirely:

1. Set GitHub repo secrets: `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`.
2. Push to `main` → Actions builds and pushes `<username>/open-translate:latest` + `:<sha>`.
3. On RunPod, create a pod using **that published image** as the container image (instead of the RunPod PyTorch template). Expose port 8000, mount a volume at `/workspace/.cache/huggingface` for HF cache persistence. The image's `CMD ["/start.sh"]` auto-launches the server.

## Building the Docker image

The Dockerfile is **Blackwell-ready** (CUDA 12.8.1 base, cu128 torch wheels, `TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0"`).

**CI (recommended):** push to `main`; [docker-publish.yml](.github/workflows/docker-publish.yml) builds on GitHub's amd64 runners and pushes to Docker Hub. The workflow already frees disk first because the image is ~10 GB.

**Manual build on an amd64 Linux host:**

```bash
SHA=$(git rev-parse --short HEAD)
docker build -t open-translate:latest -t open-translate:$SHA .
```

**Manual build on Apple Silicon (slow — qemu x86_64 emulation, 1–3 h):**

```bash
docker buildx build --platform linux/amd64 --push \
  -t <registry>/open-translate:latest .
```

Avoid this path if you can — push to `main` and let Actions do it.

**Cannot build from the running RunPod pod.** RunPod pods are themselves containers with no Docker daemon inside, so `docker build` / `docker commit` are unavailable. The image must be built from the `Dockerfile` on a machine with a working Docker daemon. This is not a problem: the `Dockerfile` already captures everything needed to reproduce the running state, minus the NLLB weights (which are downloaded to the HF cache on first run — that's why persistent volume mounts matter).

## Snapshot / replication strategy

RunPod doesn't offer running-pod snapshots in the traditional VM sense. The replication strategy is:

1. **Source of truth:** this git repo.
2. **Persistent volume** mounted at `/workspace` holds repo clone, venv, and HF model cache (~11 GB for 1.3B-distilled). This survives pod stops/starts.
3. **Published Docker image** (via GH Actions) is the canonical "snapshot" — pull and run anywhere with an NVIDIA GPU + host driver ≥ 570 (Blackwell) or ≥ 550 (Ada).
4. **Custom RunPod template** (future): point a template at the published image, set container start command, expose port 8000. New pod creation = working server in one click.

## Gotchas and lessons learned

- **RTX 5090 / Blackwell (sm_120) needs CUDA 12.8+** — torch 2.8+ with cu128 wheels. Torch 2.7 and earlier, and CUDA 12.6 toolkits, won't emit usable kernels. The `Dockerfile` already uses `nvidia/cuda:12.8.1-devel-ubuntu22.04`.
- **DeepSpeed kernel-inject may not yet target sm_120.** The `try/except` in `server.py` falls back to plain `model.to("cuda")` on failure. In practice on the current pod it's not needed — deepspeed 0.18.3 silently succeeds — but keep the fallback.
- **Ubuntu 24.04 base images enforce PEP 668** (externally-managed-environment). Create a venv with `python3 -m venv --system-site-packages /workspace/venv` so it inherits the base image's pre-installed torch without reinstalling it.
- **Strip `torch*` from `requirements.txt` before `pip install`** on the RunPod pod. Otherwise pip will upgrade to the requirements.txt-pinned torch (2.9.1), wasting ~10 min and risking ABI drift with the pod's pre-built environment.
- **`/workspace` is NOT persistent unless you mount a volume** at creation time. Both pods we tried started with empty `/workspace`. Always mount a 30 GB volume.
- **RunPod pods expose ports only if listed at creation time.** If port 8000 isn't in the "Expose HTTP Ports" list, the proxy URL won't exist. Either edit the pod (may require stop/start) or use an SSH tunnel (`ssh -N -L 8000:localhost:8000 -p <SSH_PORT> ...`) for ad-hoc access.
- **GET translate endpoint is gone.** Any legacy client using `GET /language/translate/v2?q=...` will hit a 405. This was an intentional privacy change; do not restore it.
- **Do NOT commit `.env`.** Confirmed gitignored. A GitHub PAT was briefly visible in `.env` during one session — revoke and rotate if you suspect exposure.
- **PaddlePaddle 3.0 has no cu128 wheel.** The cu126 wheel (installed from `https://www.paddlepaddle.org.cn/packages/stable/cu126/`) runs correctly on the CUDA 12.8.1 base image via CUDA minor-version forward compatibility, confirmed on RTX 5090 / sm_120 with driver ≥ 570. If the cu126 wheel ever breaks on a new pod, `server.py`'s `_startup` catches the import/init failure and leaves `_ocr_engine = None` — the server still serves text + DOCX + text-layer PDFs; image/scanned-PDF requests return 503. Do not switch to CPU paddle silently, because a CPU fallback would be ~10× slower per page with no visible signal in the UI.
- **PaddleOCR 3.0 API differs from 2.x.** Use `PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=True, lang=..., device='gpu')` and call `.predict(numpy_array)`; results expose `rec_texts` on each element. The old `use_angle_cls`/`.ocr()` API is gone.
- **PaddleOCR weights cache** lives at `$PADDLE_PDX_CACHE_HOME` (default `/workspace/.cache/paddlex` on RunPod, `/opt/paddlex_cache` in the Docker image). Mount `/workspace` as a persistent volume so both the HF and Paddle caches survive pod restarts.
- **VRAM budget.** NLLB-1.3B-distilled ≈ 3.3 GB, PaddleOCR det+rec ≈ 1–2 GB, total ~4–6 GB. Comfortable on the 5090's ~32 GB. If OOM appears under concurrent load, cap uvicorn with `--limit-concurrency 4` in [start.sh](start.sh) — do not disable the GPU path.

## Verification recipe

After any deploy, run these from a shell that can reach the server (locally, via SSH tunnel, or via the RunPod proxy URL):

```bash
BASE="http://localhost:8000"   # or https://<pod-id>-8000.proxy.runpod.net

curl -fsS $BASE/health | jq .
curl -fsS "$BASE/language/translate/v2/languages" | jq '.data.languages | length'

# Happy path: POST translate
curl -fsS -X POST "$BASE/language/translate/v2" \
  -H 'Content-Type: application/json' \
  -d '{"q":["Hello world"],"target":"es"}' | jq .

# Privacy: GET form must be gone
curl -s -o /dev/null -w '%{http_code}\n' "$BASE/language/translate/v2?q=hi&target=es"   # expect 405

# Privacy: no-store headers on translate response
curl -s -D - -o /dev/null -X POST "$BASE/language/translate/v2" \
  -H 'Content-Type: application/json' -d '{"q":["hi"],"target":"es"}' \
  | grep -iE '^(cache-control|pragma|referrer-policy)'

# UI loads
curl -fsS "$BASE/ui/" | head -5
```

## Known live deployment (as of last session)

- RunPod pod on RTX 5090 (Blackwell sm_120), torch 2.8.0+cu128, deepspeed 0.18.3.
- Public URL: `https://jz58e3ujkosmqr-8000.proxy.runpod.net/ui/`
- Model: `facebook/nllb-200-distilled-1.3B`, fp16, ~3.3 GB VRAM used.
- Access pod via: `ssh -p 46718 -i ~/.ssh/id_ed25519 root@149.36.1.202` (host + port rotate when pod is recreated).
- `/workspace` was NOT mounted as a persistent volume on this instance — state is lost on pod recreation. Next pod should mount a 30 GB volume to avoid re-downloading the model.
