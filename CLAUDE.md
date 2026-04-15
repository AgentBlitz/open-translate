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
- **UI uses POST only**, `cache: 'no-store'`, `credentials: 'same-origin'`, stores only language codes in `localStorage` (never text or filenames). The Documents tab downloads via `URL.createObjectURL` + immediate `revokeObjectURL` — no intermediate OCR preview is ever rendered in the browser. **Note:** `credentials: 'same-origin'` (was `'omit'` until 2026-04-15) is required so that when the UI is served behind Vast's caddy auth wrapper, the `C.<instance>_auth_token` cookie flows with same-origin fetches. Same-origin means the cookie never leaks to any other host.

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
| [.github/workflows/docker-publish.yml](.github/workflows/docker-publish.yml) | Builds and pushes to `ghcr.io/agentblitz/open-translate` on image-relevant pushes to main (Dockerfile, requirements.txt, server.py, start.sh, static/ui/**) or manual `workflow_dispatch`. |

## Local development

```bash
# Create venv (Python 3.11+)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./start.sh   # serves on 0.0.0.0:8005 by default
```

Open http://localhost:8005/ui/. Note that CPU-only inference works but is slow; GPU strongly recommended.

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

## Deploying anywhere via the published GHCR image (recommended primary path)

[docker-publish.yml](.github/workflows/docker-publish.yml) builds and publishes `ghcr.io/agentblitz/open-translate:latest` + `:<sha>` to GitHub Container Registry on every push that touches `Dockerfile`, `requirements.txt`, `server.py`, `start.sh`, `static/ui/**`, or the workflow itself (see the `paths:` filter). Doc-only pushes do not trigger a rebuild. Manual triggers via the Actions tab `workflow_dispatch` button also work.

The image is **fully self-contained**: NLLB-200-distilled-1.3B (~3.3 GB) and PaddleOCR detection+recognition+textline-orientation weights (~500 MB) are baked into `/opt/hf_cache` and `/opt/paddlex_cache` at build time, so `docker run` requires no network access at boot and no volume mounts. One command to run it anywhere with an NVIDIA GPU:

```bash
docker run --gpus all -p 8005:8005 ghcr.io/agentblitz/open-translate:latest
```

Then open `http://localhost:8005/ui/`.

**Host requirements**: NVIDIA Container Runtime + host driver ≥ 570 for Blackwell (RTX 5090 / sm_120) or ≥ 550 for Ada (RTX 4090 / sm_89). `TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0"` is baked into the image, covering Ampere through Blackwell.

**Environment overrides** (all optional):

- `NLLB_MODEL_SIZE` — switch model. Only `1.3B-distilled` is baked; other sizes force a HF download on first boot (mount a volume at `/opt/hf_cache` to persist it).
- `DTYPE=fp16|bf16|fp32`, `MAX_BATCH_SIZE`, `MAX_INPUT_LENGTH`, `MAX_DOC_BYTES`, `MAX_PDF_PAGES`, `OCR_DPI`, `OCR_LANG`, `HOST`, `PORT`.

**HTTPS / auth**: the container serves plain HTTP on 8005. Front it with your own reverse proxy (caddy, nginx, traefik, a cloud LB) if you need TLS termination or auth. The Vast deployment demonstrates one such pattern — see the "Raw-IP access via caddy" section below.

**GHCR package visibility**: if the first successful build publishes as **private**, flip to public via **https://github.com/orgs/AgentBlitz/packages** → `open-translate` → Package settings → Danger Zone → Change visibility → Public. Also **Connect repository** → `AgentBlitz/open-translate` so the README renders on the GHCR page.

## Deploying on RunPod (legacy path, kept for reference)

Before the self-contained GHCR image existed, the RunPod path was a source install via `rsync` + `scripts/runpod-setup.sh`. That still works but the GHCR image makes it redundant — create a RunPod pod using `ghcr.io/agentblitz/open-translate:latest` as the container image directly, expose port 8005, done.

## Building the Docker image

The Dockerfile is **Blackwell-ready** (CUDA 12.8.1 base, cu128 torch wheels, `TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0"`).

**CI (recommended):** push a change to any image-relevant file to `main`; [docker-publish.yml](.github/workflows/docker-publish.yml) builds on GitHub's amd64 runners and pushes to `ghcr.io/agentblitz/open-translate`. The workflow frees ~5 GB of runner disk first because the full build needs ~13 GB transiently (builder stage + runtime stage + layer cache).

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
- **Vast.ai (CUDA 13.1) needs Paddle cu129 ≥ 3.2.2.** The cu129 index (`https://www.paddlepaddle.org.cn/packages/stable/cu129/`) no longer ships `paddlepaddle-gpu==3.0.0` — lowest is 3.1.0. 3.2.2 is the tested pin. It overwrites torch's `nvidia-*-cu12` 12.8.x libs with 12.9.x equivalents; torch still reports `cuda 12.8` and detects `sm_120`, so this is cosmetic but worth watching on future bumps.
- **Do not pin `paddleocr==3.0.0` on Python 3.12.** Its transitive `paddlex[ocr]==3.0.0` tries to build an old pandas from source, and pip's isolated build env is missing `pkg_resources`, which aborts install with `ModuleNotFoundError`. `--no-build-isolation` does not help. Fix: install `paddleocr` unpinned → resolves to 3.4.x (all-wheel). The `PaddleOCR(use_doc_orientation_classify=…, use_doc_unwarping=…, use_textline_orientation=…, device=…)` constructor, `.predict(np_array)` call, and `rec_texts` result shape are all unchanged, so `server.py`'s `_ocr_image()` works on both 3.0.x and 3.4.x with no code changes.
- **`docker build` of paddlepaddle-gpu fails without a CUDA driver stub.** Paddle 3.x unconditionally `dlopen`s `libcuda.so.1` at `import paddle` time to probe the host driver, even when `device='cpu'` is requested later. At `docker build` time there is no host driver bind-mounted, so the stub library shipped in `-devel` CUDA base images (`/usr/local/cuda/lib64/stubs/libcuda.so`) needs to be symlinked to `libcuda.so.1` and added to `LD_LIBRARY_PATH` for the prewarm RUN. See [Dockerfile](Dockerfile)'s paddle prewarm step — the block is scoped (env vars don't leak into the runtime stage) and intentionally **non-fatal**: on any import or init failure, the Python script catches the exception, logs the full traceback, and exits 0 so the image still ships. In the fallback path, `/opt/paddlex_cache` ships empty and `server.py`'s `_startup` downloads OCR weights on first container run (~10s cold start). Do not re-enable the strict behaviour — it blocks CI.
- **PaddleOCR 3.0 API differs from 2.x.** Use `PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=True, lang=..., device='gpu')` and call `.predict(numpy_array)`; results expose `rec_texts` on each element. The old `use_angle_cls`/`.ocr()` API is gone.
- **PaddleOCR weights cache** lives at `$PADDLE_PDX_CACHE_HOME` (default `/workspace/.cache/paddlex` on RunPod, `/opt/paddlex_cache` in the Docker image). Mount `/workspace` as a persistent volume so both the HF and Paddle caches survive pod restarts.
- **VRAM budget.** NLLB-1.3B-distilled ≈ 3.3 GB, PaddleOCR det+rec ≈ 1–2 GB, total ~4–6 GB. Comfortable on the 5090's ~32 GB. If OOM appears under concurrent load, cap uvicorn with `--limit-concurrency 4` in [start.sh](start.sh) — do not disable the GPU path.

## Verification recipe

After any deploy, run these from a shell that can reach the server (locally, via SSH tunnel, or via the RunPod proxy URL):

```bash
BASE="http://localhost:8005"   # or the pod proxy URL, or the caddy-fronted HTTPS URL

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

## Deploying on Vast.ai (PyTorch template, current primary path as of 2026-04-15)

Vast.ai has replaced RunPod as the day-to-day deploy target for this project. Full recipe lives in [.planning/vast-deploy-plan.md](.planning/vast-deploy-plan.md); the short version:

- Template: **Vast PyTorch** on Ubuntu 24.04, Python 3.12, **no torch pre-installed**, 120 GB rootfs, `HF_HOME=/workspace/.hf_home`, `WORKSPACE=/workspace`.
- Port **8005** (NOT 8080 — `PORTAL_CONFIG` permanently maps 8080 → the Vast Jupyter server on this template). Tunnel from the Mac with `ssh -p <port> root@<host> -L 8080:localhost:8005` so the browser URL stays `http://localhost:8080/ui/`.
- Torch: `torch==2.9.1` with `--extra-index-url https://download.pytorch.org/whl/cu128` (plain `python3 -m venv`, **not** `--system-site-packages` — there is nothing to inherit).
- Paddle: **`paddlepaddle-gpu==3.2.2`** from `https://www.paddlepaddle.org.cn/packages/stable/cu129/`. The cu129 index no longer ships `3.0.0`.
- PaddleOCR: **unpinned** (resolves to 3.4.x). **Do not pin `paddleocr==3.0.0`** on Python 3.12 — its transitive `paddlex[ocr]==3.0.0` tries to build an old pandas from source and fails with `ModuleNotFoundError: pkg_resources` inside pip's isolated build env. PaddleOCR 3.4.x keeps the same `PaddleOCR(...).predict(np_array)` / `rec_texts` API, so [server.py](server.py)'s `_ocr_image()` works unchanged.
- Paddle 3.2.x overwrites torch's `nvidia-*-cu12` 12.8.x libs with 12.9.x equivalents. In practice this is benign — torch still reports `cuda 12.8` and detects `sm_120`. If a future paddle bump breaks generation, pin `paddlepaddle-gpu==3.2.2` and reinstall torch last.
- Launch: `PORT=8005 PADDLE_PDX_CACHE_HOME=/workspace/.cache/paddlex bash scripts/runpod-launch.sh start` (the RunPod launcher is environment-agnostic — no rename needed).

### Raw-IP access via caddy (added 2026-04-15)

If trycloudflare.com is blocked on your network **and** port 8005 wasn't declared at instance creation (so no direct NAT entry exists), you cannot reach `http://<public-ip>:<mapped-port>/` directly. Workaround: ride on top of Vast's own caddy auth wrapper, which already reverse-proxies the **Tensorboard** port slot to an empty internal port.

The chain is: public `212.13.234.30:<VAST_TCP_PORT_6006>` → caddy `:6006` → `reverse_proxy localhost:16006` (defined in `/etc/Caddyfile` by Vast's portal-aio). Nothing normally listens on `localhost:16006` except the tensorboard placeholder.

Recipe:

1. `supervisorctl stop tensorboard` — frees `127.0.0.1:16006`.
2. Launch uvicorn with `HOST=127.0.0.1 PORT=16006` (not `0.0.0.0` — keeps it behind caddy, no direct path).
3. Access via `http://<public-ip>:<VAST_TCP_PORT_6006>/ui/?token=<instance-auth-token>`. The token is in `/etc/Caddyfile` (search for `C.<instance>_auth_token`) and sets a 7-day cookie on first load. Same token works as `Authorization: Bearer` for API clients, or use basic auth with username `vastai` + the Vast instance password from the web console.
4. **Required UI change** already applied: `static/ui/app.js` uses `credentials: 'same-origin'` (not `'omit'`) so the caddy auth cookie flows on `fetch()`. Otherwise the UI shows "Failed to load languages" because the `/language/translate/v2/languages` call gets a 401 from caddy.

**Enabling HTTPS on the raw-IP URL** (added 2026-04-15, same session): Vast's boot sequence (`/etc/vast_boot.d/55-tls-cert-gen.sh`) pre-generates `/etc/instance.key` and `/etc/instance.crt`, signed by "Vast.ai Jupyter CA" with SAN `IP:<public-ip>`. Re-use that cert by adding one line to the `:6006` Caddyfile block:

```
:6006 {
    tls /etc/instance.crt /etc/instance.key
    ... (everything else unchanged)
}
```

Then `/opt/portal-aio/caddy_manager/caddy reload --config /etc/Caddyfile`. The port now speaks HTTPS with a real (if CA-untrusted) cert; plain HTTP requests to the same port return 400. Browsers show a one-time "Not Secure" warning because the Vast CA isn't public, but the IP SAN matches so there's no additional hostname-mismatch error. The cert survives pod restart (it's baked into `/etc/instance.crt`), but the **Caddyfile edit does not** — `caddy_config_manager.py` may regenerate the file on reboot. After a pod restart, re-apply the one-line patch and reload caddy.

Trade-offs:
- Tensorboard supervisor stays stopped (not used on this project anyway).
- Clients behind a network that blocks trycloudflare but allows raw TCP to arbitrary high ports can now reach the service over **HTTPS** on `https://<public-ip>:<VAST_TCP_PORT_6006>/ui/` (after the one-time CA warning).
- All traffic still goes through caddy, which adds its own request-body size limit (default generous, but worth watching for 25 MB document uploads on first use).
- Caddyfile patch is not persistent across pod recreation — document the one-liner in the raw-IP recipe above and reapply after each pod rebuild.

## Known live deployments

- **Vast.ai** pod on RTX 5090 (Blackwell sm_120), torch 2.9.1+cu128, paddle 3.2.2, paddleocr 3.4.1.
  - SSH: `ssh -p 1431 root@212.13.234.30`
  - **Direct raw-IP URL (current primary, HTTPS)**: `https://212.13.234.30:1965/ui/?token=56620e1fa2fb9c208f6a1fa29a4324007b684e18cdedad7f49a15b6f4ff39bb7` (caddy → `localhost:16006`, uvicorn bound to `HOST=127.0.0.1 PORT=16006`, tensorboard supervisor stopped to free 16006). Caddy's `:6006` vhost serves TLS using the pre-signed cert at `/etc/instance.crt` + `/etc/instance.key` (SAN `IP:212.13.234.30`, issuer `Vast.ai Jupyter CA`). Browsers show a one-time "Not Secure" warning because the Vast CA isn't in the public trust store; click through once and the 7-day auth cookie persists. After the first visit the `?token=` query parameter can be dropped. Plain HTTP on the same port now returns 400.
  - Fallback via SSH tunnel: `ssh -p 1431 root@212.13.234.30 -L 8080:localhost:8005` → `http://localhost:8080/ui/` (only if the server is relaunched with `PORT=8005 HOST=0.0.0.0`, currently not the default).
  - API clients must send `Authorization: Bearer 56620e1fa2fb9c208f6a1fa29a4324007b684e18cdedad7f49a15b6f4ff39bb7` when hitting the raw-IP URL — caddy rejects unauthenticated requests.
  - The bearer token rotates when the Vast container is recreated; grep `/etc/Caddyfile` for `auth_token` after a pod reset to get the new value.
  - Rootfs is the only persistent storage (no attached volume), but 120 GB gives comfortable headroom. HF + Paddle caches live under `/workspace`.
  - Document + image translation verified end-to-end on this pod (DOCX round-trip + PNG OCR → translated DOCX).
- **RunPod** pod on RTX 5090 (Blackwell sm_120), torch 2.8.0+cu128, deepspeed 0.18.3 — previous primary, kept as a fallback.
  - `https://jz58e3ujkosmqr-8000.proxy.runpod.net/ui/`
  - Access: `ssh -p 46718 -i ~/.ssh/id_ed25519 root@149.36.1.202` (host + port rotate when pod is recreated).
  - `/workspace` was NOT mounted as a persistent volume on this instance — state is lost on pod recreation. Next pod should mount a 30 GB volume to avoid re-downloading the model.
