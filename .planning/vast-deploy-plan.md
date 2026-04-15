# Plan: Deploy open-translate on Vast.ai PyTorch template

## Context

Vast.ai pod (PyTorch template) as a second home for open-translate alongside / replacing the RunPod instance. Pod details captured at plan time (2026-04-15):

- Host: `212.13.234.30`, SSH port `1431`
- GPU: RTX 5090 (Blackwell, sm_120), 32 GB
- Driver: `590.48.01` (well above the `≥570` that Blackwell needs)
- System CUDA: `13.1` (newer than the RunPod box — fine for bundled-runtime wheels)
- OS: Ubuntu 24.04 LTS, Python 3 + `uv` available
- **No torch pre-installed** (the Vast PyTorch template only ships the base env; torch comes from pip)
- Rootfs: **120 GB overlay**; `/workspace` lives on it (no separate persistent volume, but comfortable headroom)
- Pre-set env: `HF_HOME=/workspace/.hf_home`, `WORKSPACE=/workspace`
- **Port 8080 is already taken by the Vast Jupyter server** (`PORTAL_CONFIG` maps localhost:8080 → Jupyter). We run open-translate on internal `8005` and re-tunnel from the Mac with `-L 8080:localhost:8005` so the browser URL stays `http://localhost:8080/ui/`.

SSH command from the Mac (note the updated `-L`):

```bash
ssh -p 1431 root@212.13.234.30 -L 8080:localhost:8005
```

Goal: get the current repo (including the new PaddleOCR document-translation endpoint) running at `http://localhost:8080/ui/` via the SSH tunnel, with HF + Paddle caches persisted under `/workspace`.

## Critical differences from the RunPod recipe

1. **Install torch from scratch** — don't use `--system-site-packages`; the Vast template has nothing to inherit. Keep `torch==2.9.1` from `requirements.txt` with `--extra-index-url https://download.pytorch.org/whl/cu128`.
2. **Paddle on cu129 wheels** — Vast is on CUDA 13.1, so use PaddlePaddle's native `cu129` wheel index instead of the cu126 forward-compat hack. Driver 590 satisfies the cu129 runtime.
3. **Port 8005, not 8080** — 8080 is permanently taken by the Vast Jupyter server on this template. 8005 is also the current repo default (commit b7f6dac). Export `PORT=8005` before launching.
4. **120 GB rootfs** — plenty of room. Full budget including pip cache and NLLB-3.3B would still fit if we ever wanted it. No need to babysit disk.

## Files touched

Source repo: no code changes required — everything from the prior session already handles both environments. Only the **launcher script** and one tiny helper get touched:

| File | Change |
|---|---|
| [scripts/vast-setup.sh](scripts/vast-setup.sh) | **NEW** — first-time installer (git clone or rsync, venv, torch+deps, paddle cu129, warm OCR) |
| [scripts/runpod-launch.sh](scripts/runpod-launch.sh) | Already env-driven — works unchanged with `PORT=8080`, no edit needed |

Keep `runpod-launch.sh` as the shared `start|stop|restart|status` wrapper. Adding a new bespoke vast-launch.sh is unnecessary churn.

## Step 1 — Rsync the working tree (from local Mac)

```bash
rsync -az \
  --exclude='.env' --exclude='.env.*' \
  --exclude='.planning/' --exclude='__pycache__/' --exclude='*.pyc' \
  --exclude='.DS_Store' --exclude='.claude/' --exclude='.venv/' --exclude='venv/' \
  --exclude='server.log' --exclude='server.pid' \
  -e "ssh -p 1431" \
  ./ root@212.13.234.30:/workspace/open-translate/
```

## Step 2 — Remote install + launch

One-shot heredoc over SSH. Creates `/workspace/venv` (no `--system-site-packages`), installs torch cu128 + repo deps, installs paddle cu129 + paddleocr + doc stack, pre-warms OCR weights, launches on :8005 (Jupyter already owns :8080 on this template).

```bash
ssh -p 1431 root@212.13.234.30 'bash -s' <<'REMOTE'
set -euo pipefail

REPO_DIR=/workspace/open-translate
VENV=/workspace/venv

cd "$REPO_DIR"
git config --global --add safe.directory "$REPO_DIR" || true

# 1. venv
if [ ! -f "$VENV/bin/activate" ]; then
  python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"
pip install --upgrade pip setuptools wheel

# 2. main deps (torch 2.9.1 cu128 wheels; rest from pypi)
pip install --no-cache-dir \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -r requirements.txt

# 3. paddle + ocr stack (cu129 wheel, matches CUDA 13 host)
#    NOTE: the cu129 index no longer ships 3.0.0 — lowest is 3.1.0.
#    3.2.2 is the tested version on this pod. Paddle 3.2.x overwrites torch's
#    cu12.8 nvidia-* libs with cu12.9 equivalents; torch 2.9.1+cu128 stays
#    functional (still detects sm_120) but the mismatch warning is cosmetic.
pip install --no-cache-dir \
  paddlepaddle-gpu==3.2.2 \
  -i https://www.paddlepaddle.org.cn/packages/stable/cu129/

# 4. paddleocr (unpinned — resolves to 3.4.x which ships all-wheel).
#    Do NOT pin paddleocr==3.0.0 here: it transitively drags paddlex[ocr]==3.0.0,
#    which tries to build an old pandas from source inside pip's isolated build
#    env and fails with `ModuleNotFoundError: pkg_resources`. paddleocr 3.4.x
#    uses the same PaddleOCR(...).predict(...) API, so server.py works unchanged.
pip install --no-cache-dir \
  paddleocr \
  pypdfium2==4.30.0 \
  python-docx==1.1.2 \
  pypdf==5.1.0 \
  python-multipart==0.0.20 \
  Pillow==11.0.0

# 4. persistent cache dirs (survive inside /workspace)
export PADDLE_PDX_CACHE_HOME=/workspace/.cache/paddlex
mkdir -p "$PADDLE_PDX_CACHE_HOME" "$HF_HOME"

# 5. verify torch + cuda + sm
python - <<'PY'
import torch
p = torch.cuda.get_device_properties(0)
print(f"torch {torch.__version__}  cuda {torch.version.cuda}  avail={torch.cuda.is_available()}")
print(f"  gpu: {p.name}  sm_{p.major}{p.minor}  {p.total_memory/1e9:.1f} GB")
PY

# 6. pre-warm PaddleOCR weights (downloads ~500 MB, one-shot)
python - <<'PY'
from paddleocr import PaddleOCR
PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=True,
    lang="en",
    device="cpu",
)
print("paddleocr: weights cached")
PY

# 7. launch — 8005 because Jupyter owns 8080 on this Vast template.
#    Local tunnel is `ssh -p 1431 root@212.13.234.30 -L 8080:localhost:8005`,
#    so the browser URL stays http://localhost:8080/ui/.
export PORT=8005
export PADDLE_PDX_CACHE_HOME=/workspace/.cache/paddlex
bash scripts/runpod-launch.sh start
REMOTE
```

## Step 3 — Verify

From the Mac, with the SSH tunnel already active (`ssh -p 1431 root@212.13.234.30 -L 8080:localhost:8005`):

```bash
BASE="http://localhost:8080"

curl -fsS $BASE/health | jq .
curl -fsS "$BASE/language/translate/v2/languages" | jq '.data.languages | length'

# Text translate
curl -fsS -X POST "$BASE/language/translate/v2" \
  -H 'Content-Type: application/json' \
  -d '{"q":["Hello world"],"target":"es"}' | jq .

# Document translate (DOCX round-trip)
python3 -c "from docx import Document; d=Document(); d.add_paragraph('Hello world, this is a test.'); d.save('/tmp/t.docx')"
curl -fsS -X POST "$BASE/language/translate/v2/document" \
  -F 'file=@/tmp/t.docx' -F 'target=es' -o /tmp/out.docx
python3 -c "from docx import Document; print('\n'.join(p.text for p in Document('/tmp/out.docx').paragraphs))"

# Image OCR path
curl -fsS -X POST "$BASE/language/translate/v2/document" \
  -F 'file=@/some/screenshot.png' -F 'target=fr' -o /tmp/out.docx
python3 -c "from docx import Document; print('\n'.join(p.text for p in Document('/tmp/out.docx').paragraphs))"

# Privacy: no-store headers
curl -s -D - -o /dev/null -X POST "$BASE/language/translate/v2" \
  -H 'Content-Type: application/json' -d '{"q":["hi"],"target":"es"}' \
  | grep -iE '^(cache-control|pragma|referrer-policy)'
```

UI end-to-end: open `http://localhost:8080/ui/`, confirm both tabs load, drag a PNG into the Documents tab, click Translate, confirm a `.docx` downloads.

## Step 4 — Subsequent restarts

After the first run, daily use is:

```bash
ssh -p 1431 root@212.13.234.30 'PORT=8005 PADDLE_PDX_CACHE_HOME=/workspace/.cache/paddlex bash /workspace/open-translate/scripts/runpod-launch.sh restart'
```

`runpod-launch.sh` already handles pid/log/health polling and is environment-agnostic — no rename needed.

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| 120 GB rootfs fills up mid-install | Very unlikely — NLLB + paddle + venv total is ~23 GB with ~95 GB headroom. Monitor with `df -h /` if worried. |
| Paddle cu129 wheel has a regression on CUDA 13.1 host | Fall back to `-i .../cu126/` (the cu126 wheel is what the Docker image uses and is already known-good on sm_120). Driver 590 satisfies both. |
| Paddle 3.2.x installs nvidia cu12.9 libs, overwriting torch's cu12.8 set | Confirmed benign at deploy time — torch 2.9.1+cu128 still reports `cuda 12.8` and detects sm_120. If generation starts failing after a future paddle bump, pin `paddlepaddle-gpu==3.2.2` and reinstall torch last. |
| paddleocr 3.0.0 cannot install cleanly on Python 3.12 | Its transitive `paddlex[ocr]==3.0.0` tries to build an old pandas from source and the isolated build env is missing `pkg_resources`. Use unpinned `paddleocr` (→3.4.x). Same `.predict()` API. |
| Vast pod is ephemeral, loses state on restart | Repeat Step 1 + Step 2; the heredoc is idempotent. Longer-term: publish the Docker image and use it as the Vast container image instead of the PyTorch template |
| Port collision — 8080 is owned by Jupyter (`PORTAL_CONFIG` maps localhost:8080 → Jupyter) | Confirmed at probe time. We run on 8005 and the Mac-side tunnel maps `-L 8080:localhost:8005`. Do **not** try to reclaim 8080 — it'll break the Vast portal. |

## Critical files to read before executing

- [scripts/runpod-launch.sh](../scripts/runpod-launch.sh) — the start/stop/restart wrapper (reused unchanged)
- [scripts/runpod-setup.sh](../scripts/runpod-setup.sh) — the RunPod source-install script (template for this plan's heredoc, minus the `--system-site-packages` bit and with cu129 paddle)
- [server.py](../server.py) — confirms `OCR_LANG`, `PADDLE_PDX_CACHE_HOME`, `MAX_DOC_BYTES`, `PORT` are all env-driven
- [CLAUDE.md](../CLAUDE.md) — the privacy invariants and VRAM budget notes still apply

## Raw-IP access via caddy (added 2026-04-15, same session)

After the initial Step 2 launch on port 8005, we discovered two things back-to-back:

1. The user's network **blocks trycloudflare.com**, so the Cloudflare Quick Tunnel path (the obvious way to share a Vast service) doesn't work for them.
2. Container port 8005 was **never declared at instance creation**, so `VAST_TCP_PORT_8005` does not exist and no host→container NAT rule maps to it. There is no `http://<public-ip>:<anything>` path that reaches uvicorn on 8005 directly, no matter how it's bound.

The fix: ride on Vast's own caddy auth wrapper. Vast's portal-aio ships a caddy instance listening on container ports 1111, 6006, and 8384, each with a reverse-proxy block to an internal upstream (`localhost:11111`, `localhost:16006`, `localhost:18384`). The `:6006` block is the Tensorboard slot and reverse-proxies to `localhost:16006`, which is only occupied when the `tensorboard` supervisor job is running. Most pods don't actually use Tensorboard.

Recipe:

1. Stop the Tensorboard placeholder to free its internal upstream:
   ```bash
   supervisorctl stop tensorboard
   ```
2. Relaunch open-translate bound to that upstream, on **localhost only** so it stays behind caddy:
   ```bash
   HOST=127.0.0.1 PORT=16006 PADDLE_PDX_CACHE_HOME=/workspace/.cache/paddlex \
     bash scripts/runpod-launch.sh restart
   ```
3. Access via Vast's mapped external port for container 6006 (this pod: `VAST_TCP_PORT_6006=1965`). First-visit URL with the token handshake:
   ```
   http://212.13.234.30:1965/ui/?token=56620e1fa2fb9c208f6a1fa29a4324007b684e18cdedad7f49a15b6f4ff39bb7
   ```
   After the first load, caddy sets a 7-day `C.34991742_auth_token` cookie and the `?token=` query string can be dropped. The same token also works as `Authorization: Bearer <token>` for API clients. Username `vastai` + the Vast instance password from the web console is the basic-auth fallback.

### UI bug this surfaced, and the fix

Going through caddy means every response path requires the auth cookie to flow. The UI's `fetch()` calls were configured with `credentials: 'omit'` for privacy (to avoid carrying ambient cookies), which caused same-origin fetches to **drop the caddy cookie** — the languages dropdown then showed "Failed to load languages" because `/language/translate/v2/languages` returned 401.

Fix: change `credentials: 'omit'` → `credentials: 'same-origin'` in [static/ui/app.js](../static/ui/app.js) (3 fetch call sites). Same-origin means the cookie is sent only to the host that served the page, so it never leaks to any third party — the privacy stance is preserved.

### Header text simplification (same session)

[static/ui/index.html](../static/ui/index.html): dropped the `<h1>open-translate</h1>` brand and replaced the terse privacy footer with a strong, explicit statement that everything is discarded the moment the response finishes. [static/ui/style.css](../static/ui/style.css) lost the now-unused `.topbar h1` rule and `.privacy` was tightened with `max-width: 900px` and `line-height: 1.5` so the longer sentence reads well.

### HTTPS on the raw-IP URL (added 2026-04-15, same session)

Vast's boot sequence runs `/etc/vast_boot.d/55-tls-cert-gen.sh`, which pre-generates `/etc/instance.key` and `/etc/instance.crt` — a real TLS cert signed by "Vast.ai Jupyter CA" with SAN `IP:<public-ip>` (Vast's `sign_cert` API rewrites the template's `0.0.0.0` placeholder to the actual pod IP). Reuse it by adding a single line inside the `:6006` block of `/etc/Caddyfile`:

```
:6006 {
    tls /etc/instance.crt /etc/instance.key
    ...
}
```

Then reload caddy:

```bash
/opt/portal-aio/caddy_manager/caddy validate --config /etc/Caddyfile
/opt/portal-aio/caddy_manager/caddy reload --config /etc/Caddyfile
```

The port now serves HTTPS. The user URL becomes `https://212.13.234.30:1965/ui/?token=<auth_token>`. Browsers show a one-time "Not Secure" warning (the Vast CA is not in the public trust store) but no hostname-mismatch error because the SAN matches. Plain HTTP to the same port returns 400. The fix survives a caddy restart (config file is persistent), but may not survive a **pod recreation** — `caddy_config_manager.py` may regenerate `/etc/Caddyfile` on first boot. After any pod rebuild, reapply the one-liner.

### Trade-offs

- Tensorboard supervisor stays stopped. Not used on this project.
- Raw-IP URL is now **HTTPS** (see above); the CA isn't publicly trusted, so browsers warn once per profile.
- Caddy adds its own reverse-proxy pipeline between the client and uvicorn. Default caddy body size is generous but worth testing on first 25 MB document upload via the public URL.
- The bearer token lives in `/etc/Caddyfile` and rotates when the Vast container is recreated. After a pod restart, `grep -oE '"[a-f0-9]{64}"' /etc/Caddyfile | head -1` fetches the new one.

## Post-deploy notes (2026-04-15)

Executed end-to-end on the target pod. Final working state:

- **torch** 2.9.1+cu128, reports `cuda 12.8`, detects RTX 5090 `sm_120`
- **paddlepaddle-gpu** 3.2.2 (cu129 index) — installs cu12.9 nvidia-* libs on top of torch's cu12.8 set. Harmless in practice; both libraries run.
- **paddleocr** 3.4.1 + **paddlex** 3.4.3 (unpinned → latest all-wheel build). PaddleOCR 3.0.0 constructor arguments (`use_doc_orientation_classify`, `use_doc_unwarping`, `use_textline_orientation`, `device`) and `.predict(np_array)` result shape (`rec_texts`) are unchanged on 3.4.1, so [server.py](../server.py)'s `_ocr_image()` works with no code changes.
- Launched via `PORT=8005 PADDLE_PDX_CACHE_HOME=/workspace/.cache/paddlex bash scripts/runpod-launch.sh start`; `/health` polled OK after 54s.

Verification results:

- `POST /language/translate/v2` `{q:["Hello world"], target:"es"}` → `"Hola mundo"`
- `POST /language/translate/v2/document` with 2-paragraph `.docx`, target `es` → Spanish round-trip correct
- `POST /language/translate/v2/document` with a PIL-generated PNG containing `"Hello world, this is a test image."`, target `fr` → translated `.docx` paragraph `"Bonjour, c'est une image de test."` ✅ **image OCR path validated on real hardware for the first time**
- `GET /language/translate/v2?q=...` → 405 (privacy hold preserved)

First-time install gotchas encountered and resolved:

1. `paddlepaddle-gpu==3.0.0` does not exist on the cu129 index. Fixed by bumping to 3.2.2 (plan updated).
2. `paddleocr==3.0.0` fails `pip install` on Python 3.12 because `paddlex[ocr]==3.0.0` build requires `pkg_resources` inside pip's isolated build env. Fixed by dropping the pin (plan updated).
3. Did not encounter any runtime incompatibility between torch's cu128 ABI and paddle's cu129-bundled nvidia libs — monitor on future pod rebuilds.
