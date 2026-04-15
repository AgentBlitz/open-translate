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
pip install --no-cache-dir \
  paddlepaddle-gpu==3.0.0 \
  -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
pip install --no-cache-dir \
  paddleocr==3.0.0 \
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
| Vast pod is ephemeral, loses state on restart | Repeat Step 1 + Step 2; the heredoc is idempotent. Longer-term: publish the Docker image and use it as the Vast container image instead of the PyTorch template |
| Port collision — 8080 is owned by Jupyter (`PORTAL_CONFIG` maps localhost:8080 → Jupyter) | Confirmed at probe time. We run on 8005 and the Mac-side tunnel maps `-L 8080:localhost:8005`. Do **not** try to reclaim 8080 — it'll break the Vast portal. |

## Critical files to read before executing

- [scripts/runpod-launch.sh](../scripts/runpod-launch.sh) — the start/stop/restart wrapper (reused unchanged)
- [scripts/runpod-setup.sh](../scripts/runpod-setup.sh) — the RunPod source-install script (template for this plan's heredoc, minus the `--system-site-packages` bit and with cu129 paddle)
- [server.py](../server.py) — confirms `OCR_LANG`, `PADDLE_PDX_CACHE_HOME`, `MAX_DOC_BYTES`, `PORT` are all env-driven
- [CLAUDE.md](../CLAUDE.md) — the privacy invariants and VRAM budget notes still apply
