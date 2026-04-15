# Session: PaddleOCR document translation + Vast deploy plan

**Date:** 2026-04-15
**Branch:** main
**Commit:** (pending — see commit hash after this session)

## Summary

Executed the yesterday-planned OCR document translation feature end-to-end
(plan lived in `~/.claude/plans/floating-zooming-hoare.md`) and then wrote a
deployment plan for a new Vast.ai RTX 5090 pod. The feature adds a new
`POST /language/translate/v2/document` endpoint that accepts DOCX, PDF
(text-layer or scanned), and image uploads and returns a translated `.docx`,
all in-memory and privacy-preserving. PaddleOCR 3.0 is loaded on GPU at
startup alongside NLLB (separate `_OCR_LOCK` so both pipelines can overlap).
The existing UI grew a Text/Documents tab bar with drag-and-drop. No deploy
was executed this session — the new Vast pod plan is staged at
`.planning/vast-deploy-plan.md` for the next session to run.

## Changes

- `server.py` — new document-translation stack: imports (numpy, PIL, io,
  fastapi File/Form/UploadFile, StreamingResponse); constants `MAX_DOC_BYTES`,
  `MAX_PDF_PAGES`, `OCR_DPI`, `OCR_LANG`; globals `_ocr_engine`, `_OCR_LOCK`;
  startup init of `PaddleOCR(use_doc_orientation_classify=False,
  use_doc_unwarping=False, use_textline_orientation=True, lang=OCR_LANG,
  device='gpu' or 'cpu')` with graceful fallback; helpers
  `_classify_upload`, `_extract_docx_runs`, `_extract_pdf_text_layer`,
  `_rasterize_pdf`, `_ocr_image`, `_ocr_pdf_pages`, `_translate_paragraphs`,
  `_build_translated_docx`, `_resolve_src_for_text`; new endpoint
  `POST /language/translate/v2/document` that dispatches to docx/pdf/image
  and returns `StreamingResponse` of a `.docx` blob.
- `static/ui/index.html` — header gained a `<nav class="tabs">` with
  Text/Documents tabs; main translate panes wrapped in `<main id="text-pane">`;
  new `<section id="docs-pane" hidden>` with language dropdowns, drop zone,
  file input, Translate button, status line.
- `static/ui/app.js` — tab switch logic (`LS_TAB` preference, `setTab`);
  `mirrorLanguagesToDocs()` clones the loaded language list into the doc
  dropdowns; drag/drop/click handlers; `stageFile()` with 25 MB client cap;
  `doc-translate-btn` click handler that POSTs `FormData` to
  `/language/translate/v2/document`, receives a Blob, downloads it via a
  temporary `<a download="translated.docx">` + `URL.revokeObjectURL`.
- `static/ui/style.css` — `.tabs`, `.tab`, `.docs`, `.docs-card`,
  `.docs-langs`, `.drop-zone` (dashed border, hover/focus/drag states),
  `.drop-title`, `.drop-hint`, `.drop-file`, `.docs-actions`, `.btn-primary`;
  responsive tweaks in the `@media (max-width: 760px)` block.
- `Dockerfile` — builder stage installs `paddlepaddle-gpu==3.0.0` from the
  `cu126` wheel index, then `paddleocr==3.0.0`, `pypdfium2==4.30.0`,
  `python-docx==1.1.2`, `pypdf==5.1.0`, `python-multipart==0.0.20`,
  `Pillow==11.0.0`; pre-warms PaddleOCR weights into `/opt/paddlex_cache`.
  Runtime stage adds `MAX_DOC_BYTES`, `MAX_PDF_PAGES`, `OCR_DPI`, `OCR_LANG`,
  `PADDLE_PDX_CACHE_HOME=/workspace/.cache/paddlex` envs and copies
  `/opt/paddlex_cache` from the builder.
- `scripts/runpod-setup.sh` — after the main `pip install -r`, installs
  paddle cu126 + ocr stack, pre-warms weights (device="cpu" for warm,
  real requests use gpu), verifies CUDA.
- `scripts/runpod-launch.sh` — exports `PADDLE_PDX_CACHE_HOME` and
  `OCR_LANG` defaults so restarts don't need to re-specify them.
- `pyproject.toml` — added `python-multipart`, `python-docx`, `pypdf`,
  `pypdfium2`, `Pillow` to `[project.dependencies]`. Paddle intentionally
  omitted (lives in a non-pypi index).
- `requirements.txt` — appended the same five deps with a comment noting
  paddle is installed separately (Dockerfile / setup script handle it).
- `CLAUDE.md` — architecture bullet for PaddleOCR; new API surface bullet
  for `POST /language/translate/v2/document` with field list and env knobs;
  privacy section extended to cover the in-memory document flow (no
  `tempfile`, `pypdfium2` bytes-only, hardcoded `translated.docx` filename);
  four new gotchas (cu126 wheel on cu128 host, PaddleOCR 3.0 API shift,
  cache path, VRAM budget).
- `.planning/vast-deploy-plan.md` — **new** end-to-end plan to deploy the
  updated server on a Vast.ai RTX 5090 / CUDA 13.1 pod at
  `212.13.234.30:1431`. Keyed observations: 120 GB rootfs (plenty), no torch
  pre-installed (install torch 2.9.1 cu128 wheels), Jupyter owns :8080 so
  run on :8005 and tunnel `-L 8080:localhost:8005`, use paddle cu129 wheel
  (CUDA 13.1 host matches). Heredoc is idempotent and reuses
  `scripts/runpod-launch.sh` unchanged.
- `.planning/sessions/2026-04-15-ocr-document-translation.md` — this file.

## Decisions & Rationale

- **PaddleOCR on the same GPU, not CPU fallback.** User explicitly said
  "paddle should also run on the gpu there is enough space, let's do it all".
  Budget: NLLB-1.3B ≈ 3.3 GB + PaddleOCR ≈ 1–2 GB on a 32 GB 5090. The
  startup init does `device="gpu" if torch.cuda.is_available() else "cpu"`
  with a wider try/except around the whole block so an init failure leaves
  `_ocr_engine = None` and the server still serves text / docx / text-layer
  PDFs.
- **Separate `_OCR_LOCK`**, not sharing `_MODEL_LOCK`. OCR on one page can
  overlap NLLB translating the previous page — useful on multi-page scans.
- **PaddleOCR 3.0 API** (`use_textline_orientation=True`, `device='gpu'`,
  `.predict()`, `result[0].get("rec_texts")`) instead of 2.x
  (`use_angle_cls=True`, `use_gpu=True`, `.ocr()`). This matches the pinned
  `paddleocr==3.0.0` wheel. The plan called for 2.x-style but I upgraded to
  the current API because 2.x flags are removed in 3.0.
- **DOCX output for every input type.** Simpler than reconstructing PDFs
  and gives a consistent client UX. Image-overlay rendering is explicitly
  a stretch goal.
- **Reuse `runpod-launch.sh` for Vast** instead of forking `vast-launch.sh`.
  Script is already env-driven (`PORT`, `HF_HOME`, `PADDLE_PDX_CACHE_HOME`).
  Adding a second launcher would be pointless churn.
- **Port 8005 on Vast, not 8080.** Probed the new pod at `212.13.234.30:1431`
  and found Jupyter already listening on `:8080` (`PORTAL_CONFIG` confirms
  Vast owns it). Running open-translate on 8005 and tunneling with
  `-L 8080:localhost:8005` keeps the browser URL stable at
  `http://localhost:8080/ui/` without stepping on Jupyter.
- **Paddle cu129 wheel on Vast, cu126 wheel in Docker image.** Vast pod is
  on CUDA 13.1 so paddle's native cu129 wheel is the natural pick; the
  Docker image is pinned to the cu126 forward-compat wheel because it's the
  tested path for the existing RunPod 12.8.1 base and avoids another wheel
  variant in CI.

## Remaining Work

1. **Execute the Vast deploy** — rsync + run the heredoc in
   `.planning/vast-deploy-plan.md §Step 2`. Nothing is live yet.
2. **End-to-end verification on the new pod** — the curl / UI recipe in
   `.planning/vast-deploy-plan.md §Step 3`, especially the image-OCR path
   (prior sessions only tested text + DOCX).
3. **Unknown: PaddleOCR 3.0 result shape under different inputs** — my
   `_ocr_image()` tries both `res.get("rec_texts")` and `res["rec_texts"]`
   defensively; if 3.0.0's `.predict()` actually returns a different
   object structure, expect a silent empty list. Verify on real image
   before declaring done.
4. **Consider publishing the Docker image** so future Vast pods can skip
   the rsync+install dance and just pull — the CI workflow is already in
   place (`.github/workflows/docker-publish.yml`).

## Resumption Context

Key files to review when picking this up:
- `.planning/vast-deploy-plan.md` — the full deploy recipe for the new pod
- `server.py` — especially the startup OCR init and the
  `translate_document` endpoint (search for `"document"`)
- `scripts/runpod-launch.sh` — the launcher that Vast will reuse
- `CLAUDE.md` — updated gotchas section (paddle wheel, VRAM budget, API
  shift) and the new API surface entry

Live boxes:
- RunPod (older, working): `ssh -p 46718 -i ~/.ssh/id_ed25519 root@149.36.1.202`
  — still the last known-good deployment, but does NOT yet have the
  document-translation endpoint (not deployed this session).
- Vast (new, target for next session):
  `ssh -p 1431 root@212.13.234.30 -L 8080:localhost:8005` — empty pod,
  rsync then run the heredoc from the plan.
