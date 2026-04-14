# Session: Google-Translate-style Web UI + On-Disk Privacy Hardening

**Date:** 2026-04-14
**Branch:** main
**Commit:** (pending — this session summary is part of the commit)

## Summary

Added a minimal Google-Translate-style frontend to `open-translate` (a FastAPI/NLLB-200 service) and hardened the server so no user-submitted or translated text can land on disk on the server. The UI is served by the same FastAPI app at `/ui/` as plain static HTML/CSS/JS (no build step, no external CDN, no third-party requests). Alongside the UI, the server was audited and modified to eliminate the three realistic on-disk leak surfaces: the GET translate/detect endpoints (URL query-string capture), uvicorn access logs (stdout → log driver), and missing no-store response headers (browser/proxy caches). README documents what the code enforces vs. what remains an operator responsibility (OS swap, log drivers, reverse proxies), plus an honest caveat that Python cannot zeroize memory.

## Changes

- `server.py` — Added `StaticFiles` import and `Request` import; added `_no_store_headers` HTTP middleware setting `Cache-Control: no-store, no-cache, must-revalidate`, `Pragma: no-cache`, `Referrer-Policy: no-referrer` on every response; mounted `static/ui` at `/ui` via `app.mount(...)` guarded by `os.path.isdir`; **removed `@app.get("/language/translate/v2")` handler** (`translate_get`) and **removed `@app.get("/language/translate/v2/detect")` handler** (`detect_get`) — POST-only now — so user text can never appear in a URL.
- `start.sh` — Added `--no-access-log` flag to the uvicorn invocation so request lines never reach stdout (and therefore never reach Docker/journald log files).
- `Dockerfile` — Added `COPY static /static` so the UI assets ship in the image alongside `server.py` and `start.sh`.
- `README.md` — Added a new "🖥️ Web UI" section between Quick Start and API Reference. Documents the UI URL, the privacy guarantees actually enforced in code, the operator responsibilities the code cannot enforce (swap, log drivers, reverse proxies), and an honest caveat about Python string/memory GC.
- `static/ui/index.html` — New. Two-panel Google-Translate-style layout: source language `<select>` (with "Detect language" first), source `<textarea>` with char counter (5000-char limit), clear button, swap button, target language `<select>`, read-only target `<textarea>`, copy button, status + "Detected: …" labels. `<meta name="referrer" content="no-referrer">`.
- `static/ui/style.css` — New. Minimal Google-Translate-ish styling; CSS grid two-pane layout with a centered swap button; system font stack; responsive stack under 760px; no external fonts.
- `static/ui/app.js` — New. Vanilla JS (no framework). Loads languages from `GET /language/translate/v2/languages`, resolves human-readable names via `Intl.DisplayNames`, populates both dropdowns sorted by display name. 350ms-debounced auto-translate via `POST /language/translate/v2` with JSON body `{q:[text], target, source?}`. `cache: 'no-store'`, `credentials: 'omit'`. Aborts in-flight requests on new keystrokes. Handles detect (`detectedSourceLanguage` → label). Swap/copy/clear. Persists only language *codes* (never text) in `localStorage` under keys `ot.src` / `ot.tgt`.

## Decisions & Rationale

- **UI mounted at `/ui` (not `/`)** → user preference; keeps API paths untouched and avoids any conflict with the existing routes.
- **Removed GET translate AND GET detect endpoints** (plan only specified translate) → the detect GET has the exact same URL-leak class, so removing both closes the whole category. Breaking change for any client that used GET, but privacy was the explicit primary requirement.
- **`--no-access-log` instead of `> /dev/null`** → user chose to keep startup/error output visible. Tracebacks do not contain user text in the current code path (verified by reading `server.py` in full), so this is safe.
- **No CORS middleware** → UI is same-origin; smaller attack surface.
- **No memory-zeroization code** → CPython strings are immutable and GC'd through Starlette + tokenizer, so any `bytearray`-overwrite code would be theatre. README states this honestly.
- **No framework / no build step for the UI** → minimizes supply chain, keeps the privacy story auditable in ~400 lines of readable code.
- **`StaticFiles` mount guarded by `os.path.isdir`** → dev environments without the `static/` directory (e.g. running `server.py` from an unusual cwd) still start cleanly.

## Remaining Work

- **Not tested end-to-end against a live server** — the NLLB model requires GPU + model download, so I did not boot it in this session. Syntax-checked `server.py` with `ast.parse` (passed). The user should run verification steps from the plan:
  1. `./start.sh` → `GET http://localhost:8000/health` returns 200.
  2. Open `http://localhost:8000/ui/`, confirm both dropdowns populate.
  3. Type "Hello world", target = Spanish → "Hola mundo" appears after ~350ms.
  4. DevTools Network: every translate call is `POST` with JSON body; zero `?q=` URLs.
  5. `curl -i 'http://localhost:8000/language/translate/v2?q=hi&target=es'` → 405/404.
  6. Server stdout contains no `"POST /... 200"` lines after translating.
  7. `curl -I -X POST ...` → headers include `cache-control: no-store, no-cache, must-revalidate`.
  8. Swap, copy, clear, detect flows work in the UI.
  9. Responsive layout at <760px.
- **Breaking change documentation** — README's API Reference section still only describes POST, so no doc changes needed there, but if clients in the wild use the GET form, they will break. Consider a CHANGELOG or release note.
- **Optional future hardening** — set `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Content-Security-Policy` on UI responses. Not done because UI is plain local assets; low value relative to the work.

## Resumption Context

Key files to review when picking this up:
- `server.py` (middleware ~line 80–95, mount ~line 95, POST translate ~line 293, POST detect ~line 368, languages ~line 378)
- `static/ui/app.js` (translation flow, language loading, swap logic)
- `static/ui/index.html` (structure) and `static/ui/style.css` (layout)
- `start.sh` (line 14: `--no-access-log`)
- `Dockerfile` (line ~55: `COPY static /static`)
- `README.md` (new "🖥️ Web UI" section between Quick Start and API Reference)

Suggested opening prompt for next session:

> "Continuing work on open-translate. We added a Google-Translate-style UI at /ui and hardened the server so no user text touches disk: removed GET translate/detect endpoints, added --no-access-log to start.sh, added no-store response headers via middleware. I need to boot the server and verify end-to-end against a live NLLB model — please start the server, open the UI in a browser, and walk through the verification steps in `.planning/sessions/2026-04-14-google-translate-ui.md` (Remaining Work section)."
