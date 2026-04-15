(() => {
  const srcSel = document.getElementById("source-lang");
  const tgtSel = document.getElementById("target-lang");
  const srcText = document.getElementById("source-text");
  const tgtText = document.getElementById("target-text");
  const swapBtn = document.getElementById("swap-btn");
  const clearBtn = document.getElementById("clear-btn");
  const copyBtn = document.getElementById("copy-btn");
  const statusEl = document.getElementById("status");
  const detectedEl = document.getElementById("detected-label");
  const counter = document.getElementById("char-count");

  const LS_SRC = "ot.src";
  const LS_TGT = "ot.tgt";
  const MAX_CHARS = 5000;
  const DEBOUNCE_MS = 350;

  const languageName = new Intl.DisplayNames(["en"], { type: "language" });
  const nameOf = (code) => {
    if (!code || code === "auto") return "Detect language";
    try {
      return languageName.of(code) || code;
    } catch {
      return code;
    }
  };

  let currentController = null;
  let debounceTimer = null;

  async function loadLanguages() {
    const res = await fetch("/language/translate/v2/languages?target=en", {
      cache: "no-store",
      credentials: "omit",
    });
    if (!res.ok) throw new Error("Failed to load languages");
    const data = await res.json();
    const langs = (data && data.data && data.data.languages) || [];

    const decorated = langs
      .map((l) => {
        const browserName = nameOf(l.language);
        // Prefer the browser's localized name when it actually resolves
        // (i.e. it's not just the raw code echoed back). Otherwise fall back
        // to the server-provided name (which knows NLLB-specific codes).
        const name =
          browserName && browserName.toLowerCase() !== l.language.toLowerCase()
            ? browserName
            : l.name || l.language;
        return { code: l.language, name };
      })
      .sort((a, b) => a.name.localeCompare(b.name));

    for (const l of decorated) {
      const o1 = document.createElement("option");
      o1.value = l.code;
      o1.textContent = l.name;
      srcSel.appendChild(o1);

      const o2 = document.createElement("option");
      o2.value = l.code;
      o2.textContent = l.name;
      tgtSel.appendChild(o2);
    }

    const savedSrc = localStorage.getItem(LS_SRC) || "auto";
    const savedTgt = localStorage.getItem(LS_TGT) || "en";
    srcSel.value = [...srcSel.options].some((o) => o.value === savedSrc) ? savedSrc : "auto";
    tgtSel.value = [...tgtSel.options].some((o) => o.value === savedTgt) ? savedTgt : "en";
  }

  function setStatus(msg, isError = false) {
    statusEl.textContent = msg || "";
    statusEl.classList.toggle("error", !!isError);
  }

  function updateCounter() {
    counter.textContent = `${srcText.value.length} / ${MAX_CHARS}`;
  }

  async function translate() {
    const text = srcText.value;
    detectedEl.textContent = "";

    if (!text.trim()) {
      tgtText.value = "";
      setStatus("");
      if (currentController) currentController.abort();
      return;
    }

    const target = tgtSel.value;
    if (!target) return;

    const body = { q: [text], target };
    if (srcSel.value && srcSel.value !== "auto") {
      body.source = srcSel.value;
    }

    if (currentController) currentController.abort();
    currentController = new AbortController();

    setStatus("Translating…");

    try {
      const res = await fetch("/language/translate/v2", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        cache: "no-store",
        credentials: "omit",
        body: JSON.stringify(body),
        signal: currentController.signal,
      });

      if (!res.ok) {
        let detail = `HTTP ${res.status}`;
        try {
          const err = await res.json();
          if (err && err.detail) detail = err.detail;
        } catch {}
        throw new Error(detail);
      }

      const data = await res.json();
      const t = data && data.data && data.data.translations && data.data.translations[0];
      if (!t) throw new Error("Empty response");

      tgtText.value = t.translatedText || "";
      if (t.detectedSourceLanguage) {
        detectedEl.textContent = `Detected: ${nameOf(t.detectedSourceLanguage)}`;
      }
      setStatus("");
    } catch (e) {
      if (e.name === "AbortError") return;
      setStatus(e.message || "Translation failed", true);
    }
  }

  function scheduleTranslate() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(translate, DEBOUNCE_MS);
  }

  srcText.addEventListener("input", () => {
    updateCounter();
    scheduleTranslate();
  });

  srcSel.addEventListener("change", () => {
    localStorage.setItem(LS_SRC, srcSel.value);
    swapBtn.disabled = srcSel.value === "auto";
    scheduleTranslate();
  });

  tgtSel.addEventListener("change", () => {
    localStorage.setItem(LS_TGT, tgtSel.value);
    scheduleTranslate();
  });

  swapBtn.addEventListener("click", () => {
    if (srcSel.value === "auto") return;
    const s = srcSel.value;
    srcSel.value = tgtSel.value;
    tgtSel.value = s;
    const st = srcText.value;
    srcText.value = tgtText.value;
    tgtText.value = st;
    localStorage.setItem(LS_SRC, srcSel.value);
    localStorage.setItem(LS_TGT, tgtSel.value);
    updateCounter();
    translate();
  });

  clearBtn.addEventListener("click", () => {
    srcText.value = "";
    tgtText.value = "";
    detectedEl.textContent = "";
    setStatus("");
    updateCounter();
    srcText.focus();
  });

  copyBtn.addEventListener("click", async () => {
    if (!tgtText.value) return;
    try {
      await navigator.clipboard.writeText(tgtText.value);
      setStatus("Copied");
      setTimeout(() => setStatus(""), 1200);
    } catch {
      setStatus("Copy failed", true);
    }
  });

  // --- Tabs ---
  const tabText = document.getElementById("tab-text");
  const tabDocs = document.getElementById("tab-docs");
  const textPane = document.getElementById("text-pane");
  const docsPane = document.getElementById("docs-pane");
  const LS_TAB = "ot.tab";

  function setTab(which) {
    const docs = which === "docs";
    tabText.classList.toggle("active", !docs);
    tabDocs.classList.toggle("active", docs);
    tabText.setAttribute("aria-selected", String(!docs));
    tabDocs.setAttribute("aria-selected", String(docs));
    textPane.hidden = docs;
    docsPane.hidden = !docs;
    try {
      localStorage.setItem(LS_TAB, which);
    } catch {}
  }

  tabText.addEventListener("click", () => setTab("text"));
  tabDocs.addEventListener("click", () => setTab("docs"));

  // --- Document translation ---
  const docSrcSel = document.getElementById("doc-source-lang");
  const docTgtSel = document.getElementById("doc-target-lang");
  const dropZone = document.getElementById("drop-zone");
  const dropInput = document.getElementById("drop-input");
  const dropFile = document.getElementById("drop-file");
  const docStatus = document.getElementById("doc-status");
  const docTranslateBtn = document.getElementById("doc-translate-btn");
  const MAX_DOC_BYTES = 25 * 1024 * 1024;

  let stagedFile = null;

  function mirrorLanguagesToDocs() {
    for (const opt of srcSel.options) {
      const o = document.createElement("option");
      o.value = opt.value;
      o.textContent = opt.textContent;
      docSrcSel.appendChild(o);
    }
    for (const opt of tgtSel.options) {
      const o = document.createElement("option");
      o.value = opt.value;
      o.textContent = opt.textContent;
      docTgtSel.appendChild(o);
    }
    docSrcSel.value = srcSel.value;
    docTgtSel.value = tgtSel.value;
  }

  function setDocStatus(msg, isError = false) {
    docStatus.textContent = msg || "";
    docStatus.classList.toggle("error", !!isError);
  }

  function stageFile(file) {
    if (!file) {
      stagedFile = null;
      dropFile.textContent = "";
      docTranslateBtn.disabled = true;
      return;
    }
    if (file.size > MAX_DOC_BYTES) {
      setDocStatus("File too large (max 25 MB).", true);
      return;
    }
    stagedFile = file;
    dropFile.textContent = `${file.name} · ${(file.size / 1024).toFixed(0)} KB`;
    docTranslateBtn.disabled = false;
    setDocStatus("");
  }

  dropZone.addEventListener("click", () => dropInput.click());
  dropZone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      dropInput.click();
    }
  });
  dropInput.addEventListener("change", () => {
    stageFile(dropInput.files && dropInput.files[0]);
  });
  ["dragenter", "dragover"].forEach((evt) => {
    dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.add("drag");
    });
  });
  ["dragleave", "drop"].forEach((evt) => {
    dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove("drag");
    });
  });
  dropZone.addEventListener("drop", (e) => {
    const files = e.dataTransfer && e.dataTransfer.files;
    if (files && files[0]) stageFile(files[0]);
  });

  docTranslateBtn.addEventListener("click", async () => {
    if (!stagedFile) return;
    docTranslateBtn.disabled = true;
    setDocStatus("Uploading & translating…");
    try {
      const fd = new FormData();
      fd.append("file", stagedFile);
      fd.append("target", docTgtSel.value);
      if (docSrcSel.value && docSrcSel.value !== "auto") {
        fd.append("source", docSrcSel.value);
      }
      const res = await fetch("/language/translate/v2/document", {
        method: "POST",
        cache: "no-store",
        credentials: "omit",
        body: fd,
      });
      if (!res.ok) {
        let detail = `HTTP ${res.status}`;
        try {
          const err = await res.json();
          if (err && err.detail) detail = err.detail;
        } catch {}
        throw new Error(detail);
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "translated.docx";
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
      setDocStatus("Done.");
    } catch (e) {
      setDocStatus(e.message || "Translation failed", true);
    } finally {
      docTranslateBtn.disabled = !stagedFile;
    }
  });

  (async () => {
    try {
      await loadLanguages();
      mirrorLanguagesToDocs();
      swapBtn.disabled = srcSel.value === "auto";
      updateCounter();
      const savedTab = localStorage.getItem(LS_TAB) || "text";
      setTab(savedTab === "docs" ? "docs" : "text");
      srcText.focus();
    } catch (e) {
      setStatus(e.message || "Failed to load languages", true);
    }
  })();
})();
