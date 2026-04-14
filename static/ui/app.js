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
    const res = await fetch("/language/translate/v2/languages", {
      cache: "no-store",
      credentials: "omit",
    });
    if (!res.ok) throw new Error("Failed to load languages");
    const data = await res.json();
    const langs = (data && data.data && data.data.languages) || [];

    const decorated = langs
      .map((l) => ({ code: l.language, name: nameOf(l.language) }))
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

  (async () => {
    try {
      await loadLanguages();
      swapBtn.disabled = srcSel.value === "auto";
      updateCounter();
      srcText.focus();
    } catch (e) {
      setStatus(e.message || "Failed to load languages", true);
    }
  })();
})();
