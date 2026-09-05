// Custom lightweight suggestion dropdown for the LLM/embedding model fields.
// Native <datalist> doesn't render its suggestion list in WebKitGTK (the
// Linux Tauri webview), so we build a small filtered dropdown ourselves.
// The input stays a plain free-text field - picking a suggestion just fills it in.
export const LLM_MODEL_OPTIONS = [
  { value: "qwen2:0.5b", label: "Very lightweight (~350MB) - fastest, lowest quality" },
  { value: "qwen2.5:0.5b", label: "Very lightweight (~400MB) - fastest, newer generation" },
  { value: "qwen2:1.5b", label: "Lightweight (~1GB) - runs well on CPU" },
  { value: "qwen2.5:1.5b", label: "Lightweight (~1GB) - newer generation, runs well on CPU" },
  { value: "qwen2.5:3b", label: "Lightweight-moderate (~1.9GB) - CPU-friendly, better quality" },
  { value: "llama3.2:1b", label: "Very lightweight - fastest, lowest quality" },
  { value: "llama3.2:3b", label: "Lightweight - good balance for CPU" },
  { value: "phi3:mini", label: "Lightweight - needs ~4GB+ RAM" },
  { value: "mistral:7b", label: "Resource intensive - GPU strongly recommended" },
  { value: "llama3.1:8b", label: "Resource intensive - GPU strongly recommended" },
];

// Ordered lightest-to-heaviest. Sizes and standing are sourced from Ollama's
// library and published benchmarks (general MTEB, and ChemTEB for chemistry
// literature retrieval specifically) - not guesses - since this library's
// content is chemistry-heavy and generic embedding quality varies a lot by
// domain.
export const EMBEDDING_MODEL_OPTIONS = [
  { value: "all-minilm", label: "Very lightweight (~46MB) - short 256-token context, weak on technical text" },
  { value: "nomic-embed-text", label: "Lightweight (~274MB) - long context, solid general-purpose default" },
  { value: "qwen3-embedding:0.6b", label: "Lightweight (~640MB) - top general benchmarks, strong on scientific text" },
  { value: "bge-m3", label: "Moderate (~1.2GB) - benchmarks well on chemistry-literature retrieval" },
  { value: "qwen3-embedding:4b", label: "Resource intensive (16GB+ RAM) - best quality for technical/scientific text" },
];

function setupOne(input, options) {
  const wrapper = document.createElement("div");
  wrapper.className = "model-select-wrapper";
  input.parentNode.insertBefore(wrapper, input);
  wrapper.appendChild(input);

  const panel = document.createElement("ul");
  panel.className = "model-suggestions";
  panel.hidden = true;
  wrapper.appendChild(panel);

  function render(filterText) {
    const query = filterText.trim().toLowerCase();
    const matches = options.filter(
      (option) => !query || option.value.toLowerCase().includes(query) || option.label.toLowerCase().includes(query),
    );

    panel.innerHTML = "";
    if (matches.length === 0) {
      panel.hidden = true;
      return;
    }

    for (const option of matches) {
      const item = document.createElement("li");
      item.className = "model-suggestion-item";

      const name = document.createElement("span");
      name.className = "model-suggestion-name";
      name.textContent = option.value;

      const desc = document.createElement("span");
      desc.className = "model-suggestion-desc";
      desc.textContent = option.label;

      item.appendChild(name);
      item.appendChild(desc);

      // mousedown (not click) fires before the input's blur, so we can
      // still read/set its value before the panel gets hidden on blur.
      item.addEventListener("mousedown", (event) => {
        event.preventDefault();
        input.value = option.value;
        panel.hidden = true;
        input.dispatchEvent(new Event("input", { bubbles: true }));
      });

      panel.appendChild(item);
    }
    panel.hidden = false;
  }

  // On focus, show the full list even if the field already has a value
  // (e.g. the default model) - only narrow it once the user actually types.
  input.addEventListener("focus", () => render(""));
  input.addEventListener("input", () => render(input.value));
  input.addEventListener("blur", () => {
    panel.hidden = true;
  });
}

export function initModelSelects(root = document) {
  const inputs = root.querySelectorAll("input[data-model-select]:not([data-model-select-ready])");
  for (const input of inputs) {
    input.dataset.modelSelectReady = "true";
    const options = input.dataset.modelSelect === "embedding" ? EMBEDDING_MODEL_OPTIONS : LLM_MODEL_OPTIONS;
    setupOne(input, options);
  }
}
