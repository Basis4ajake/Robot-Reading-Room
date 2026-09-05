import { getHealth, getModels } from "./api.js";
import { initLibraries } from "./libraries.js";
import { initSources } from "./sources.js";
import { initChat } from "./chat.js";
import { initTooltips } from "./tooltip.js";
import { initModelSelects } from "./model-select.js";
import { initTheme } from "./theme.js";
import { initExternalLinks } from "./external-link.js";

let serverStatusEl;
let dataDirEl;
let ollamaStatusEl;
let modelsListEl;
let errorMsgEl;
let refreshBtn;

function showError(message) {
  errorMsgEl.textContent = message;
  errorMsgEl.hidden = false;
}

function clearError() {
  errorMsgEl.hidden = true;
  errorMsgEl.textContent = "";
}

async function refreshStatus() {
  clearError();
  serverStatusEl.textContent = "checking...";
  dataDirEl.textContent = "-";
  ollamaStatusEl.textContent = "-";
  modelsListEl.textContent = "-";

  try {
    const health = await getHealth();
    serverStatusEl.textContent = health.status;
    dataDirEl.textContent = health.data_dir;
    ollamaStatusEl.textContent = health.ollama_available ? "reachable" : "not reachable";

    const models = await getModels();
    modelsListEl.textContent =
      models.available && models.models.length > 0
        ? models.models.map((model) => model.name).join(", ")
        : "none found";
  } catch (err) {
    serverStatusEl.textContent = "unreachable";
    showError(
      `Could not reach the backend. Is "python -m local_knowledge_library.api" running? (${err.message})`,
    );
  }
}

window.addEventListener("DOMContentLoaded", () => {
  initTheme();

  serverStatusEl = document.querySelector("#server-status");
  dataDirEl = document.querySelector("#data-dir");
  ollamaStatusEl = document.querySelector("#ollama-status");
  modelsListEl = document.querySelector("#models-list");
  errorMsgEl = document.querySelector("#error-msg");
  refreshBtn = document.querySelector("#refresh-btn");

  refreshBtn.addEventListener("click", refreshStatus);

  refreshStatus();
  initSources();
  initChat();
  initLibraries();
  initTooltips();
  initModelSelects();
  initExternalLinks();
});
