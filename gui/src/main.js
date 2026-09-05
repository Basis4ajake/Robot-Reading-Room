// Base URL of the local FastAPI server started with:
//   python -m local_knowledge_library.api
// Matches the LKL_API_HOST/LKL_API_PORT defaults in .env.example.
const API_BASE = "http://127.0.0.1:8000/api/v1";

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

async function loadHealth() {
  const response = await fetch(`${API_BASE}/health`);
  if (!response.ok) {
    throw new Error(`GET /health failed with status ${response.status}`);
  }
  const health = await response.json();

  serverStatusEl.textContent = health.status;
  dataDirEl.textContent = health.data_dir;
  ollamaStatusEl.textContent = health.ollama_available ? "reachable" : "not reachable";
}

async function loadModels() {
  const response = await fetch(`${API_BASE}/models`);
  if (!response.ok) {
    throw new Error(`GET /models failed with status ${response.status}`);
  }
  const models = await response.json();

  if (!models.available || models.models.length === 0) {
    modelsListEl.textContent = "none found";
    return;
  }

  modelsListEl.textContent = models.models.map((model) => model.name).join(", ");
}

async function refreshStatus() {
  clearError();
  serverStatusEl.textContent = "checking...";
  dataDirEl.textContent = "-";
  ollamaStatusEl.textContent = "-";
  modelsListEl.textContent = "-";

  try {
    await loadHealth();
    await loadModels();
  } catch (err) {
    serverStatusEl.textContent = "unreachable";
    showError(
      `Could not reach the backend at ${API_BASE}. Is "python -m local_knowledge_library.api" running? (${err.message})`,
    );
  }
}

window.addEventListener("DOMContentLoaded", () => {
  serverStatusEl = document.querySelector("#server-status");
  dataDirEl = document.querySelector("#data-dir");
  ollamaStatusEl = document.querySelector("#ollama-status");
  modelsListEl = document.querySelector("#models-list");
  errorMsgEl = document.querySelector("#error-msg");
  refreshBtn = document.querySelector("#refresh-btn");

  refreshBtn.addEventListener("click", refreshStatus);

  refreshStatus();
});
