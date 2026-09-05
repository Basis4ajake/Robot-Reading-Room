import { listSources, addSource, removeSource, ingestLibrary } from "./api.js";

let sectionEl;
let listEl;
let addBtn;
let ingestBtn;
let resultEl;
let errorEl;

let currentLibraryId = null;

function showError(message) {
  errorEl.textContent = message;
  errorEl.hidden = false;
}

function clearError() {
  errorEl.hidden = true;
  errorEl.textContent = "";
}

function clearResult() {
  resultEl.hidden = true;
  resultEl.textContent = "";
}

function renderSourceList(sources) {
  listEl.innerHTML = "";

  if (sources.length === 0) {
    const empty = document.createElement("li");
    empty.className = "library-empty";
    empty.textContent = "No sources yet. Add a file to get started.";
    listEl.appendChild(empty);
    return;
  }

  for (const source of sources) {
    const item = document.createElement("li");
    item.className = "library-item library-item-btn";

    const info = document.createElement("span");
    info.className = "library-name";
    info.textContent = `${source.filename} (${source.file_type})`;

    const removeBtn = document.createElement("button");
    removeBtn.type = "button";
    removeBtn.className = "danger";
    removeBtn.textContent = "Remove";
    removeBtn.addEventListener("click", () => handleRemove(source.source_id, source.filename));

    item.appendChild(info);
    item.appendChild(removeBtn);
    listEl.appendChild(item);
  }
}

async function refreshSources() {
  if (!currentLibraryId) return;
  clearError();
  try {
    const sources = await listSources(currentLibraryId);
    renderSourceList(sources);
  } catch (err) {
    showError(`Could not load sources: ${err.message}`);
  }
}

async function handleAddSource() {
  clearError();
  clearResult();

  const selected = await window.__TAURI__.dialog.open({
    multiple: false,
    filters: [{ name: "Documents", extensions: ["pdf", "txt", "md", "markdown"] }],
  });

  if (!selected) return;

  try {
    await addSource(currentLibraryId, selected);
    await refreshSources();
  } catch (err) {
    showError(`Could not add source: ${err.message}`);
  }
}

async function handleRemove(sourceId, filename) {
  const confirmed = window.confirm(`Remove "${filename}" from this library? This deletes its chunks from the index and cannot be undone.`);
  if (!confirmed) return;

  clearError();
  try {
    await removeSource(currentLibraryId, sourceId);
    await refreshSources();
  } catch (err) {
    showError(`Could not remove source: ${err.message}`);
  }
}

async function handleIngest() {
  clearError();
  clearResult();
  ingestBtn.disabled = true;
  ingestBtn.textContent = "Ingesting...";

  try {
    const result = await ingestLibrary(currentLibraryId);
    resultEl.textContent = `Processed: ${result.processed.length}, skipped: ${result.skipped.length}, removed: ${result.removed.length}`;
    resultEl.hidden = false;
    await refreshSources();
  } catch (err) {
    showError(`Ingestion failed: ${err.message}`);
  } finally {
    ingestBtn.disabled = false;
    ingestBtn.textContent = "Ingest";
  }
}

export function showSourcesFor(libraryId) {
  currentLibraryId = libraryId;
  clearError();
  clearResult();
  sectionEl.hidden = false;
  refreshSources();
}

export function hideSources() {
  currentLibraryId = null;
  sectionEl.hidden = true;
  listEl.innerHTML = "";
}

export function initSources() {
  sectionEl = document.querySelector("#sources-section");
  listEl = document.querySelector("#source-list");
  addBtn = document.querySelector("#add-source-btn");
  ingestBtn = document.querySelector("#ingest-btn");
  resultEl = document.querySelector("#ingest-result");
  errorEl = document.querySelector("#sources-error-msg");

  addBtn.addEventListener("click", handleAddSource);
  ingestBtn.addEventListener("click", handleIngest);

  hideSources();
}
