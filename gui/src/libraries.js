import { listLibraries, createLibrary, getLibrary, updateLibrary, deleteLibrary } from "./api.js";
import { showSourcesFor, hideSources } from "./sources.js";
import { showChatFor, hideChat } from "./chat.js";

let listEl;
let errorEl;
let newBtn;
let createForm;
let createCancelBtn;
let detailSection;
let detailForm;
let detailTitle;
let detailCloseBtn;
let detailDeleteBtn;

function showError(message) {
  errorEl.textContent = message;
  errorEl.hidden = false;
}

function clearError() {
  errorEl.hidden = true;
  errorEl.textContent = "";
}

function renderLibraryList(libraries) {
  listEl.innerHTML = "";

  if (libraries.length === 0) {
    const empty = document.createElement("li");
    empty.className = "library-empty";
    empty.textContent = "No libraries yet. Create one to get started.";
    listEl.appendChild(empty);
    return;
  }

  for (const library of libraries) {
    const item = document.createElement("li");
    item.className = "library-item";

    const button = document.createElement("button");
    button.type = "button";
    button.className = "library-item-btn";
    button.innerHTML = `
      <span class="library-name">${library.name}</span>
      <span class="library-meta">${library.source_count} sources · ${library.chunk_count} chunks</span>
    `;
    button.addEventListener("click", () => openDetail(library.library_id));

    item.appendChild(button);
    listEl.appendChild(item);
  }
}

async function refreshList() {
  clearError();
  try {
    const libraries = await listLibraries();
    renderLibraryList(libraries);
  } catch (err) {
    showError(`Could not load libraries: ${err.message}`);
  }
}

function showCreateForm() {
  createForm.hidden = false;
  detailSection.hidden = true;
}

function hideCreateForm() {
  createForm.hidden = true;
  createForm.reset();
}

async function handleCreateSubmit(event) {
  event.preventDefault();
  clearError();

  const formData = new FormData(createForm);
  const payload = {
    library_id: formData.get("library_id").trim(),
    name: formData.get("name").trim(),
    description: formData.get("description").trim(),
    chunk_size: Number(formData.get("chunk_size")),
    chunk_overlap: Number(formData.get("chunk_overlap")),
    top_k: Number(formData.get("top_k")),
    llm_model: formData.get("llm_model").trim(),
    embedding_model: formData.get("embedding_model").trim(),
  };

  try {
    await createLibrary(payload);
    hideCreateForm();
    await refreshList();
  } catch (err) {
    showError(`Could not create library: ${err.message}`);
  }
}

async function openDetail(libraryId) {
  clearError();
  createForm.hidden = true;

  try {
    const library = await getLibrary(libraryId);
    detailForm.dataset.libraryId = library.library_id;
    detailTitle.textContent = library.name;
    detailForm.elements.name.value = library.name;
    detailForm.elements.description.value = library.description;
    detailForm.elements.chunk_size.value = library.chunk_size;
    detailForm.elements.chunk_overlap.value = library.chunk_overlap;
    detailForm.elements.top_k.value = library.top_k;
    detailForm.elements.llm_model.value = library.llm_model;
    detailForm.elements.embedding_model.value = library.embedding_model || "";
    detailSection.hidden = false;
    showSourcesFor(library.library_id);
    showChatFor(library.library_id);
  } catch (err) {
    showError(`Could not open library "${libraryId}": ${err.message}`);
  }
}

function closeDetail() {
  detailSection.hidden = true;
  detailForm.reset();
  delete detailForm.dataset.libraryId;
  hideSources();
  hideChat();
}

async function handleUpdateSubmit(event) {
  event.preventDefault();
  clearError();

  const libraryId = detailForm.dataset.libraryId;
  const formData = new FormData(detailForm);
  const payload = {
    name: formData.get("name").trim(),
    description: formData.get("description").trim(),
    chunk_size: Number(formData.get("chunk_size")),
    chunk_overlap: Number(formData.get("chunk_overlap")),
    top_k: Number(formData.get("top_k")),
    llm_model: formData.get("llm_model").trim(),
    embedding_model: formData.get("embedding_model").trim(),
  };

  try {
    // config.json on disk is authoritative, so re-open the library with
    // the server's own response rather than trusting our local form state.
    await updateLibrary(libraryId, payload);
    await openDetail(libraryId);
    await refreshList();
  } catch (err) {
    showError(`Could not update library "${libraryId}": ${err.message}`);
  }
}

async function handleDelete() {
  const libraryId = detailForm.dataset.libraryId;
  if (!libraryId) return;

  const confirmed = window.confirm(
    `Delete library "${libraryId}"? This removes its sources and index from disk and cannot be undone.`,
  );
  if (!confirmed) return;

  try {
    await deleteLibrary(libraryId);
    closeDetail();
    await refreshList();
  } catch (err) {
    showError(`Could not delete library "${libraryId}": ${err.message}`);
  }
}

export function initLibraries() {
  listEl = document.querySelector("#library-list");
  errorEl = document.querySelector("#library-error-msg");
  newBtn = document.querySelector("#new-library-btn");
  createForm = document.querySelector("#create-library-form");
  createCancelBtn = document.querySelector("#create-library-cancel");
  detailSection = document.querySelector("#library-detail");
  detailForm = document.querySelector("#library-detail-form");
  detailTitle = document.querySelector("#library-detail-title");
  detailCloseBtn = document.querySelector("#library-detail-close");
  detailDeleteBtn = document.querySelector("#library-detail-delete");

  newBtn.addEventListener("click", showCreateForm);
  createCancelBtn.addEventListener("click", hideCreateForm);
  createForm.addEventListener("submit", handleCreateSubmit);

  detailCloseBtn.addEventListener("click", closeDetail);
  detailForm.addEventListener("submit", handleUpdateSubmit);
  detailDeleteBtn.addEventListener("click", handleDelete);

  refreshList();
}
