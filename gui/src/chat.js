import { getHealth, getModels, sendChatMessage } from "./api.js";

let sectionEl;
let demoBadgeEl;
let logEl;
let formEl;
let inputEl;
let submitBtn;
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

async function refreshDemoBadge() {
  try {
    const [health, models] = await Promise.all([getHealth(), getModels()]);
    const demoMode = !health.ollama_available || !models.available;
    demoBadgeEl.hidden = !demoMode;
  } catch {
    demoBadgeEl.hidden = true;
  }
}

function appendMessage(query, result) {
  const entry = document.createElement("div");
  entry.className = "chat-entry";

  const questionEl = document.createElement("p");
  questionEl.className = "chat-question";
  questionEl.textContent = query;
  entry.appendChild(questionEl);

  const answerEl = document.createElement("p");
  answerEl.className = "chat-answer";
  answerEl.textContent = result.answer;
  entry.appendChild(answerEl);

  // Per-message ground truth from the server, not the library-level /health
  // + /models guess the top-of-window "Demo mode" badge relies on - "llm"
  // needs no badge (the normal case), "dummy"/"computed" get one so a fake
  // or non-retrieved answer is never mistaken for a real one.
  if (result.answer_source && result.answer_source !== "llm") {
    const sourceBadge = document.createElement("span");
    sourceBadge.className = "badge";
    sourceBadge.textContent = result.answer_source === "dummy" ? "Demo answer" : "Computed answer";
    entry.appendChild(sourceBadge);
  }

  if (result.citations.length > 0) {
    const citationsEl = document.createElement("ul");
    citationsEl.className = "chat-citations";
    for (const citation of result.citations) {
      const item = document.createElement("li");
      const parts = [citation.filename || citation.document_id];
      if (citation.page_number != null) parts.push(`page ${citation.page_number}`);
      if (citation.section) parts.push(citation.section);
      item.textContent = `[${citation.citation_id}] ${parts.join(" · ")}`;
      citationsEl.appendChild(item);
    }
    entry.appendChild(citationsEl);
  }

  if (result.chunks.length > 0) {
    const details = document.createElement("details");
    details.className = "chat-chunks";

    const summary = document.createElement("summary");
    summary.textContent = `Retrieved chunks (${result.chunks.length})`;
    details.appendChild(summary);

    for (const chunk of result.chunks) {
      const chunkEl = document.createElement("p");
      chunkEl.className = "chat-chunk-text";
      chunkEl.textContent = `[${chunk.citation_id}] ${chunk.text}`;
      details.appendChild(chunkEl);
    }

    entry.appendChild(details);
  }

  logEl.appendChild(entry);
  logEl.scrollTop = logEl.scrollHeight;
}

async function handleSubmit(event) {
  event.preventDefault();
  clearError();

  const query = inputEl.value.trim();
  if (!query) return;

  inputEl.value = "";
  inputEl.disabled = true;
  submitBtn.disabled = true;
  submitBtn.textContent = "Thinking...";

  try {
    const result = await sendChatMessage(currentLibraryId, query);
    appendMessage(query, result);
  } catch (err) {
    showError(`Chat request failed: ${err.message}`);
  } finally {
    inputEl.disabled = false;
    submitBtn.disabled = false;
    submitBtn.textContent = "Ask";
    inputEl.focus();
  }
}

export function showChatFor(libraryId) {
  currentLibraryId = libraryId;
  clearError();
  logEl.innerHTML = "";
  sectionEl.hidden = false;
  refreshDemoBadge();
}

export function hideChat() {
  currentLibraryId = null;
  sectionEl.hidden = true;
  logEl.innerHTML = "";
}

export function initChat() {
  sectionEl = document.querySelector("#chat-section");
  demoBadgeEl = document.querySelector("#chat-demo-badge");
  logEl = document.querySelector("#chat-log");
  formEl = document.querySelector("#chat-form");
  inputEl = document.querySelector("#chat-input");
  submitBtn = document.querySelector("#chat-submit-btn");
  errorEl = document.querySelector("#chat-error-msg");

  formEl.addEventListener("submit", handleSubmit);

  hideChat();
}
