import { listEvalCases, addEvalCase, removeEvalCase, runEvaluation, listEvalRuns } from "./api.js";

let sectionEl;
let formEl;
let questionInput;
let keywordInput;
let caseListEl;
let runBtn;
let historyEl;
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

function renderCaseList(cases) {
  caseListEl.innerHTML = "";

  if (cases.length === 0) {
    const empty = document.createElement("li");
    empty.className = "library-empty";
    empty.textContent = "No eval questions yet. Add one below, then Run Evaluation.";
    caseListEl.appendChild(empty);
    return;
  }

  for (const evalCase of cases) {
    const item = document.createElement("li");
    item.className = "library-item library-item-btn";

    const info = document.createElement("span");
    info.className = "library-name";
    info.textContent = `${evalCase.question} → expects "${evalCase.expected_keyword}"`;

    const removeBtn = document.createElement("button");
    removeBtn.type = "button";
    removeBtn.className = "danger";
    removeBtn.textContent = "Remove";
    removeBtn.addEventListener("click", () => handleRemoveCase(evalCase.eval_case_id));

    item.appendChild(info);
    item.appendChild(removeBtn);
    caseListEl.appendChild(item);
  }
}

function formatTimestamp(isoString) {
  try {
    return new Date(isoString).toLocaleString();
  } catch {
    return isoString;
  }
}

function renderRunHistory(runs) {
  historyEl.innerHTML = "";

  if (runs.length === 0) {
    const empty = document.createElement("p");
    empty.className = "library-empty";
    empty.textContent = "No evaluation runs yet.";
    historyEl.appendChild(empty);
    return;
  }

  for (const run of runs) {
    const details = document.createElement("details");
    details.className = "eval-run";

    const summary = document.createElement("summary");
    const passBadge = document.createElement("span");
    passBadge.className = run.passed_count === run.total_count ? "badge badge-pass" : "badge badge-fail";
    passBadge.textContent = `${run.passed_count}/${run.total_count} passed`;
    summary.appendChild(passBadge);
    summary.append(
      ` ${formatTimestamp(run.timestamp)} — chunk_size=${run.chunk_size}, chunk_overlap=${run.chunk_overlap}, ` +
        `top_k=${run.top_k}, llm_model=${run.llm_model}, embedding_model=${run.embedding_model ?? "none"}`,
    );
    details.appendChild(summary);

    const table = document.createElement("table");
    table.className = "eval-result-table";
    const header = document.createElement("tr");
    for (const label of ["Question", "Expected", "Evidence", "Answer text", "Answer"]) {
      const th = document.createElement("th");
      th.textContent = label;
      header.appendChild(th);
    }
    table.appendChild(header);

    for (const result of run.results) {
      const row = document.createElement("tr");

      const question = document.createElement("td");
      question.textContent = result.question;
      row.appendChild(question);

      const keyword = document.createElement("td");
      keyword.textContent = result.expected_keyword;
      row.appendChild(keyword);

      const evidence = document.createElement("td");
      evidence.textContent = result.keyword_in_citations ? "✓" : "✗";
      row.appendChild(evidence);

      const answerText = document.createElement("td");
      answerText.textContent = result.keyword_in_answer ? "✓" : "✗";
      row.appendChild(answerText);

      const answer = document.createElement("td");
      answer.className = "eval-result-answer";
      answer.textContent = result.answer;
      row.appendChild(answer);

      table.appendChild(row);
    }

    details.appendChild(table);
    historyEl.appendChild(details);
  }
}

async function refreshCases() {
  if (!currentLibraryId) return;
  try {
    const cases = await listEvalCases(currentLibraryId);
    renderCaseList(cases);
  } catch (err) {
    showError(`Could not load eval questions: ${err.message}`);
  }
}

async function refreshHistory() {
  if (!currentLibraryId) return;
  try {
    const runs = await listEvalRuns(currentLibraryId);
    renderRunHistory(runs);
  } catch (err) {
    showError(`Could not load evaluation history: ${err.message}`);
  }
}

async function handleAddCase(event) {
  event.preventDefault();
  clearError();

  const question = questionInput.value.trim();
  const keyword = keywordInput.value.trim();
  if (!question || !keyword) return;

  try {
    await addEvalCase(currentLibraryId, question, keyword);
    questionInput.value = "";
    keywordInput.value = "";
    await refreshCases();
  } catch (err) {
    showError(`Could not add eval question: ${err.message}`);
  }
}

async function handleRemoveCase(evalCaseId) {
  clearError();
  try {
    await removeEvalCase(currentLibraryId, evalCaseId);
    await refreshCases();
  } catch (err) {
    showError(`Could not remove eval question: ${err.message}`);
  }
}

async function handleRunEvaluation() {
  clearError();
  runBtn.disabled = true;
  runBtn.textContent = "Running...";

  try {
    await runEvaluation(currentLibraryId);
    await refreshHistory();
  } catch (err) {
    showError(`Evaluation run failed: ${err.message}`);
  } finally {
    runBtn.disabled = false;
    runBtn.textContent = "Run Evaluation";
  }
}

export function showEvalFor(libraryId) {
  currentLibraryId = libraryId;
  clearError();
  sectionEl.hidden = false;
  refreshCases();
  refreshHistory();
}

export function hideEval() {
  currentLibraryId = null;
  sectionEl.hidden = true;
  caseListEl.innerHTML = "";
  historyEl.innerHTML = "";
}

export function initEval() {
  sectionEl = document.querySelector("#eval-section");
  formEl = document.querySelector("#eval-case-form");
  questionInput = document.querySelector("#eval-question-input");
  keywordInput = document.querySelector("#eval-keyword-input");
  caseListEl = document.querySelector("#eval-case-list");
  runBtn = document.querySelector("#run-eval-btn");
  historyEl = document.querySelector("#eval-run-history");
  errorEl = document.querySelector("#eval-error-msg");

  formEl.addEventListener("submit", handleAddCase);
  runBtn.addEventListener("click", handleRunEvaluation);

  hideEval();
}
