// Thin wrapper around the local FastAPI server started with:
//   python -m local_knowledge_library.api
// Matches the LKL_API_HOST/LKL_API_PORT defaults in .env.example.
export const API_BASE = "http://127.0.0.1:8000/api/v1";

// FastAPI error bodies are JSON ({"detail": "..."} for a raised
// HTTPException, or {"detail": [{"msg": "...", ...}, ...]} for a Pydantic
// validation error) - show that message directly instead of the raw JSON
// text, falling back to the raw body if it's not that shape.
function describeErrorBody(rawText) {
  try {
    const parsed = JSON.parse(rawText);
    const detail = parsed.detail;
    if (typeof detail === "string") {
      return detail;
    }
    if (Array.isArray(detail)) {
      return detail.map((item) => item.msg || JSON.stringify(item)).join("; ");
    }
  } catch {
    // Not JSON - fall through to the raw text.
  }
  return rawText;
}

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`${options.method || "GET"} ${path} failed with status ${response.status}: ${describeErrorBody(body)}`);
  }

  if (response.status === 204) {
    return null;
  }

  return response.json();
}

export function getHealth() {
  return request("/health");
}

export function getModels() {
  return request("/models");
}

export function listLibraries() {
  return request("/libraries");
}

export function createLibrary(data) {
  return request("/libraries", { method: "POST", body: JSON.stringify(data) });
}

export function getLibrary(libraryId) {
  return request(`/libraries/${encodeURIComponent(libraryId)}`);
}

export function updateLibrary(libraryId, data) {
  return request(`/libraries/${encodeURIComponent(libraryId)}`, {
    method: "PATCH",
    body: JSON.stringify(data),
  });
}

export function deleteLibrary(libraryId) {
  return request(`/libraries/${encodeURIComponent(libraryId)}?confirm=true`, {
    method: "DELETE",
  });
}

export function listSources(libraryId) {
  return request(`/libraries/${encodeURIComponent(libraryId)}/sources`);
}

export function addSource(libraryId, sourcePath) {
  return request(`/libraries/${encodeURIComponent(libraryId)}/sources`, {
    method: "POST",
    body: JSON.stringify({ source_path: sourcePath }),
  });
}

export function removeSource(libraryId, sourceId) {
  return request(
    `/libraries/${encodeURIComponent(libraryId)}/sources/${encodeURIComponent(sourceId)}`,
    { method: "DELETE" },
  );
}

export function ingestLibrary(libraryId) {
  return request(`/libraries/${encodeURIComponent(libraryId)}/ingest`, {
    method: "POST",
  });
}

export function sendChatMessage(libraryId, query) {
  return request(`/libraries/${encodeURIComponent(libraryId)}/chat`, {
    method: "POST",
    body: JSON.stringify({ query }),
  });
}

export function listEvalCases(libraryId) {
  return request(`/libraries/${encodeURIComponent(libraryId)}/eval-cases`);
}

export function addEvalCase(libraryId, question, expectedKeyword) {
  return request(`/libraries/${encodeURIComponent(libraryId)}/eval-cases`, {
    method: "POST",
    body: JSON.stringify({ question, expected_keyword: expectedKeyword }),
  });
}

export function removeEvalCase(libraryId, evalCaseId) {
  return request(
    `/libraries/${encodeURIComponent(libraryId)}/eval-cases/${encodeURIComponent(evalCaseId)}`,
    { method: "DELETE" },
  );
}

export function runEvaluation(libraryId) {
  return request(`/libraries/${encodeURIComponent(libraryId)}/evaluate`, {
    method: "POST",
  });
}

export function listEvalRuns(libraryId) {
  return request(`/libraries/${encodeURIComponent(libraryId)}/eval-runs`);
}
