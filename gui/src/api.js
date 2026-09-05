// Thin wrapper around the local FastAPI server started with:
//   python -m local_knowledge_library.api
// Matches the LKL_API_HOST/LKL_API_PORT defaults in .env.example.
export const API_BASE = "http://127.0.0.1:8000/api/v1";

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`${options.method || "GET"} ${path} failed with status ${response.status}: ${body}`);
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
