from __future__ import annotations

import random
from typing import Callable, Sequence

from ..abstracts import Embedder, LLMProvider
from ..models import Chunk

# Ollama's own default num_ctx (4096, confirmed empirically on this project's
# hardware/Ollama 0.32.7) is silently enforced via --context-shift: an
# over-length prompt gets old tokens dropped rather than an error, not a
# hard failure - so a growing RAG prompt (bigger chunk_size/top_k) can
# silently lose earlier retrieved chunks with no warning. Sizing num_ctx from
# the actual prompt at generate() time, rather than trusting the default,
# closes that off. 2 chars/token is deliberately conservative: this
# project's own chemistry-heavy content measured ~2.65 chars/token in
# practice (denser than typical English's ~4), so this errs toward
# overestimating tokens rather than under.
_CHARS_PER_TOKEN_ESTIMATE = 2
# A practical ceiling, not "the biggest any model supports" - llm_model is
# freely user-selectable (curated presets go up to llama3.1:8b's 128k native
# context), but this project's realistic RAG prompts don't need anywhere
# near that, and requesting a huge context has real memory/latency cost
# regardless of whether the model could technically serve it.
_NUM_CTX_BUCKETS = (2048, 4096, 8192, 16384, 32768)

_model_max_context_cache: dict[str, int] = {}


def _model_max_context(model_name: str) -> int:
    """Look up a model's real max context length via `ollama show`, so
    _estimate_num_ctx never requests more than a small model (e.g.
    phi3:mini) actually supports. Falls back to the largest bucket above if
    the lookup fails for any reason (Ollama unreachable, unexpected SDK
    response shape, model not pulled yet) - same conservative-but-not-fatal
    posture as the rest of this fallback chain."""
    if model_name in _model_max_context_cache:
        return _model_max_context_cache[model_name]
    fallback = _NUM_CTX_BUCKETS[-1]
    try:
        import ollama

        modelinfo = getattr(ollama.show(model_name), "modelinfo", None) or {}
        for key, value in modelinfo.items():
            if key.endswith(".context_length"):
                _model_max_context_cache[model_name] = int(value)
                return _model_max_context_cache[model_name]
    except Exception:
        pass
    _model_max_context_cache[model_name] = fallback
    return fallback


def _estimate_num_ctx(prompt: str, max_tokens: int, max_ctx: int = _NUM_CTX_BUCKETS[-1]) -> int:
    estimated_input_tokens = len(prompt) // _CHARS_PER_TOKEN_ESTIMATE
    needed = estimated_input_tokens + max_tokens + 256  # + safety margin
    candidates = [bucket for bucket in _NUM_CTX_BUCKETS if bucket <= max_ctx] or [max_ctx]
    for bucket in candidates:
        if needed <= bucket:
            return bucket
    return max(candidates)  # needed exceeds every allowed bucket; best effort, capped at max_ctx


class DummyEmbedder(Embedder):
    def embed_text(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [[float(sum(ord(ch) for ch in text) % 100) / 100.0 for _ in range(16)] for text in texts]


class DummyLLMProvider(LLMProvider):
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        return "This is a dummy response. The system retrieved evidence from the provided citations."

    def embed_text(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return DummyEmbedder().embed_text(texts)


class OllamaQwenProvider(LLMProvider, Embedder):
    def __init__(self, model_name: str = "qwen2:1.5b", embedding_model: str | None = None):
        self.model_name = model_name
        self.embedding_model = embedding_model or model_name

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        try:
            import ollama
        except ImportError as exc:
            raise RuntimeError("Ollama SDK is required for OllamaQwenProvider") from exc
        try:
            max_ctx = _model_max_context(self.model_name)
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={"num_predict": max_tokens, "num_ctx": _estimate_num_ctx(prompt, max_tokens, max_ctx)},
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to generate text from Ollama. Ensure Ollama is installed, running, and the model is available."
            ) from exc
        return getattr(response, "response", str(response))

    def embed_text(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        try:
            import ollama
        except ImportError as exc:
            raise RuntimeError("Ollama SDK is required for OllamaQwenProvider") from exc
        try:
            response = ollama.embed(model=self.embedding_model, input=list(texts))
        except Exception as exc:
            raise RuntimeError(
                "Failed to embed text with Ollama. Ensure Ollama is started with embedding support and the model supports embedding."
            ) from exc
        if hasattr(response, "embeddings"):
            return [list(item) for item in response.embeddings]
        if hasattr(response, "embedding"):
            return [list(response.embedding)]
        raise RuntimeError("Ollama embed response did not contain expected embedding fields.")


class SqliteVectorStore:
    def __init__(self, db_path: str):
        import sqlite3

        from pathlib import Path

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vectors (
                chunk_id TEXT PRIMARY KEY,
                chunk_json TEXT NOT NULL,
                embedding_json TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def add(self, chunks: Sequence["Chunk"], embeddings: Sequence[Sequence[float]]) -> None:
        import json

        with self.conn:
            for chunk, vector in zip(chunks, embeddings):
                self.conn.execute(
                    "REPLACE INTO vectors (chunk_id, chunk_json, embedding_json) VALUES (?, ?, ?)",
                    (chunk.chunk_id, json.dumps(chunk.to_dict(), ensure_ascii=False), json.dumps(vector, ensure_ascii=False)),
                )

    def search(self, query_embedding: Sequence[float], top_k: int) -> Sequence["Chunk"]:
        import json

        rows = self.conn.execute("SELECT chunk_json, embedding_json FROM vectors").fetchall()
        scored: list[tuple[float, "Chunk"]] = []
        for chunk_json, embedding_json in rows:
            chunk = Chunk.from_dict(json.loads(chunk_json))
            vector = json.loads(embedding_json)
            score = self._cosine_similarity(query_embedding, vector)
            scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [chunk for _, chunk in scored[:top_k]]

    def remove(self, chunk_ids: Sequence[str]) -> None:
        with self.conn:
            for chunk_id in chunk_ids:
                self.conn.execute("DELETE FROM vectors WHERE chunk_id = ?", (chunk_id,))

    def close(self) -> None:
        try:
            self.conn.commit()
        finally:
            self.conn.close()

    def __enter__(self) -> "SqliteVectorStore":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _cosine_similarity(self, a: Sequence[float], b: Sequence[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


class InMemoryVectorStore:
    def __init__(self):
        self.embeddings = {}
        self.chunks = {}

    def add(self, chunks: Sequence["Chunk"], embeddings: Sequence[Sequence[float]]) -> None:
        for chunk, vector in zip(chunks, embeddings):
            self.embeddings[chunk.chunk_id] = vector
            self.chunks[chunk.chunk_id] = chunk

    def search(self, query_embedding: Sequence[float], top_k: int) -> Sequence["Chunk"]:
        distances = []
        for chunk_id, vector in self.embeddings.items():
            score = self._cosine_similarity(query_embedding, vector)
            distances.append((score, self.chunks[chunk_id]))
        distances.sort(key=lambda item: item[0], reverse=True)
        return [chunk for _, chunk in distances[:top_k]]

    def remove(self, chunk_ids: Sequence[str]) -> None:
        for chunk_id in chunk_ids:
            self.embeddings.pop(chunk_id, None)
            self.chunks.pop(chunk_id, None)

    def _cosine_similarity(self, a: Sequence[float], b: Sequence[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


class SimpleKeywordSearcher:
    def __init__(self, chunks_provider: Callable[[], Sequence["Chunk"]]):
        # Takes a zero-arg callable rather than a static chunk list. Retriever
        # (and this searcher) live inside AppState's per-library runtime
        # cache, which is built once and reused across many requests - a
        # static snapshot taken at build time would silently go stale the
        # moment the next /ingest adds or removes chunks, reproducing the
        # exact "quietly wrong after a content change" bug class this
        # project has spent real effort hunting down elsewhere (embedding
        # signatures, chunking signatures, AppState locking). Calling this
        # fresh on every search reads whatever KnowledgeLibrary.open()
        # loaded for THIS request, which is already always current.
        self.chunks_provider = chunks_provider

    def search(self, query: str, top_k: int) -> Sequence["Chunk"]:
        terms = query.lower().split()
        scored = []
        for chunk in self.chunks_provider():
            chunk_text = chunk.text.lower()
            score = sum(chunk_text.count(term) for term in terms)
            scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [chunk for _, chunk in scored[:top_k]]


class DummyReranker:
    def rerank(self, query: str, candidates: Sequence["Chunk"]):
        return candidates
