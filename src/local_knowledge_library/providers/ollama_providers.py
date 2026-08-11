from __future__ import annotations

import random
from typing import Sequence

from ..abstracts import Embedder, LLMProvider


class DummyEmbedder(Embedder):
    def embed_text(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [[float(sum(ord(ch) for ch in text) % 100) / 100.0 for _ in range(16)] for text in texts]


class DummyLLMProvider(LLMProvider):
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        return "This is a dummy response. The system retrieved evidence from the provided citations."

    def embed_text(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return DummyEmbedder().embed_text(texts)


class OllamaQwenProvider(LLMProvider, Embedder):
    def __init__(self, model_name: str = "qwen-7b", embedding_model: str = "mpt-7b-instruct"):
        self.model_name = model_name
        self.embedding_model = embedding_model

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        try:
            from ollama import Ollama
        except ImportError as exc:
            raise RuntimeError("Ollama SDK is required for OllamaQwenProvider") from exc
        client = Ollama()
        response = client.create(model=self.model_name, prompt=prompt, max_tokens=max_tokens)
        return response.text

    def embed_text(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        try:
            from ollama import Ollama
        except ImportError as exc:
            raise RuntimeError("Ollama SDK is required for OllamaQwenProvider") from exc
        client = Ollama()
        embeddings = client.embeddings(model=self.embedding_model, input=list(texts))
        return [item.embedding for item in embeddings]


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
    def __init__(self, chunks: Sequence["Chunk"]):
        self.chunks = list(chunks)

    def search(self, query: str, top_k: int) -> Sequence["Chunk"]:
        lowercase = query.lower()
        scored = []
        for chunk in self.chunks:
            score = sum(lowercase.count(term) for term in lowercase.split())
            scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [chunk for _, chunk in scored[:top_k]]


class DummyReranker:
    def rerank(self, query: str, candidates: Sequence["Chunk"]):
        return candidates
