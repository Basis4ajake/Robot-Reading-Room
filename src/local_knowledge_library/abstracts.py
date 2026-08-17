from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Sequence

from .models import DocumentMetadata, Chunk, Citation

class DocumentLoader(ABC):
    @abstractmethod
    def supported_extensions(self) -> Sequence[str]:
        raise NotImplementedError

    @abstractmethod
    def load(self, source_path: str) -> Iterable[DocumentMetadata]:
        raise NotImplementedError

class Chunker(ABC):
    @abstractmethod
    def chunk(self, document: DocumentMetadata) -> Sequence[Chunk]:
        raise NotImplementedError

class Embedder(ABC):
    @abstractmethod
    def embed_text(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        raise NotImplementedError

class VectorStore(ABC):
    @abstractmethod
    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def remove(self, chunk_ids: Sequence[str]) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, query_embedding: Sequence[float], top_k: int) -> Sequence[Chunk]:
        raise NotImplementedError

class KeywordSearcher(ABC):
    @abstractmethod
    def search(self, query: str, top_k: int) -> Sequence[Chunk]:
        raise NotImplementedError

class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, candidates: Sequence[Chunk]) -> Sequence[Chunk]:
        raise NotImplementedError

class LLMProvider(Embedder, ABC):
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        raise NotImplementedError

class QueryPlannerStrategy(ABC):
    @abstractmethod
    def plan(self, query: str) -> str:
        raise NotImplementedError
