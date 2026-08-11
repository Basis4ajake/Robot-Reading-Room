from __future__ import annotations

from typing import List, Optional, Sequence

from .abstracts import Embedder, KeywordSearcher, Reranker, VectorStore
from .models import Chunk


class Retriever:
    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        keyword_searcher: KeywordSearcher | None = None,
        reranker: Reranker | None = None,
        debug: bool = False,
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.keyword_searcher = keyword_searcher
        self.reranker = reranker
        self.debug = debug

    def semantic_search(self, query: str, top_k: int = 5) -> List[Chunk]:
        query_embedding = self.embedder.embed_text([query])[0]
        results = self.vector_store.search(query_embedding, top_k)
        if self.debug:
            print(f"[Retriever] semantic search results: {len(results)} chunks")
        if self.reranker is not None:
            results = list(self.reranker.rerank(query, results))
            if self.debug:
                print(f"[Retriever] reranked results: {len(results)} chunks")
        return results

    def keyword_search(self, query: str, top_k: int = 5) -> List[Chunk]:
        if self.keyword_searcher is None:
            return []
        return list(self.keyword_searcher.search(query, top_k))

    def hybrid_search(self, query: str, top_k_semantic: int = 5, top_k_keyword: int = 5) -> List[Chunk]:
        semantic_results = self.semantic_search(query, top_k_semantic)
        keyword_results = self.keyword_search(query, top_k_keyword) if self.keyword_searcher else []
        combined = {chunk.chunk_id: chunk for chunk in semantic_results}
        for chunk in keyword_results:
            combined.setdefault(chunk.chunk_id, chunk)
        return list(combined.values())
