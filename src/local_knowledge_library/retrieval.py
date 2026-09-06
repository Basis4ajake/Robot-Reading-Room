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

    def hybrid_search(self, query: str, top_k: int = 5) -> List[Chunk]:
        """Semantic results first, keyword results filling any remaining
        slots - capped at a single top_k total. (Previously took separate
        top_k_semantic/top_k_keyword and unioned both in full, which could
        silently return up to 2x top_k chunks - quietly doubling prompt
        size versus what the caller actually asked for. Not currently
        called by anything, so no external behavior changes.)
        """
        semantic_results = self.semantic_search(query, top_k)
        combined = {chunk.chunk_id: chunk for chunk in semantic_results}
        if self.keyword_searcher is not None and len(combined) < top_k:
            for chunk in self.keyword_search(query, top_k):
                if chunk.chunk_id in combined:
                    continue
                combined[chunk.chunk_id] = chunk
                if len(combined) >= top_k:
                    break
        return list(combined.values())
