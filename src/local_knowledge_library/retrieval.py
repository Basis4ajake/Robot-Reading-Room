from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .abstracts import Embedder, KeywordSearcher, Reranker, VectorStore
from .models import Chunk

# Standard reciprocal-rank-fusion damping constant (the usual default in RRF
# literature/implementations) - large enough that a couple of rank
# positions near the top don't swing a score wildly, small enough that
# rank still matters more than raw list length.
_RRF_K = 60


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
        results = self._raw_semantic_search(query, top_k)
        if self.reranker is not None:
            results = list(self.reranker.rerank(query, results))
            if self.debug:
                print(f"[Retriever] reranked results: {len(results)} chunks")
        return results

    def _raw_semantic_search(self, query: str, top_k: int) -> List[Chunk]:
        """Embedding similarity only, with no reranking applied. Used
        directly by hybrid_search() so its RRF fusion combines a genuine
        semantic ranking with the keyword ranking - if it used
        semantic_search() instead, a lexical reranker (see
        LexicalOverlapReranker) would have already re-sorted this "semantic"
        list by literal term overlap before fusion ever saw it, silently
        turning "fuse semantic + lexical signals" into "fuse lexical +
        lexical signals" and burying genuine paraphrase-style semantic
        matches that happen to share no literal words with the query -
        exactly the case embeddings exist to catch.
        """
        query_embedding = self.embedder.embed_text([query])[0]
        results = self.vector_store.search(query_embedding, top_k)
        if self.debug:
            print(f"[Retriever] semantic search results: {len(results)} chunks")
        return results

    def keyword_search(self, query: str, top_k: int = 5) -> List[Chunk]:
        if self.keyword_searcher is None:
            return []
        return list(self.keyword_searcher.search(query, top_k))

    def hybrid_search(self, query: str, top_k: int = 5) -> List[Chunk]:
        """Fuse semantic and keyword result lists by reciprocal rank
        fusion: each chunk's score is the sum of 1/(_RRF_K + rank) across
        every list it appears in, so a chunk ranked decently by both
        signals can outrank one that's #1 in only one of them. (Previously
        did a naive "semantic first, keyword fills remaining slots" union,
        which never let keyword evidence promote a chunk over a weaker
        semantic-only match, and separately used to take independent
        top_k_semantic/top_k_keyword and union both in full - capped here
        at a single top_k total either way.)

        Uses the RAW (unreranked) semantic order as RRF's semantic leg -
        see _raw_semantic_search()'s docstring for why. If a reranker is
        configured, it's applied AFTER fusion has already picked the
        final top_k set, as a same-size reordering pass over that set -
        not before fusion (would corrupt the semantic leg, see above) and
        not over the full pre-truncation candidate pool (would let a
        crude lexical re-sort override RRF's fused ranking entirely,
        defeating the point of fusing two signals in the first place).

        Pulls a wider candidate pool (2x top_k) from each source before
        fusing, so fusion has more than top_k candidates per list to
        actually combine - fusing two already-top_k-truncated lists would
        rarely change anything.
        """
        pool_k = top_k * 2
        semantic_results = self._raw_semantic_search(query, pool_k)
        keyword_results = self.keyword_search(query, pool_k) if self.keyword_searcher is not None else []

        scores: Dict[str, float] = {}
        chunks_by_id: Dict[str, Chunk] = {}
        for result_list in (semantic_results, keyword_results):
            for rank, chunk in enumerate(result_list):
                scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (_RRF_K + rank)
                chunks_by_id[chunk.chunk_id] = chunk

        # dict preserves insertion order, so among tied scores (e.g. a
        # chunk appearing in only one list at the same rank as another)
        # semantic results still come first, matching the old default.
        ranked_ids = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)
        fused = [chunks_by_id[chunk_id] for chunk_id in ranked_ids[:top_k]]

        if self.reranker is not None:
            fused = list(self.reranker.rerank(query, fused))
        return fused
