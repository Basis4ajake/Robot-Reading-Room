from local_knowledge_library.providers.ollama_providers import (
    DummyEmbedder,
    InMemoryVectorStore,
    LexicalOverlapReranker,
    SimpleKeywordSearcher,
)
from local_knowledge_library.retrieval import Retriever
from local_knowledge_library.models import Chunk


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        library_id="lib-1",
        source_id="source-1",
        document_id="doc-1",
        text=text,
        metadata={},
        citation_id=f"cite-{chunk_id}",
    )


def test_retriever_semantic_search():
    vector_store = InMemoryVectorStore()
    chunks = [
        _chunk("chunk-1", "The quick brown fox jumps over the lazy dog."),
        _chunk("chunk-2", "A different sentence with dogs and foxes."),
    ]
    embeddings = DummyEmbedder().embed_text([chunk.text for chunk in chunks])
    vector_store.add(chunks, embeddings)
    retriever = Retriever(vector_store=vector_store, embedder=DummyEmbedder(), debug=False)
    results = retriever.semantic_search("fox dog", top_k=2)
    assert len(results) == 2
    assert results[0].chunk_id in {"chunk-1", "chunk-2"}


def test_hybrid_search_falls_back_to_semantic_only_with_no_keyword_searcher():
    vector_store = InMemoryVectorStore()
    chunks = [_chunk("chunk-1", "fox dog"), _chunk("chunk-2", "cat mouse")]
    vector_store.add(chunks, DummyEmbedder().embed_text([c.text for c in chunks]))
    retriever = Retriever(vector_store=vector_store, embedder=DummyEmbedder())

    results = retriever.hybrid_search("fox", top_k=2)

    assert len(results) == 2


def test_hybrid_search_includes_a_keyword_only_match():
    vector_store = InMemoryVectorStore()
    chunks = [_chunk(f"c{i}", f"filler text {i}") for i in range(3)]
    chunks.append(_chunk("keyword-hit", "a distinctive keyword appears here"))
    vector_store.add(chunks, DummyEmbedder().embed_text([c.text for c in chunks]))
    keyword_searcher = SimpleKeywordSearcher(lambda: chunks)
    retriever = Retriever(vector_store=vector_store, embedder=DummyEmbedder(), keyword_searcher=keyword_searcher)

    results = retriever.hybrid_search("distinctive", top_k=1)

    assert len(results) == 1


def test_hybrid_search_ignores_a_real_keyword_searcher_that_finds_nothing():
    """Regression test using the REAL SimpleKeywordSearcher (not a fixed-
    order stub): a query whose terms appear in none of the chunks must
    not perturb the final ranking at all. Before SimpleKeywordSearcher
    filtered out zero-score chunks, it padded its result up to top_k with
    completely irrelevant chunks in arbitrary (chunk-list) order - RRF
    then treated that arbitrary order as a real rank and mixed it into
    the fused score, letting content that matches nothing outrank a
    genuine semantic match."""
    vector_store = InMemoryVectorStore()
    chunks = [_chunk(f"c{i}", f"unrelated filler content number {i}") for i in range(10)]
    vector_store.add(chunks, DummyEmbedder().embed_text([c.text for c in chunks]))
    keyword_searcher = SimpleKeywordSearcher(lambda: chunks)
    retriever = Retriever(vector_store=vector_store, embedder=DummyEmbedder(), keyword_searcher=keyword_searcher)

    with_keyword_searcher = retriever.hybrid_search("nonexistentterm", top_k=4)
    pure_semantic = retriever.semantic_search("nonexistentterm", top_k=4)

    assert [c.chunk_id for c in with_keyword_searcher] == [c.chunk_id for c in pure_semantic]


def test_hybrid_search_never_returns_more_than_top_k():
    """Previously took separate top_k_semantic/top_k_keyword and unioned
    both in full - could silently return up to 2x top_k chunks, doubling
    prompt size versus what the caller actually asked for."""
    vector_store = InMemoryVectorStore()
    chunks = [_chunk(f"c{i}", f"distinctive keyword {i}") for i in range(10)]
    vector_store.add(chunks, DummyEmbedder().embed_text([c.text for c in chunks]))
    keyword_searcher = SimpleKeywordSearcher(lambda: chunks)
    retriever = Retriever(vector_store=vector_store, embedder=DummyEmbedder(), keyword_searcher=keyword_searcher)

    results = retriever.hybrid_search("distinctive keyword", top_k=4)

    assert len(results) <= 4


class _FixedOrderVectorStore:
    """Returns a fixed chunk order regardless of the query embedding, so a
    test can dictate exact semantic ranks instead of relying on real
    cosine similarity."""

    def __init__(self, ordered_chunks):
        self._ordered = ordered_chunks

    def search(self, query_embedding, top_k):
        return self._ordered[:top_k]


class _FixedOrderKeywordSearcher:
    def __init__(self, ordered_chunks):
        self._ordered = ordered_chunks

    def search(self, query, top_k):
        return self._ordered[:top_k]


def test_hybrid_search_rrf_promotes_a_chunk_strong_in_both_lists():
    """The core RRF promise: a chunk ranked well in both semantic and
    keyword search should be able to outrank a chunk that's #1 in only
    one of them - the old "semantic first, keyword only fills leftover
    slots" union could never do this, since a full semantic top_k list
    left no room for keyword evidence to matter at all."""
    chunk_x = _chunk("x", "semantic top pick, lexically irrelevant")
    chunk_y = _chunk("y", "decent in both signals")
    chunk_z = _chunk("z", "decent in both signals too")

    # Semantic ranks: x=0, z=1, y=2. Keyword ranks: z=0, y=1 (x never
    # shows up in keyword results at all - it's lexically irrelevant).
    vector_store = _FixedOrderVectorStore([chunk_x, chunk_z, chunk_y])
    keyword_searcher = _FixedOrderKeywordSearcher([chunk_z, chunk_y])
    retriever = Retriever(
        vector_store=vector_store, embedder=DummyEmbedder(), keyword_searcher=keyword_searcher
    )

    results = retriever.hybrid_search("query", top_k=2)

    # z and y (present in both lists) win over x (semantic-only #1).
    assert {c.chunk_id for c in results} == {"z", "y"}
    assert results[0].chunk_id == "z"


def test_hybrid_search_rrf_uses_true_semantic_rank_even_with_a_reranker_configured():
    """Regression test: hybrid_search() must feed RRF the RAW semantic
    order, not the reranker-adjusted one. If it fed semantic_search()'s
    (reranked) output into RRF instead, a lexical reranker would silently
    resort the "semantic" leg by literal term overlap before fusion ever
    ran, so a chunk that's genuinely the #1 semantic (embedding) match
    but shares zero literal words with the query could lose to a weaker
    semantic match that merely happens to contain the query's words -
    exactly backwards from what embeddings exist to catch.

    No keyword_searcher here deliberately, to isolate this specific
    failure mode (semantic-leg corruption) from RRF's separate, already-
    covered promotion behavior (test above) when a keyword list is also
    in play.
    """
    paraphrase_match = _chunk("paraphrase", "a rewording with no literal overlap")
    literal_match = _chunk("literal", "distinctive keyword appears literally")

    # True semantic (cosine) order: paraphrase_match ranks #1, literal_match #2.
    vector_store = _FixedOrderVectorStore([paraphrase_match, literal_match])
    retriever = Retriever(
        vector_store=vector_store,
        embedder=DummyEmbedder(),
        reranker=LexicalOverlapReranker(),
    )

    results = retriever.hybrid_search("distinctive keyword", top_k=1)

    assert results[0].chunk_id == "paraphrase"


def test_lexical_overlap_reranker_boosts_literal_query_term_matches():
    reranker = LexicalOverlapReranker()
    chunks = [
        _chunk("no-match", "an entirely unrelated sentence"),
        _chunk("match", "the quick brown fox jumps over the lazy dog"),
    ]

    result = reranker.rerank("fox", chunks)

    assert result[0].chunk_id == "match"


def test_lexical_overlap_reranker_stable_on_ties():
    """Equal (including zero) overlap must keep original relative order,
    not get shuffled - this is a boost on top of the caller's existing
    ranking, not a full independent re-sort."""
    reranker = LexicalOverlapReranker()
    chunks = [_chunk("a", "no overlap here"), _chunk("b", "also nothing relevant")]

    result = reranker.rerank("nonexistent-term", chunks)

    assert [c.chunk_id for c in result] == ["a", "b"]
