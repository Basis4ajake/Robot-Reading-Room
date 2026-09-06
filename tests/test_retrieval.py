from local_knowledge_library.providers.ollama_providers import (
    DummyEmbedder,
    InMemoryVectorStore,
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


def test_hybrid_search_fills_remaining_slots_from_keyword_results():
    vector_store = InMemoryVectorStore()
    # DummyEmbedder's vectors are content-independent (see its own docs) -
    # semantic_search returns everything in essentially arbitrary order, so
    # cap semantic results below top_k to force keyword results to matter.
    chunks = [_chunk(f"c{i}", f"filler text {i}") for i in range(3)]
    chunks.append(_chunk("keyword-hit", "a distinctive keyword appears here"))
    vector_store.add(chunks, DummyEmbedder().embed_text([c.text for c in chunks]))
    keyword_searcher = SimpleKeywordSearcher(lambda: chunks)
    retriever = Retriever(vector_store=vector_store, embedder=DummyEmbedder(), keyword_searcher=keyword_searcher)

    results = retriever.hybrid_search("distinctive", top_k=1)

    assert len(results) == 1


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
