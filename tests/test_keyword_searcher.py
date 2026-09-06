from local_knowledge_library.models import Chunk
from local_knowledge_library.providers import SimpleKeywordSearcher


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        library_id="lib",
        source_id="src",
        document_id="doc",
        text=text,
        metadata={},
        citation_id=f"cite-{chunk_id}",
    )


def test_search_ranks_by_query_terms_found_in_the_chunk_text():
    """A prior bug counted query terms within the query itself, not the
    chunk text - every chunk scored identically regardless of content, so
    this never actually searched anything."""
    chunks = [
        _chunk("no-match", "This paragraph is about gardening and flowers."),
        _chunk("one-match", "This paragraph mentions chemistry once."),
        _chunk("two-match", "Chemistry is discussed here, and chemistry again."),
    ]
    searcher = SimpleKeywordSearcher(chunks)

    results = searcher.search("chemistry", top_k=3)

    assert [chunk.chunk_id for chunk in results] == ["two-match", "one-match", "no-match"]


def test_search_respects_top_k():
    chunks = [_chunk(f"c{i}", "keyword keyword keyword") for i in range(5)]
    searcher = SimpleKeywordSearcher(chunks)

    results = searcher.search("keyword", top_k=2)

    assert len(results) == 2
