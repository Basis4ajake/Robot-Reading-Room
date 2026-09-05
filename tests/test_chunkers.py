from local_knowledge_library.chunkers import ParagraphChunker
from local_knowledge_library.models import DocumentMetadata, StructureMetadata


def _words(n: int, prefix: str = "w") -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


def _make_document(text: str, page_boundaries=None, page_number=None) -> DocumentMetadata:
    return DocumentMetadata(
        source_id="src-1",
        library_id="lib-1",
        document_id="doc-1",
        source_path="/tmp/doc.pdf",
        filename="doc.pdf",
        file_type="pdf",
        title=None,
        author=None,
        publisher=None,
        publication_date=None,
        edition=None,
        isbn=None,
        content_hash="",
        structure=StructureMetadata(page_number=page_number),
        text=text,
        page_boundaries=page_boundaries,
    )


def test_paragraph_chunker_attributes_one_paragraph_per_page_to_that_page():
    # Three pages, one paragraph each - exactly the shape confirmed against
    # a real 227-page book (see project_embedding_bug_2026_09_05 memory).
    text = "\n\n".join(["Page one text.", "Page two text.", "Page three text."])
    document = _make_document(text, page_boundaries=[1, 1, 1])

    chunks = ParagraphChunker().chunk(document)

    assert [c.metadata["page_number"] for c in chunks] == ["1", "2", "3"]


def test_paragraph_chunker_handles_multi_paragraph_and_blank_pages():
    # Page 1 has two paragraphs, page 2 is blank (extracted no text), page 3
    # has one paragraph.
    text = "\n\n".join(["P1 para A.", "P1 para B.", "P3 para A."])
    document = _make_document(text, page_boundaries=[2, 0, 1])

    chunks = ParagraphChunker().chunk(document)

    assert [c.metadata["page_number"] for c in chunks] == ["1", "1", "3"]


def test_paragraph_chunker_warns_on_page_boundaries_desync(capsys):
    # page_boundaries claims 2 paragraphs total, but the text only has 1 -
    # PdfLoader and this method's split logic have desynced. Must still
    # attribute something (best-effort) but warn loudly rather than silently
    # misattribute with no indication anything is wrong.
    document = _make_document("Only one paragraph.", page_boundaries=[2])

    chunks = ParagraphChunker().chunk(document)

    assert len(chunks) == 1  # best-effort: still produced a citable chunk
    warning = capsys.readouterr().out
    assert "WARNING" in warning
    assert "page_boundaries" in warning


def test_paragraph_chunker_falls_back_to_document_page_number_without_boundaries():
    text = "Only paragraph."
    document = _make_document(text, page_boundaries=None, page_number=42)

    chunks = ParagraphChunker().chunk(document)

    assert chunks[0].metadata["page_number"] == "42"


def test_paragraph_chunker_leaves_small_paragraphs_alone_by_default():
    # chunk_size=None (the default) preserves the old behavior.
    document = _make_document(_words(50), page_boundaries=[1])
    chunks = ParagraphChunker().chunk(document)
    assert len(chunks) == 1


def test_paragraph_chunker_splits_oversized_paragraph_to_chunk_size():
    # One "page" with 250 words, chunk_size=100, overlap=20 -> step of 80
    # words: windows at 0-99, 80-179, 160-249 = 3 sub-chunks.
    document = _make_document(_words(250), page_boundaries=[1])
    chunker = ParagraphChunker(chunk_size=100, chunk_overlap=20)

    chunks = chunker.chunk(document)

    assert len(chunks) == 3
    for chunk in chunks:
        assert len(chunk.text.split()) <= 100
    # Overlap: the last 20 words of chunk 1 should equal the first 20 of chunk 2.
    assert chunks[0].text.split()[-20:] == chunks[1].text.split()[:20]


def test_paragraph_chunker_clamps_overlap_that_would_overshoot_chunk_size():
    # overlap >= chunk_size would otherwise collapse the sliding-window step
    # to 1, exploding a paragraph into thousands of near-duplicate chunks.
    document = _make_document(_words(500), page_boundaries=[1])
    chunker = ParagraphChunker(chunk_size=100, chunk_overlap=300)

    chunks = chunker.chunk(document)

    # Clamped to overlap=chunk_size-1=99 -> step=1 is what we're guarding
    # against, so assert it did NOT produce anywhere near one chunk per word.
    assert len(chunks) < 20


def test_paragraph_chunker_clamps_negative_overlap_to_avoid_dropping_words():
    # A negative overlap would make step > chunk_size, silently skipping
    # whole spans of text that never appear in any chunk.
    document = _make_document(_words(1000), page_boundaries=[1])
    chunker = ParagraphChunker(chunk_size=100, chunk_overlap=-50)

    chunks = chunker.chunk(document)

    covered_words: set[str] = set()
    for chunk in chunks:
        covered_words.update(chunk.text.split())
    all_words = set(_words(1000).split())
    assert covered_words == all_words


def test_paragraph_chunker_attributes_all_sub_chunks_to_the_source_page():
    # A 250-word "page 2" (after a short page 1) split into multiple pieces -
    # every piece must still cite page 2, never a neighboring page.
    text = "\n\n".join(["Short page one.", _words(250)])
    document = _make_document(text, page_boundaries=[1, 1])
    chunker = ParagraphChunker(chunk_size=100, chunk_overlap=0)

    chunks = chunker.chunk(document)

    page_one_chunks = [c for c in chunks if c.text == "Short page one."]
    page_two_chunks = [c for c in chunks if c.text != "Short page one."]
    assert len(page_one_chunks) == 1
    assert page_one_chunks[0].metadata["page_number"] == "1"
    assert len(page_two_chunks) > 1
    assert all(c.metadata["page_number"] == "2" for c in page_two_chunks)


def test_paragraph_chunker_produces_stable_chunk_ids_across_runs():
    # Determinism matters: IngestionPipeline relies on identical input
    # producing identical chunk_ids so SQLite REPLACE cleanly overwrites
    # rather than accumulating duplicates.
    document = _make_document(_words(250), page_boundaries=[1])
    chunker = ParagraphChunker(chunk_size=100, chunk_overlap=20)

    ids_a = [c.chunk_id for c in chunker.chunk(document)]
    ids_b = [c.chunk_id for c in chunker.chunk(document)]

    assert ids_a == ids_b
    assert len(set(ids_a)) == len(ids_a)
