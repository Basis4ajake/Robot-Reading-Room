import os
from pathlib import Path

from local_knowledge_library.models import (
    Chunk,
    Citation,
    DocumentMetadata,
    IngestionState,
    StructureMetadata,
    compute_content_hash,
    make_chunk_id,
    make_source_id,
)


def test_compute_content_hash_is_stable():
    value = "Hello world"
    first = compute_content_hash(value)
    second = compute_content_hash(value)
    assert first == second
    assert len(first) == 64


def test_chunk_and_citation_provenance():
    chunk = Chunk(
        chunk_id=make_chunk_id(),
        library_id="lib-1",
        source_id="source-1",
        document_id="doc-1",
        text="Example chunk text.",
        metadata={"page_number": "1", "section": "Intro"},
        citation_id="cite-1",
    )
    citation = Citation(
        citation_id=chunk.citation_id,
        library_id=chunk.library_id,
        source_id=chunk.source_id,
        document_id=chunk.document_id,
        chunk_id=chunk.chunk_id,
        page_number=int(chunk.metadata["page_number"]),
        section=chunk.metadata["section"],
    )
    assert citation.citation_id == "cite-1"
    assert citation.chunk_id == chunk.chunk_id
    assert citation.section == "Intro"


def test_document_structure_metadata():
    meta = StructureMetadata(page_number=10, chapter="One", section="First")
    data = meta.to_dict()
    assert data["page_number"] == 10
    assert data["chapter"] == "One"
    restored = StructureMetadata.from_dict(data)
    assert restored.section == "First"
