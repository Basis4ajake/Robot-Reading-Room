import os
from pathlib import Path

from local_knowledge_library.models import (
    Chunk,
    Citation,
    DocumentMetadata,
    IngestionState,
    LibraryConfig,
    RecipeFact,
    StructureMetadata,
    compute_content_hash,
    make_chunk_id,
    make_recipe_fact_citation_id,
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


def test_library_config_normalizes_explicit_none_embedding_model():
    # The dataclass default only applies when the field is omitted - an
    # already-persisted config.json with "embedding_model": null (from
    # before this default existed) or a direct API POST with an explicit
    # null must still be normalized, or it silently reproduces the original
    # DummyEmbedder-fallback bug.
    config = LibraryConfig(library_id="lib", name="Lib", embedding_model=None)
    assert config.embedding_model == "nomic-embed-text"

    from_disk = LibraryConfig.from_dict(
        {"library_id": "lib", "name": "Lib", "embedding_model": None}
    )
    assert from_disk.embedding_model == "nomic-embed-text"


def test_library_config_defaults_recipe_extraction_off():
    assert LibraryConfig(library_id="lib", name="Lib").enable_recipe_extraction is False


def test_recipe_fact_roundtrip_and_ingredient_count():
    fact = RecipeFact(
        library_id="lib-1", source_id="src-1", document_id="doc-1",
        recipe_name="Gnocchi", ingredients=["potatoes", "cheese", "eggs"],
        step_count=3, source_excerpt="Prepare boiled potatoes...", page_number=42,
    )
    assert fact.ingredient_count == 3

    restored = RecipeFact.from_dict(fact.to_dict())
    assert restored == fact


def test_make_recipe_fact_citation_id_disambiguates_same_titled_recipes():
    shared = dict(library_id="l", source_id="s", document_id="doc-1", recipe_name="BISCUIT")
    first = RecipeFact(**shared, ingredients=["flour"], source_excerpt="First biscuit recipe...")
    second = RecipeFact(**shared, ingredients=["sugar"], source_excerpt="Second biscuit recipe...")

    assert make_recipe_fact_citation_id(first) != make_recipe_fact_citation_id(second)
    # Deterministic for the same fact.
    assert make_recipe_fact_citation_id(first) == make_recipe_fact_citation_id(first)


def test_document_structure_metadata():
    meta = StructureMetadata(page_number=10, chapter="One", section="First")
    data = meta.to_dict()
    assert data["page_number"] == 10
    assert data["chapter"] == "One"
    restored = StructureMetadata.from_dict(data)
    assert restored.section == "First"
