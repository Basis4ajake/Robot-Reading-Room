import json
import tempfile
from pathlib import Path

from local_knowledge_library.abstracts import Chunker, DocumentLoader
from local_knowledge_library.chunkers import ParagraphChunker
from local_knowledge_library.ingestion import IngestionPipeline
from local_knowledge_library.models import (
    Chunk,
    DocumentMetadata,
    LibraryConfig,
    StructureMetadata,
)
from local_knowledge_library.providers.ollama_providers import DummyEmbedder, InMemoryVectorStore
from local_knowledge_library.storage import KnowledgeLibrary


class TextLoader(DocumentLoader):
    def supported_extensions(self):
        return ["txt"]

    def load(self, source_path: str):
        path = Path(source_path)
        with path.open("r", encoding="utf-8") as handle:
            text = handle.read()
        return [DocumentMetadata(
            source_id="",
            library_id="",
            document_id="",
            source_path=source_path,
            filename=path.name,
            file_type="txt",
            title=None,
            author=None,
            publisher=None,
            publication_date=None,
            edition=None,
            isbn=None,
            content_hash="",
            structure=StructureMetadata(),
            text=text,
        )]


def test_incremental_ingestion_detects_unchanged(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    source_file = tmp_path / "source.txt"
    source_file.write_text("Hello incremental ingestion", encoding="utf-8")
    config = LibraryConfig(library_id="test-lib", name="Test", data_dir=str(data_dir))
    library = KnowledgeLibrary.create(config)
    library.add_source(str(source_file))
    class TestChunker(Chunker):
        def chunk(self, document):
            return [
                Chunk(
                    chunk_id="chunk-1",
                    library_id=document.library_id,
                    source_id=document.source_id,
                    document_id=document.document_id,
                    text=document.text or "",
                    metadata={"source": document.filename},
                    citation_id="cite-1",
                )
            ]

    pipeline = IngestionPipeline(
        loaders=[TextLoader()],
        chunker=TestChunker(),
        embedder=DummyEmbedder(),
        vector_store=InMemoryVectorStore(),
    )
    result1 = pipeline.ingest(library)
    assert len(result1["processed"]) == 1
    result2 = pipeline.ingest(library)
    assert len(result2["skipped"]) == 1


class _NamedEmbedder:
    """Stands in for OllamaQwenProvider's embed_text, distinguished by embedding_model."""

    def __init__(self, embedding_model: str):
        self.embedding_model = embedding_model

    def embed_text(self, texts):
        return [[1.0] for _ in texts]


def test_ingestion_forces_reembed_when_embedding_model_changes(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    source_file = tmp_path / "source.txt"
    source_file.write_text("Hello incremental ingestion", encoding="utf-8")
    config = LibraryConfig(library_id="test-lib", name="Test", data_dir=str(data_dir))
    library = KnowledgeLibrary.create(config)
    library.add_source(str(source_file))

    class TestChunker(Chunker):
        def chunk(self, document):
            return [
                Chunk(
                    chunk_id="chunk-1",
                    library_id=document.library_id,
                    source_id=document.source_id,
                    document_id=document.document_id,
                    text=document.text or "",
                    metadata={"source": document.filename},
                    citation_id="cite-1",
                )
            ]

    vector_store = InMemoryVectorStore()

    pipeline_a = IngestionPipeline(
        loaders=[TextLoader()],
        chunker=TestChunker(),
        embedder=_NamedEmbedder("model-a"),
        vector_store=vector_store,
    )
    result1 = pipeline_a.ingest(library)
    assert len(result1["processed"]) == 1

    # Unchanged content, same embedder -> normally skipped.
    result2 = pipeline_a.ingest(library)
    assert len(result2["skipped"]) == 1

    # Switching embedding models with unchanged content must re-embed rather
    # than silently leaving stale, wrong-dimension vectors in the index.
    pipeline_b = IngestionPipeline(
        loaders=[TextLoader()],
        chunker=TestChunker(),
        embedder=_NamedEmbedder("model-b"),
        vector_store=vector_store,
    )
    result3 = pipeline_b.ingest(library)
    assert len(result3["processed"]) == 1
    assert len(result3["skipped"]) == 0

    # Back to normal incremental behavior once the signature is stable again.
    result4 = pipeline_b.ingest(library)
    assert len(result4["skipped"]) == 1


def test_ingestion_reembeds_legacy_index_with_no_recorded_signature(tmp_path):
    """A library indexed before signature-tracking existed (e.g. by a silent
    DummyEmbedder fallback) has documents recorded but no embedding_signature.
    The next ingest must not trust that unverified index and skip everything."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    source_file = tmp_path / "source.txt"
    source_file.write_text("Hello legacy ingestion", encoding="utf-8")
    config = LibraryConfig(library_id="test-lib", name="Test", data_dir=str(data_dir))
    library = KnowledgeLibrary.create(config)
    library.add_source(str(source_file))

    class TestChunker(Chunker):
        def chunk(self, document):
            return [
                Chunk(
                    chunk_id="chunk-1",
                    library_id=document.library_id,
                    source_id=document.source_id,
                    document_id=document.document_id,
                    text=document.text or "",
                    metadata={"source": document.filename},
                    citation_id="cite-1",
                )
            ]

    vector_store = InMemoryVectorStore()
    pipeline = IngestionPipeline(
        loaders=[TextLoader()],
        chunker=TestChunker(),
        embedder=_NamedEmbedder("real-model"),
        vector_store=vector_store,
    )
    result1 = pipeline.ingest(library)
    assert len(result1["processed"]) == 1

    # Simulate a pre-fix library.state.json: content was already indexed but
    # embedding_signature was never recorded.
    library.state.embedding_signature = None
    library.persist()
    library.load_state()

    result2 = pipeline.ingest(library)
    assert len(result2["processed"]) == 1
    assert len(result2["skipped"]) == 0

    # Once verified, normal incremental behavior resumes.
    result3 = pipeline.ingest(library)
    assert len(result3["skipped"]) == 1


def test_ingestion_reprocesses_and_cleans_up_when_chunk_size_changes(tmp_path):
    """A chunk_size/overlap change doesn't touch source content, so
    hash-based incremental ingestion would otherwise skip it forever. It also
    changes how many chunks a document produces, so this must actually clean
    up stale chunk rows rather than leave orphans behind (a bigger chunk_size
    produces FEWER chunks than before)."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    source_file = tmp_path / "source.txt"
    # 300 words, no blank lines -> one big paragraph/page.
    source_file.write_text(" ".join(f"word{i}" for i in range(300)), encoding="utf-8")
    config = LibraryConfig(library_id="test-lib", name="Test", data_dir=str(data_dir))
    library = KnowledgeLibrary.create(config)
    library.add_source(str(source_file))

    vector_store = InMemoryVectorStore()

    small_chunks_pipeline = IngestionPipeline(
        loaders=[TextLoader()],
        chunker=ParagraphChunker(chunk_size=50, chunk_overlap=0),
        embedder=DummyEmbedder(),
        vector_store=vector_store,
    )
    result1 = small_chunks_pipeline.ingest(library)
    assert len(result1["processed"]) == 1
    chunk_count_small = len(vector_store.chunks)
    assert chunk_count_small == 6  # 300 words / 50 per chunk, no overlap

    # Same content, bigger chunk_size -> fewer, larger chunks. Must reprocess
    # (not skip) and must not leave the old 6 chunks behind as orphans.
    big_chunks_pipeline = IngestionPipeline(
        loaders=[TextLoader()],
        chunker=ParagraphChunker(chunk_size=300, chunk_overlap=0),
        embedder=DummyEmbedder(),
        vector_store=vector_store,
    )
    result2 = big_chunks_pipeline.ingest(library)
    assert len(result2["processed"]) == 1
    assert len(result2["skipped"]) == 0
    assert len(vector_store.chunks) == 1  # whole paragraph fits in one chunk now

    # Back to normal incremental behavior once the new signature is recorded.
    result3 = big_chunks_pipeline.ingest(library)
    assert len(result3["skipped"]) == 1
    assert len(vector_store.chunks) == 1


def test_ingestion_refuses_to_overwrite_real_vectors_with_a_transient_dummy_fallback(tmp_path):
    """A library indexed with a real embedder, then hit by a transient
    Ollama outage (build_providers() falls back to DummyEmbedder for this
    runtime only), must not have its good vectors silently destroyed by the
    force-reprocess mechanism treating the outage as a deliberate model
    switch. Ingestion should refuse outright instead."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    source_file = tmp_path / "source.txt"
    source_file.write_text("Hello outage test", encoding="utf-8")
    config = LibraryConfig(library_id="test-lib", name="Test", data_dir=str(data_dir))
    library = KnowledgeLibrary.create(config)
    library.add_source(str(source_file))

    class TestChunker(Chunker):
        def chunk(self, document):
            return [
                Chunk(
                    chunk_id="chunk-1",
                    library_id=document.library_id,
                    source_id=document.source_id,
                    document_id=document.document_id,
                    text=document.text or "",
                    metadata={"source": document.filename},
                    citation_id="cite-1",
                )
            ]

    vector_store = InMemoryVectorStore()

    real_pipeline = IngestionPipeline(
        loaders=[TextLoader()],
        chunker=TestChunker(),
        embedder=_NamedEmbedder("nomic-embed-text"),
        vector_store=vector_store,
    )
    real_pipeline.ingest(library)
    assert library.state.embedding_signature == "nomic-embed-text"
    vectors_before = dict(vector_store.embeddings)

    outage_pipeline = IngestionPipeline(
        loaders=[TextLoader()],
        chunker=TestChunker(),
        embedder=DummyEmbedder(),
        vector_store=vector_store,
    )
    try:
        outage_pipeline.ingest(library)
        assert False, "expected ingest() to refuse rather than re-embed with DummyEmbedder"
    except RuntimeError as exc:
        assert "DummyEmbedder" in str(exc)

    # Nothing was touched: the real vectors and recorded signature survive.
    assert library.state.embedding_signature == "nomic-embed-text"
    assert vector_store.embeddings == vectors_before


class _ScriptedRecipeLLM:
    """Returns valid recipe-extraction JSON for every segment it's asked about."""

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        return json.dumps({"recipe_name": "Gnocchi", "ingredients": ["potatoes", "cheese"], "step_count": 2})

    def embed_text(self, texts):
        raise NotImplementedError


_RECIPE_BOOK_TEXT = (
    "GNOCCHI\n\n"
    "Prepare a certain quantity of boiled potatoes and mix them with grated "
    "cheese, salt and nutmeg then roll into little sticks and boil until "
    "they float to the top of a pot of boiling salted water.\n"
)


def _make_recipe_library(tmp_path, enable_recipe_extraction: bool):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    source_file = tmp_path / "cookbook.txt"
    source_file.write_text(_RECIPE_BOOK_TEXT, encoding="utf-8")
    config = LibraryConfig(
        library_id="test-lib",
        name="Test",
        data_dir=str(data_dir),
        enable_recipe_extraction=enable_recipe_extraction,
    )
    library = KnowledgeLibrary.create(config)
    library.add_source(str(source_file))
    return library


class _WholeDocChunker(Chunker):
    def chunk(self, document):
        return [
            Chunk(
                chunk_id="chunk-1",
                library_id=document.library_id,
                source_id=document.source_id,
                document_id=document.document_id,
                text=document.text or "",
                metadata={"source": document.filename},
                citation_id="cite-1",
            )
        ]


def test_ingestion_persists_recipe_facts_when_extraction_enabled(tmp_path):
    library = _make_recipe_library(tmp_path, enable_recipe_extraction=True)
    pipeline = IngestionPipeline(
        loaders=[TextLoader()],
        chunker=_WholeDocChunker(),
        embedder=DummyEmbedder(),
        vector_store=InMemoryVectorStore(),
        llm_provider=_ScriptedRecipeLLM(),
    )

    pipeline.ingest(library)

    facts = library.list_recipe_facts()
    assert len(facts) == 1
    assert facts[0].recipe_name == "Gnocchi"
    assert facts[0].ingredients == ["potatoes", "cheese"]
    assert facts[0].ingredient_count == 2
    assert facts[0].step_count == 2
    assert "boiled potatoes" in facts[0].source_excerpt

    # Persists to disk and survives a reopen, like every other artifact.
    reopened = KnowledgeLibrary.open(library.config)
    reopened_facts = reopened.list_recipe_facts()
    assert len(reopened_facts) == 1
    assert reopened_facts[0].recipe_name == "Gnocchi"


def test_ingestion_skips_recipe_extraction_when_disabled(tmp_path):
    library = _make_recipe_library(tmp_path, enable_recipe_extraction=False)
    pipeline = IngestionPipeline(
        loaders=[TextLoader()],
        chunker=_WholeDocChunker(),
        embedder=DummyEmbedder(),
        vector_store=InMemoryVectorStore(),
        llm_provider=_ScriptedRecipeLLM(),
    )

    pipeline.ingest(library)

    assert library.list_recipe_facts() == []


def test_ingestion_extracts_recipes_when_flag_is_toggled_on_after_first_ingest(tmp_path):
    """Turning enable_recipe_extraction on for an already-ingested library
    (unchanged content) must not be silently skipped by the unchanged-content
    check - the same failure class embedding_signature/chunking_signature
    already guard against."""
    library = _make_recipe_library(tmp_path, enable_recipe_extraction=False)
    vector_store = InMemoryVectorStore()
    pipeline = IngestionPipeline(
        loaders=[TextLoader()],
        chunker=_WholeDocChunker(),
        embedder=DummyEmbedder(),
        vector_store=vector_store,
        llm_provider=_ScriptedRecipeLLM(),
    )
    pipeline.ingest(library)
    assert library.list_recipe_facts() == []

    library.config.enable_recipe_extraction = True
    library.persist()
    library.load_config()

    pipeline.ingest(library)
    facts = library.list_recipe_facts()
    assert len(facts) == 1
    assert facts[0].recipe_name == "Gnocchi"


def test_ingestion_clears_recipe_facts_when_flag_is_toggled_off(tmp_path):
    library = _make_recipe_library(tmp_path, enable_recipe_extraction=True)
    vector_store = InMemoryVectorStore()
    pipeline = IngestionPipeline(
        loaders=[TextLoader()],
        chunker=_WholeDocChunker(),
        embedder=DummyEmbedder(),
        vector_store=vector_store,
        llm_provider=_ScriptedRecipeLLM(),
    )
    pipeline.ingest(library)
    assert len(library.list_recipe_facts()) == 1

    library.config.enable_recipe_extraction = False
    library.persist()

    pipeline.ingest(library)
    assert library.list_recipe_facts() == []


def test_ingestion_does_not_force_reprocess_for_ordinary_libraries_that_never_enable_recipe_extraction(tmp_path):
    """enable_recipe_extraction defaults to False for every library - a
    missing/None recorded signature must NOT be treated as "unverified" the
    way embedding/chunking signatures are, or every pre-existing library
    would get an expensive forced full reprocess the first time this ships."""
    library = _make_recipe_library(tmp_path, enable_recipe_extraction=False)
    pipeline = IngestionPipeline(
        loaders=[TextLoader()],
        chunker=_WholeDocChunker(),
        embedder=DummyEmbedder(),
        vector_store=InMemoryVectorStore(),
    )
    pipeline.ingest(library)

    result = pipeline.ingest(library)
    assert len(result["skipped"]) == 1
    assert len(result["processed"]) == 0
