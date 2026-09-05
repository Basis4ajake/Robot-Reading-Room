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
