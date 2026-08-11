import tempfile
from pathlib import Path

from local_knowledge_library.abstracts import Chunker, DocumentLoader
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
