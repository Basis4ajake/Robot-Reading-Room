import json
from pathlib import Path

from local_knowledge_library.models import Chunk
from local_knowledge_library.providers import DummyEmbedder, SqliteVectorStore


def test_sqlite_vector_store_persists_across_instances(tmp_path):
    db_path = tmp_path / "vectors.db"
    store = SqliteVectorStore(str(db_path))
    chunk = Chunk(
        chunk_id="chunk-1",
        library_id="lib-1",
        source_id="source-1",
        document_id="doc-1",
        text="This is a test chunk.",
        metadata={"page_number": "1"},
        citation_id="cite-1",
    )
    embeddings = DummyEmbedder().embed_text([chunk.text])

    store.add([chunk], embeddings)
    result_before = store.search(embeddings[0], top_k=1)
    assert len(result_before) == 1
    assert result_before[0].chunk_id == chunk.chunk_id

    reopened = SqliteVectorStore(str(db_path))
    result_after = reopened.search(embeddings[0], top_k=1)
    assert len(result_after) == 1
    assert result_after[0].chunk_id == chunk.chunk_id
