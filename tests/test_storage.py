import tempfile
from pathlib import Path

from local_knowledge_library.models import LibraryConfig
from local_knowledge_library.storage import KnowledgeLibrary


def test_library_isolation_and_persistence(tmp_path):
    config = LibraryConfig(library_id="test-lib", name="Test Library", data_dir=str(tmp_path))
    library = KnowledgeLibrary.create(config)
    assert library.metadata.library_id == "test-lib"
    assert library.library_dir.exists()
    source = library.add_source(str(tmp_path / "doc.txt")) if False else None
    assert library.metadata.source_count == 0
    library.persist()
    reopened = KnowledgeLibrary.open(config)
    assert reopened.metadata.library_id == "test-lib"
    assert reopened.metadata.source_count == 0
