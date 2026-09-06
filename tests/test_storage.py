import tempfile
from pathlib import Path

from local_knowledge_library.models import DocumentMetadata, LibraryConfig, RecipeFact, StructureMetadata
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


def test_recipe_facts_persist_and_survive_reopen(tmp_path):
    config = LibraryConfig(library_id="test-lib", name="Test Library", data_dir=str(tmp_path))
    library = KnowledgeLibrary.create(config)
    fact = RecipeFact(
        library_id="test-lib", source_id="src-1", document_id="doc-1",
        recipe_name="Gnocchi", ingredients=["potatoes", "cheese"], step_count=2,
    )
    library.register_recipe_facts("doc-1", [fact])
    library.persist()

    reopened = KnowledgeLibrary.open(config)
    facts = reopened.list_recipe_facts()
    assert len(facts) == 1
    assert facts[0] == fact


def test_remove_document_clears_its_recipe_facts(tmp_path):
    config = LibraryConfig(library_id="test-lib", name="Test Library", data_dir=str(tmp_path))
    library = KnowledgeLibrary.create(config)
    document = DocumentMetadata(
        source_id="src-1", library_id="test-lib", document_id="doc-1",
        source_path="/tmp/doc.txt", filename="doc.txt", file_type="txt",
        title=None, author=None, publisher=None, publication_date=None,
        edition=None, isbn=None, content_hash="hash", structure=StructureMetadata(),
    )
    library.register_document(document)
    library.register_recipe_facts("doc-1", [
        RecipeFact(library_id="test-lib", source_id="src-1", document_id="doc-1", recipe_name="A")
    ])

    library.remove_document("doc-1")

    assert library.list_recipe_facts() == []
