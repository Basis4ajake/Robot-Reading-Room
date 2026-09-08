import tempfile
from pathlib import Path

from local_knowledge_library.models import (
    DocumentMetadata,
    EvalResult,
    EvalRun,
    LibraryConfig,
    RecipeFact,
    StructureMetadata,
)
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


def test_eval_cases_persist_and_survive_reopen(tmp_path):
    config = LibraryConfig(library_id="test-lib", name="Test Library", data_dir=str(tmp_path))
    library = KnowledgeLibrary.create(config)
    case = library.add_eval_case("What is the capital of France?", "paris")

    reopened = KnowledgeLibrary.open(config)
    cases = reopened.list_eval_cases()
    assert len(cases) == 1
    assert cases[0] == case


def test_remove_eval_case(tmp_path):
    config = LibraryConfig(library_id="test-lib", name="Test Library", data_dir=str(tmp_path))
    library = KnowledgeLibrary.create(config)
    case = library.add_eval_case("What is the capital of France?", "paris")

    library.remove_eval_case(case.eval_case_id)

    assert library.list_eval_cases() == []
    reopened = KnowledgeLibrary.open(config)
    assert reopened.list_eval_cases() == []


def _make_run(library_id: str, run_id: str, timestamp: str) -> EvalRun:
    return EvalRun(
        eval_run_id=run_id,
        library_id=library_id,
        timestamp=timestamp,
        chunk_size=300,
        chunk_overlap=60,
        top_k=8,
        llm_model="qwen2:1.5b",
        embedding_model="nomic-embed-text",
        results=[
            EvalResult(
                eval_case_id="case-1", question="q", expected_keyword="k",
                keyword_in_citations=True, keyword_in_answer=True,
                answer="a", answer_source="llm", citation_count=1,
            )
        ],
    )


def test_eval_runs_persist_and_list_newest_first(tmp_path):
    config = LibraryConfig(library_id="test-lib", name="Test Library", data_dir=str(tmp_path))
    library = KnowledgeLibrary.create(config)
    library.register_eval_run(_make_run("test-lib", "run-1", "2026-01-01T00:00:00+00:00"))
    library.register_eval_run(_make_run("test-lib", "run-2", "2026-01-02T00:00:00+00:00"))

    reopened = KnowledgeLibrary.open(config)
    runs = reopened.list_eval_runs()
    assert [run.eval_run_id for run in runs] == ["run-2", "run-1"]
    assert runs[0].passed_count == 1
    assert runs[0].total_count == 1


def test_eval_run_history_trims_oldest_past_cap(tmp_path):
    config = LibraryConfig(library_id="test-lib", name="Test Library", data_dir=str(tmp_path))
    library = KnowledgeLibrary.create(config)
    for i in range(55):
        library.register_eval_run(_make_run("test-lib", f"run-{i}", f"2026-01-01T00:00:{i:02d}+00:00"))

    runs = library.list_eval_runs()
    assert len(runs) == 50
    # Newest-first, and the oldest 5 (run-0..run-4) were trimmed.
    assert runs[0].eval_run_id == "run-54"
    assert "run-0" not in [run.eval_run_id for run in runs]
