from local_knowledge_library.models import Chunk, LibraryConfig, RecipeFact, StructureMetadata
from local_knowledge_library.providers.ollama_providers import DummyEmbedder, DummyLLMProvider, InMemoryVectorStore
from local_knowledge_library.query_planner import QueryPlanner
from local_knowledge_library.qa import GroundedQA
from local_knowledge_library.retrieval import Retriever
from local_knowledge_library.storage import KnowledgeLibrary


def test_grounded_qa_builds_prompt(tmp_path):
    config = LibraryConfig(library_id="test-lib", name="Test", data_dir=str(tmp_path))
    library = KnowledgeLibrary.create(config)
    chunk = Chunk(
        chunk_id="chunk-1",
        library_id="test-lib",
        source_id="source-1",
        document_id="doc-1",
        text="Example evidence text.",
        metadata={"page_number": "1"},
        citation_id="cite-1",
    )
    library.register_chunk(chunk)
    retriever = Retriever(vector_store=InMemoryVectorStore(), embedder=DummyEmbedder(), debug=False)
    qa = GroundedQA(retriever=retriever, query_planner=QueryPlanner(), llm_provider=DummyLLMProvider(), debug=True)
    result = qa.answer_query("What is this?", library=library, top_k=1)
    assert result["query"] == "What is this?"
    assert "citations" in result
    assert result["answer"].startswith("This is a dummy response")


def _make_qa(library):
    retriever = Retriever(vector_store=InMemoryVectorStore(), embedder=DummyEmbedder(), debug=False)
    return GroundedQA(retriever=retriever, query_planner=QueryPlanner(), llm_provider=DummyLLMProvider())


def test_answer_query_computes_fewest_ingredients_deterministically(tmp_path):
    config = LibraryConfig(library_id="test-lib", name="Test", data_dir=str(tmp_path))
    library = KnowledgeLibrary.create(config)
    source_file = tmp_path / "cookbook.txt"
    source_file.write_text("placeholder", encoding="utf-8")
    source = library.add_source(str(source_file))

    library.register_recipe_facts("doc-1", [
        RecipeFact(
            library_id="test-lib", source_id=source.source_id, document_id="doc-1",
            recipe_name="Gnocchi", ingredients=["potatoes", "cheese", "eggs"], step_count=3,
            source_excerpt="Prepare boiled potatoes...",
        ),
        RecipeFact(
            library_id="test-lib", source_id=source.source_id, document_id="doc-1",
            recipe_name="Bread Soup", ingredients=["bread", "cheese"], step_count=4,
            source_excerpt="Soak bread in broth...",
        ),
    ])

    qa = _make_qa(library)
    result = qa.answer_query("Which recipe uses the fewest ingredients?", library=library)

    assert "Bread Soup" in result["answer"]
    assert "2 ingredients" in result["answer"]
    assert result["plan"] == "aggregate_superlative"
    assert len(result["citations"]) == 1
    assert result["citations"][0]["section"] == "Bread Soup"
    assert len(result["chunks"]) == 1
    assert "Soak bread in broth" in result["chunks"][0]["text"]


def test_answer_query_lists_all_tied_recipes(tmp_path):
    config = LibraryConfig(library_id="test-lib", name="Test", data_dir=str(tmp_path))
    library = KnowledgeLibrary.create(config)
    library.register_recipe_facts("doc-1", [
        RecipeFact(library_id="test-lib", source_id="src-1", document_id="doc-1",
                   recipe_name="A", ingredients=["x"], step_count=1),
        RecipeFact(library_id="test-lib", source_id="src-1", document_id="doc-1",
                   recipe_name="B", ingredients=["y"], step_count=1),
    ])

    qa = _make_qa(library)
    result = qa.answer_query("Which recipe has the fewest ingredients?", library=library)

    assert "2 recipes are tied" in result["answer"]
    assert "A" in result["answer"] and "B" in result["answer"]
    assert len(result["citations"]) == 2


def test_answer_query_refuses_cost_questions_honestly(tmp_path):
    config = LibraryConfig(library_id="test-lib", name="Test", data_dir=str(tmp_path))
    library = KnowledgeLibrary.create(config)
    library.register_recipe_facts("doc-1", [
        RecipeFact(library_id="test-lib", source_id="src-1", document_id="doc-1",
                   recipe_name="A", ingredients=["x"], step_count=1),
    ])

    qa = _make_qa(library)
    result = qa.answer_query("Which recipe is cheapest to make?", library=library)

    assert "can't answer" in result["answer"].lower()
    assert result["citations"] == []
    assert result["chunks"] == []


def test_answer_query_handles_missing_recipe_data_gracefully(tmp_path):
    config = LibraryConfig(library_id="test-lib", name="Test", data_dir=str(tmp_path))
    library = KnowledgeLibrary.create(config)

    qa = _make_qa(library)
    result = qa.answer_query("Which recipe uses the fewest ingredients?", library=library)

    assert "doesn't have recipe data" in result["answer"]
    assert result["citations"] == []


def test_answer_query_ignores_plain_ingredient_mentions_without_superlative(tmp_path):
    """'What ingredients are in the risotto?' must stay on the normal
    vector-search path - it names the metric word but has no superlative, so
    routing it to the aggregate branch would silently ignore what was
    actually asked."""
    config = LibraryConfig(library_id="test-lib", name="Test", data_dir=str(tmp_path))
    library = KnowledgeLibrary.create(config)
    library.register_recipe_facts("doc-1", [
        RecipeFact(library_id="test-lib", source_id="src-1", document_id="doc-1",
                   recipe_name="A", ingredients=["x"], step_count=1),
    ])

    qa = _make_qa(library)
    result = qa.answer_query("What ingredients are in the risotto?", library=library, top_k=1)

    assert result["answer"].startswith("This is a dummy response")
