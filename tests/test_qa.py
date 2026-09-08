from local_knowledge_library.abstracts import LLMProvider
from local_knowledge_library.models import Chunk, LibraryConfig, RecipeFact, StructureMetadata
from local_knowledge_library.providers.ollama_providers import DummyEmbedder, DummyLLMProvider, InMemoryVectorStore
from local_knowledge_library.query_planner import QueryPlanner
from local_knowledge_library.qa import GroundedQA
from local_knowledge_library.retrieval import Retriever
from local_knowledge_library.storage import KnowledgeLibrary


class _FakeRealLLM(LLMProvider):
    """Stands in for a real (non-dummy) LLMProvider, e.g. OllamaQwenProvider."""

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        return "A real-looking answer."

    def embed_text(self, texts):
        raise NotImplementedError


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
    assert result["answer_source"] == "dummy"


def test_answer_query_reports_llm_source_for_a_real_provider(tmp_path):
    """The long-standing gap: ChatResponse had no field distinguishing a
    real LLM answer from the DummyLLMProvider fallback."""
    config = LibraryConfig(library_id="test-lib", name="Test", data_dir=str(tmp_path))
    library = KnowledgeLibrary.create(config)
    chunk = Chunk(
        chunk_id="chunk-1", library_id="test-lib", source_id="source-1", document_id="doc-1",
        text="Example evidence text.", metadata={"page_number": "1"}, citation_id="cite-1",
    )
    library.register_chunk(chunk)
    retriever = Retriever(vector_store=InMemoryVectorStore(), embedder=DummyEmbedder())
    qa = GroundedQA(retriever=retriever, query_planner=QueryPlanner(), llm_provider=_FakeRealLLM())

    result = qa.answer_query("What is this?", library=library, top_k=1)

    assert result["answer_source"] == "llm"
    assert result["answer"] == "A real-looking answer."


def _make_qa(library):
    retriever = Retriever(vector_store=InMemoryVectorStore(), embedder=DummyEmbedder(), debug=False)
    return GroundedQA(retriever=retriever, query_planner=QueryPlanner(), llm_provider=DummyLLMProvider())


def test_answer_query_computes_fewest_ingredients_deterministically(tmp_path):
    config = LibraryConfig(
        library_id="test-lib", name="Test", data_dir=str(tmp_path), enable_recipe_extraction=True
    )
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
    assert result["answer_source"] == "computed"


def test_answer_query_lists_all_tied_recipes(tmp_path):
    config = LibraryConfig(
        library_id="test-lib", name="Test", data_dir=str(tmp_path), enable_recipe_extraction=True
    )
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
    assert "under-counted" not in result["answer"]


def test_answer_query_adds_a_caveat_for_a_wide_tie(tmp_path):
    """A real full-book run produced 8-way and 16-way ties at the minimum
    ingredient/step count - implausible for real recipes, and a sign of the
    small extraction model under-counting rather than genuine equality. A
    wide tie should say so instead of presenting the list as precise."""
    config = LibraryConfig(
        library_id="test-lib", name="Test", data_dir=str(tmp_path), enable_recipe_extraction=True
    )
    library = KnowledgeLibrary.create(config)
    library.register_recipe_facts("doc-1", [
        RecipeFact(library_id="test-lib", source_id="src-1", document_id="doc-1",
                   recipe_name=name, ingredients=["x"], step_count=1)
        for name in ["A", "B", "C", "D"]
    ])

    qa = _make_qa(library)
    result = qa.answer_query("Which recipe has the fewest ingredients?", library=library)

    assert "4 recipes are tied" in result["answer"]
    assert "under-counted" in result["answer"]


def test_answer_query_gives_distinct_citation_ids_for_same_titled_recipes(tmp_path):
    """A real book has two different recipes both titled "BISCUIT" - their
    citation_id/chunk_id must not collide just because they share a
    document_id and recipe_name, or the GUI can't tell them apart."""
    config = LibraryConfig(
        library_id="test-lib", name="Test", data_dir=str(tmp_path), enable_recipe_extraction=True
    )
    library = KnowledgeLibrary.create(config)
    library.register_recipe_facts("doc-1", [
        RecipeFact(library_id="test-lib", source_id="src-1", document_id="doc-1",
                   recipe_name="BISCUIT", ingredients=["flour"], step_count=1,
                   source_excerpt="First biscuit recipe text..."),
        RecipeFact(library_id="test-lib", source_id="src-1", document_id="doc-1",
                   recipe_name="BISCUIT", ingredients=["sugar"], step_count=1,
                   source_excerpt="Second, different biscuit recipe text..."),
    ])

    qa = _make_qa(library)
    # Both facts have 1 ingredient, so both tie for "fewest" and both land in
    # the same answer - exactly the scenario where the old document_id +
    # recipe_name-only id would have collided.
    result = qa.answer_query("Which recipe has the fewest ingredients?", library=library)

    citation_ids = [c["citation_id"] for c in result["citations"]]
    assert len(citation_ids) == 2
    assert len(set(citation_ids)) == 2, "citation_ids collided for two distinct same-titled recipes"


def test_answer_query_refuses_cost_questions_honestly(tmp_path):
    config = LibraryConfig(
        library_id="test-lib", name="Test", data_dir=str(tmp_path), enable_recipe_extraction=True
    )
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
    assert result["answer_source"] == "computed"


def test_answer_query_handles_missing_recipe_data_gracefully(tmp_path):
    config = LibraryConfig(
        library_id="test-lib", name="Test", data_dir=str(tmp_path), enable_recipe_extraction=True
    )
    library = KnowledgeLibrary.create(config)

    qa = _make_qa(library)
    result = qa.answer_query("Which recipe uses the fewest ingredients?", library=library)

    assert "doesn't have recipe data" in result["answer"]
    assert result["citations"] == []
    assert result["answer_source"] == "computed"


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


def test_answer_query_ignores_superlatives_when_recipe_extraction_disabled(tmp_path):
    """interpret_aggregate_query() matches bare superlative words ("longest",
    "highest", "most", ...) with no recipe-specific requirement at all - a
    real, non-hypothetical library (e.g. a geography book, or this
    project's own scripts/eval_retrieval.py-style test corpus) asking
    "What is the longest river in the world?" must not be misrouted into a
    recipe-only refusal just because it never turned recipe extraction on."""
    config = LibraryConfig(library_id="test-lib", name="Test", data_dir=str(tmp_path))
    library = KnowledgeLibrary.create(config)
    chunk = Chunk(
        chunk_id="chunk-1", library_id="test-lib", source_id="source-1", document_id="doc-1",
        text="The Nile is the longest river in the world.", metadata={"page_number": "1"},
        citation_id="cite-1",
    )
    library.register_chunk(chunk)

    qa = _make_qa(library)
    result = qa.answer_query("What is the longest river in the world?", library=library, top_k=1)

    assert result["answer_source"] != "computed"
    assert "doesn't have recipe data" not in result["answer"]
