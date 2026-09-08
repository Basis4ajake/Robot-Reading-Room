from local_knowledge_library.abstracts import LLMProvider
from local_knowledge_library.evaluation import run_evaluation
from local_knowledge_library.models import Chunk, LibraryConfig
from local_knowledge_library.providers.ollama_providers import DummyEmbedder, InMemoryVectorStore
from local_knowledge_library.qa import GroundedQA
from local_knowledge_library.query_planner import QueryPlanner
from local_knowledge_library.retrieval import Retriever
from local_knowledge_library.storage import KnowledgeLibrary


class _FakeLLM(LLMProvider):
    """Returns a fixed answer that paraphrases away the expected keyword -
    the real-world case this project's docstring warns about (retrieval is
    correct, but the LLM's prose doesn't repeat the exact term)."""

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        return "The river in question is the world's longest."

    def embed_text(self, texts):
        raise NotImplementedError


def _make_qa_and_library(tmp_path):
    config = LibraryConfig(library_id="test-lib", name="Test", data_dir=str(tmp_path))
    library = KnowledgeLibrary.create(config)
    chunk = Chunk(
        chunk_id="chunk-1", library_id="test-lib", source_id="source-1", document_id="doc-1",
        text="The Nile is the longest river in the world.", metadata={"page_number": "1"},
        citation_id="cite-1",
    )
    library.register_chunk(chunk)
    embedder = DummyEmbedder()
    vector_store = InMemoryVectorStore()
    vector_store.add([chunk], embedder.embed_text([chunk.text]))
    retriever = Retriever(vector_store=vector_store, embedder=embedder)
    qa = GroundedQA(retriever=retriever, query_planner=QueryPlanner(), llm_provider=_FakeLLM())
    return library, qa


def test_run_evaluation_checks_citations_and_answer_independently(tmp_path):
    library, qa = _make_qa_and_library(tmp_path)
    library.add_eval_case("What is the longest river?", "nile")

    run = run_evaluation(library, qa)

    assert len(run.results) == 1
    result = run.results[0]
    # Correct evidence was retrieved (the real regression signal)...
    assert result.keyword_in_citations is True
    # ...even though the LLM's prose never says "Nile" - the exact case
    # that makes answer-text-only checking unreliable.
    assert result.keyword_in_answer is False
    assert result.answer_source == "llm"


def test_run_evaluation_fails_citations_when_keyword_absent(tmp_path):
    library, qa = _make_qa_and_library(tmp_path)
    library.add_eval_case("What is the capital of France?", "paris")

    run = run_evaluation(library, qa)

    assert run.results[0].keyword_in_citations is False
    assert run.passed_count == 0
    assert run.total_count == 1


def test_run_evaluation_stamps_config_snapshot(tmp_path):
    config = LibraryConfig(
        library_id="test-lib", name="Test", data_dir=str(tmp_path),
        chunk_size=123, chunk_overlap=45, top_k=6, llm_model="qwen2:1.5b",
        embedding_model="nomic-embed-text",
    )
    library = KnowledgeLibrary.create(config)
    retriever = Retriever(vector_store=InMemoryVectorStore(), embedder=DummyEmbedder())
    qa = GroundedQA(retriever=retriever, query_planner=QueryPlanner(), llm_provider=_FakeLLM())

    run = run_evaluation(library, qa)

    assert run.chunk_size == 123
    assert run.chunk_overlap == 45
    assert run.top_k == 6
    assert run.llm_model == "qwen2:1.5b"
    assert run.embedding_model == "nomic-embed-text"
    assert run.results == []


def test_run_evaluation_with_no_cases_returns_empty_results(tmp_path):
    library, qa = _make_qa_and_library(tmp_path)

    run = run_evaluation(library, qa)

    assert run.results == []
    assert run.passed_count == 0
    assert run.total_count == 0
