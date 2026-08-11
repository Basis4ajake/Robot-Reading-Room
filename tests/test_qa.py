from local_knowledge_library.models import Chunk, LibraryConfig, StructureMetadata
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
