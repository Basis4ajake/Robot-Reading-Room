from pathlib import Path

from local_knowledge_library.chunkers import ParagraphChunker
from local_knowledge_library.ingestion import IngestionPipeline
from local_knowledge_library.loaders import TextLoader
from local_knowledge_library.models import LibraryConfig
from local_knowledge_library.providers import DummyEmbedder, DummyLLMProvider, SqliteVectorStore
from local_knowledge_library.qa import GroundedQA
from local_knowledge_library.query_planner import QueryPlanner
from local_knowledge_library.retrieval import Retriever
from local_knowledge_library.storage import KnowledgeLibrary


def test_demo_pipeline_ingest_query(tmp_path):
    data_dir = tmp_path / "library"
    data_dir.mkdir()
    source_file = tmp_path / "example.txt"
    source_file.write_text("Hello world. This is a simple knowledge base entry.", encoding="utf-8")

    config = LibraryConfig(library_id="demo-lib", name="Demo Library", data_dir=str(data_dir))
    library = KnowledgeLibrary.create(config)
    library.add_source(str(source_file))

    vector_db = tmp_path / "vectors.db"
    vector_store = SqliteVectorStore(str(vector_db))

    pipeline = IngestionPipeline(
        loaders=[TextLoader()],
        chunker=ParagraphChunker(),
        embedder=DummyEmbedder(),
        vector_store=vector_store,
        debug=False,
    )
    pipeline.ingest(library)

    retriever = Retriever(vector_store=vector_store, embedder=DummyEmbedder(), debug=False)
    qa = GroundedQA(retriever=retriever, query_planner=QueryPlanner(), llm_provider=DummyLLMProvider(), debug=False)
    result = qa.answer_query("What does the document say?", library, top_k=1)

    assert result["query"] == "What does the document say?"
    assert "answer" in result
    assert result["citations"]
