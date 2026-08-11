from local_knowledge_library.providers.ollama_providers import DummyEmbedder, InMemoryVectorStore
from local_knowledge_library.retrieval import Retriever
from local_knowledge_library.models import Chunk


def test_retriever_semantic_search():
    vector_store = InMemoryVectorStore()
    chunks = [
        Chunk(
            chunk_id="chunk-1",
            library_id="lib-1",
            source_id="source-1",
            document_id="doc-1",
            text="The quick brown fox jumps over the lazy dog.",
            metadata={},
            citation_id="cite-1",
        ),
        Chunk(
            chunk_id="chunk-2",
            library_id="lib-1",
            source_id="source-1",
            document_id="doc-1",
            text="A different sentence with dogs and foxes.",
            metadata={},
            citation_id="cite-2",
        ),
    ]
    embeddings = DummyEmbedder().embed([chunk.text for chunk in chunks])
    vector_store.add(chunks, embeddings)
    retriever = Retriever(vector_store=vector_store, embedder=DummyEmbedder(), debug=False)
    results = retriever.semantic_search("fox dog", top_k=2)
    assert len(results) == 2
    assert results[0].chunk_id in {"chunk-1", "chunk-2"}
