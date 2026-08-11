# How to Use Local Knowledge Library

This document is intended to evolve alongside the project. It is written as a working guide: usable today, but clearly aware that the platform is still being built.

## Status

This repository is an early MVP for a local, modular Retrieval-Augmented Generation platform. The current features include:

- Knowledge Library creation and management
- Library isolation and local storage
- Text and markdown ingestion
- Incremental indexing by content hash
- A local provider abstraction layer for loaders, chunkers, embedders, vector stores, and LLMs
- Grounded answer generation with citation tracking
- In-memory vector search for local testing
- Basic automated tests

Planned future improvements are noted at the end of this document.

## 1. Setup

1. Copy the safe example configuration:

```bash
cp .env.example .env
```

2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

3. Confirm that the local data directory is excluded from version control.

The default local library directory is configured in `.env` as `LKL_DATA_DIR` and is excluded by `.gitignore`.

## 2. Project Layout

- `src/` — application source code
- `tests/` — automated tests
- `docs/` — architecture, design notes, and usage documentation
- `data/` — local library data and indexes (private, ignored)

## 3. Create and Open a Knowledge Library

The current codebase exposes a `KnowledgeLibrary` object for library lifecycle management.

Example:

```python
from local_knowledge_library.models import LibraryConfig
from local_knowledge_library.storage import KnowledgeLibrary

config = LibraryConfig(
    library_id="example-library",
    name="Example Library",
    description="A local knowledge collection for testing.",
    data_dir="./data/libraries",
    debug=True,
)

library = KnowledgeLibrary.create(config)
```

To open an existing library:

```python
library = KnowledgeLibrary.open(config)
```

## 4. Add Sources

A source is a file that is ingested into a library.

```python
library.add_source("./my-documents/example.md")
library.persist()
```

The source path is normalized and stored with a content hash. The system can later detect changed and removed files.

## 5. Ingestion Pipeline

The ingestion pipeline is intentionally modular:

- Loader chooses a parser based on file extension
- Document metadata is created and normalized
- Structure metadata is detected where available
- Chunking splits content into retrievable fragments
- Embeddings are generated through the configured provider
- Chunk vectors are indexed in a local vector store

### Example ingestion setup

```python
from local_knowledge_library.ingestion import IngestionPipeline
from local_knowledge_library.query_planner import QueryPlanner
from local_knowledge_library.retrieval import Retriever
from local_knowledge_library.providers.ollama_providers import DummyEmbedder, InMemoryVectorStore, DummyLLMProvider
from local_knowledge_library.qa import GroundedQA

pipeline = IngestionPipeline(
    loaders=[TextLoader(), MarkdownLoader(), PdfLoader()],
    chunker=ParagraphChunker(),
    embedder=DummyEmbedder(),
    vector_store=InMemoryVectorStore(),
    debug=True,
)
```

This repository currently includes placeholders and examples. The system is designed so new loaders and chunkers can be added without changing the pipeline implementation.

## 6. Incremental Indexing

The ingest pipeline identifies document states using content hashes:

- `processed` — newly ingested or modified documents
- `skipped` — unchanged documents
- `removed` — sources deleted from disk

This allows adding a new document without rebuilding an entire library.

## 7. Retrieval and Question Answering

The current MVP supports semantic search with a local vector store and a query planner that selects a conceptual strategy.

Example:

```python
retriever = Retriever(
    vector_store=vector_store,
    embedder=embedder,
)
qa = GroundedQA(
    retriever=retriever,
    query_planner=QueryPlanner(),
    llm_provider=DummyLLMProvider(),
)

response = qa.answer_query("What does the document say about local privacy?", library)
print(response["answer"])
print(response["citations"])
```

The generated answer is assembled from retrieved chunks and citation metadata. The model is not used as the knowledge store.

## 8. Citations and Provenance

Each chunk retains provenance metadata such as:

- library_id
- source_id
- document_id
- chunk_id
- page number
- section
- filename

The system resolves citation identifiers into structured citation objects before passing them to the LLM.

## 9. Debug Mode

A debug mode is available to inspect pipeline flow and detect which stage may be responsible for a weak answer.

Enable debug via configuration or constructor flags on the classes. In debug mode, the current implementation logs:

- query plan selection
- retrieval results
- reranking (when enabled)
- prompt assembly
- citation resolution

## 10. Running Tests

Run the available tests with:

```bash
pytest
```

Current test coverage includes:

- provider abstractions
- ingestion and incremental indexing
- retrieval behavior
- citation/provenance integrity
- library persistence and isolation

## 11. Work-in-Progress Notes

This guide is intentionally written as an incrementally updated document.

Current limitations:

- Only basic text and markdown loaders are scaffolding examples
- PDF ingestion is conceptual and may require additional parser wiring
- In-memory vector store is suitable for testing, not large collections
- Ollama/Qwen integration is provided as an adapter stub and depends on local Ollama SDK availability
- Query planning is rule-based and designed to grow over time

Future improvements planned for the next iterations:

- EPUB/DOCX/HTML and OCR-based loaders
- Structure-aware chunking by chapter/section
- Keyword, hybrid, and reciprocal rank fusion search
- Configurable rerankers
- Persistent local vector indexes
- Export/import of libraries
- Research workspace workflows
- Evaluation and RAG benchmarking

## 12. Contribution and Extension

If you extend this project, follow these guidelines:

- Add new providers behind the interface definitions in `src/local_knowledge_library/abstracts.py`
- Keep library data out of git and use `data/` for private content only
- Preserve provenance metadata through every pipeline stage
- Add tests for every new loader, search strategy, or provider implementation
- Update this document when the usage story changes
