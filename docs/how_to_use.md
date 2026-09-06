# How to Use Local Knowledge Library

This document is intended to evolve alongside the project. It is written as a working guide: usable today, but clearly aware that the platform is still being built.

## Status

This repository is an early MVP for a local, modular Retrieval-Augmented Generation platform. The current features include:

- Knowledge Library creation and management
- Library isolation and local storage
- Text, Markdown, and PDF ingestion (PDF via PyPDF2, with per-page citation tracking)
- Incremental indexing by content hash, with automatic detection/recovery if a library's embedding model or chunking configuration changes since it was last indexed
- A local provider abstraction layer for loaders, chunkers, embedders, vector stores, and LLMs, backed by real local Ollama models (chat + embedding) with a dummy fallback for trying the app without Ollama running
- Grounded answer generation with citation tracking, including accurate per-chunk page numbers for PDFs
- Persistent SQLite vector search (the in-memory store exists too, but only for tests)
- A FastAPI service layer exposing library management, ingestion, model listing, and chat over HTTP (see section 9)
- A Tauri desktop GUI built on that API — library management, sources/ingestion, and chat from a native window (see section 10)
- Persistent per-library configuration (model choice, chunk settings)
- Automated tests (44 and growing)

Planned future improvements are noted at the end of this document.

## 1. Setup

1. Copy the safe example configuration:

```bash
cp .env.example .env
```

2. Install dependencies:

```bash
python -m pip install -e ".[dev]"
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

## 9. Running the API Server

A FastAPI service wraps the library, registry, and QA pipeline so a GUI (or `curl`) can drive the system without writing Python.

Start it with:

```bash
python -m local_knowledge_library.api
```

By default it binds to `127.0.0.1:8000` (local-only). Configure it with environment variables:

- `LKL_DATA_DIR` — root directory for libraries (default `./data/libraries`)
- `LKL_API_HOST` / `LKL_API_PORT` — bind address (default `127.0.0.1:8000`)
- `LKL_FORCE_DUMMY` — set `true` to force the dummy embedder/LLM (useful without Ollama running)

Key endpoints, all under `/api/v1`:

- `GET /health` — server + Ollama reachability status
- `GET /models` — locally pulled Ollama model tags (quantization is part of the tag, e.g. `qwen2:7b-q4_0`, so there's no separate quantize control)
- `GET /libraries`, `POST /libraries`, `GET /libraries/{id}`, `PATCH /libraries/{id}`, `DELETE /libraries/{id}?confirm=true` — library lifecycle and per-library config (model, chunk size, top_k)
- `GET /libraries/{id}/sources`, `POST /libraries/{id}/sources`, `DELETE /libraries/{id}/sources/{source_id}` — manage sources by local file path
- `POST /libraries/{id}/ingest` — run incremental ingestion
- `POST /libraries/{id}/chat` — ask a grounded question, returns the answer plus citations and retrieved chunks

Example end-to-end session:

```bash
curl -X POST localhost:8000/api/v1/libraries \
  -H "Content-Type: application/json" \
  -d '{"library_id": "my-library", "name": "My Library"}'

curl -X POST localhost:8000/api/v1/libraries/my-library/sources \
  -H "Content-Type: application/json" \
  -d '{"source_path": "/absolute/path/to/file.md"}'

curl -X POST localhost:8000/api/v1/libraries/my-library/ingest

curl -X POST localhost:8000/api/v1/libraries/my-library/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Summarize the document."}'
```

Per-library configuration (`llm_model`, `embedding_model`, `chunk_size`, `chunk_overlap`, `top_k`) now persists to `config.json` inside the library's data directory, so a library reopens with the same settings it was created or last updated with.

Not yet implemented: streaming chat responses, authentication (fine for a single local user, not for anything exposed beyond localhost), and `ollama pull` model downloading through the API.

## 10. Using the GUI

A Tauri desktop app (`gui/`) is the primary way to use this project day-to-day instead of `curl`/Python — it's a native window wrapping the same API described in section 9, so the backend must be running first.

1. Start the backend (section 9): `python -m local_knowledge_library.api`
2. From `gui/`, run `npm run tauri dev` (first run installs Rust/Tauri build dependencies if they aren't already present; see `gui/README.md`).

From the window you can:

- See live backend/Ollama status and locally pulled models (top of the window).
- Create, edit, and delete libraries, including model choice (LLM + embedding model, with curated suggestions and a "Browse models ↗" link to Ollama's catalog) and chunk_size/chunk_overlap/top_k.
- Add sources via a native file picker, and run ingestion with processed/skipped/removed counts.
- Chat with a library and see the answer alongside its citations and (expandable) the raw retrieved chunks — a "demo mode" badge appears if Ollama/a real model isn't actually available, so a dummy answer is never mistaken for a real one.
- Toggle an optional "Bendy Toon" claymation theme (persisted per-browser via `localStorage`); the default theme follows your OS light/dark setting.

Not yet in the GUI: streaming responses (the backend doesn't support it yet either — see section 9), and multi-window/tabbed navigation (everything for an open library — its settings, sources, and chat — appears inline on one screen).

## 11. Debug Mode

A debug mode is available to inspect pipeline flow and detect which stage may be responsible for a weak answer.

Enable debug via configuration or constructor flags on the classes. In debug mode, the current implementation logs:

- query plan selection
- retrieval results
- reranking (when enabled)
- prompt assembly
- citation resolution

## 12. Running Tests

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

## 13. Work-in-Progress Notes

This guide is intentionally written as an incrementally updated document.

Current limitations:

- Text, Markdown, and PDF loaders are implemented and in real use (not scaffolding); EPUB/DOCX/HTML/OCR are not.
- Chunking is paragraph/page-based with word-count sub-splitting (`chunk_size`/`chunk_overlap`) — not yet chapter/section-aware.
- Ollama integration is real (not a stub) and depends on the local Ollama SDK/daemon being available; a dummy fallback exists for trying the app without Ollama, and now warns loudly (rather than substituting silently) when it's used unintentionally.
- Query planning is rule-based and designed to grow over time.
- `SimpleKeywordSearcher`/hybrid search and a real reranker (`DummyReranker` is currently a no-op passthrough) exist as interfaces but aren't wired into the default pipeline.
- `AppState`'s per-library runtime cache has no concurrency locking — concurrent requests to the same library (e.g. an ingest and a config update at the same time) can race.

Future improvements planned for the next iterations:

- EPUB/DOCX/HTML and OCR-based loaders
- Structure-aware chunking by chapter/section (page-level is done; chapter/section is not)
- Keyword, hybrid, and reciprocal rank fusion search
- Configurable rerankers
- Export/import of libraries
- Research workspace workflows
- Evaluation and RAG benchmarking
- Streaming chat responses; a `ChatResponse` field distinguishing a real answer from the dummy/degraded fallback (currently only inferable via `/health` + `/models`)

## 14. Contribution and Extension

If you extend this project, follow these guidelines:

- Add new providers behind the interface definitions in `src/local_knowledge_library/abstracts.py`
- Keep library data out of git and use `data/` for private content only
- Preserve provenance metadata through every pipeline stage
- Add tests for every new loader, search strategy, or provider implementation
- Update this document when the usage story changes
