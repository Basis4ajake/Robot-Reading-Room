# Architecture

## Core Modules

- `src/local_knowledge_library/abstracts.py`
  - Defines provider interfaces for loaders, chunkers, embedders, vector stores, search engines, query planners, rerankers, and LLM providers.

- `src/local_knowledge_library/models.py`
  - Contains provenance-aware dataclasses for libraries, sources, documents, chunks, citations, and ingestion state.

- `src/local_knowledge_library/storage.py`
  - Manages library isolation, local data directories, metadata persistence, and incremental ingestion state.

- `src/local_knowledge_library/ingestion.py`
  - Implements the ingestion pipeline with loader selection, normalization, structure detection, chunking, embedding, and indexing.

- `src/local_knowledge_library/retrieval.py`
  - Provides semantic retrieval, keyword search, and result fusion abstractions.

- `src/local_knowledge_library/qa.py`
  - Builds grounded prompt context, resolves citations, and generates answers with LLM providers.

- `src/local_knowledge_library/providers/ollama_providers.py`
  - Contains local Ollama provider adapters and a safe dummy fallback for testing.

- `src/local_knowledge_library/providers/factory.py`
  - Builds embedder/LLM providers from a `LibraryConfig`, falling back to dummy providers when Ollama is unavailable. Also lists locally pulled Ollama model tags.

- `src/local_knowledge_library/registry.py`
  - `LibraryRegistry` manages the set of libraries under a data directory: create, list, open, update config, delete.

- `src/local_knowledge_library/api/`
  - A FastAPI service exposing the library, ingestion, and QA layers over HTTP for the Tauri GUI (`gui/`) and any `curl`/scripting use. `app.py` builds the app and owns an `AppState` (the registry plus a small cache of per-library vector store/retriever/QA instances, with no concurrency locking — see docs/how_to_use.md §13). Routers live under `api/routers/`: `libraries`, `sources`, `chat`, `models`, `health`. Run with `python -m local_knowledge_library.api`.

- `gui/`
  - The Tauri desktop GUI: a plain HTML/JS/CSS frontend (`gui/src/`, no framework/build step) served directly by Tauri's webview, plus a thin Rust shell (`gui/src-tauri/`) for the native window, file-picker dialog, and opening external links in the system browser. Talks to the API above over `fetch()`. See `gui/README.md`.

## Data Directory

The default local library directory is configured via `LKL_DATA_DIR`, typically `./data/libraries`.

Each library is isolated under:

```
data/libraries/{library_id}/
  config.json
  meta.json
  sources.json
  documents.json
  chunks.json
  state.json       # per-source/document content hashes, plus embedding_signature
                    # and chunking_signature (which model/config last built the index)
  indexes/
    vectors.db      # SqliteVectorStore
```

Sources are referenced by their original absolute path on disk (`add_source`), not copied into the library directory — there is no `source_files/`.

`config.json` persists the library's `LibraryConfig` (model choice, chunk settings, top_k) so it survives across process restarts and API calls — it is the source of truth once a library has been created; `KnowledgeLibrary.open()` loads it and ignores an in-memory config passed by the caller.

## Provenance

Provenance flows through the system as:

Library → Source → Document → Section → Chunk

Each chunk carries metadata that enables citation resolution without relying on the LLM to invent bibliographic information.

## Incremental Indexing

Ingestion uses content hashing to detect unchanged documents. A file that is unchanged is skipped. Modified and new files are processed and indexed without rebuilding the entire library.

## Query Planning

A lightweight query planner classifies queries into modes such as lookup, summarization, comparison, and synthesis. The architecture allows additional planning strategies to be added later.

## Grounded Generation

The application separates:

1. User conversation context
2. Retrieved knowledge context
3. Generated response

The LLM is supplied only with assembled, cited context and citation identifiers. Citation metadata is resolved by the application.
