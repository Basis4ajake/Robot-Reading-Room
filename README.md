# Local Knowledge Library

A local-first modular Retrieval-Augmented Generation (RAG) platform built for isolated Knowledge Libraries.

## Vision

Local Knowledge Library is designed to let users create private, self-contained knowledge collections from books, PDFs, Markdown, and text files. Each library is independently configurable and indexed, with provenance preserved from library → source → document → section → chunk.

## Key Features

- Modular provider interfaces for loaders, chunkers, embedders, vector stores, search engines, rerankers, and LLMs.
- Local-first operation with no dependency on cloud AI services for normal usage.
- Incremental ingestion that detects new, modified, deleted, and unchanged documents.
- Traceable provenance and citation objects for grounded responses.
- Library isolation so each knowledge library has its own metadata, indexes, and storage.
- Secure local data directory excluded from version control.
- Automated tests covering ingestion, retrieval, provenance, and provider abstraction.

## Recent updates

- Added a Tauri desktop GUI (`gui/`) — library management, source/ingestion, and chat all driven from a native window instead of `curl`/Python. See `gui/README.md` to run it.
- Fixed several correctness bugs found via real usage: libraries could silently index fake (non-semantic) vectors if `embedding_model` was unset or the embedder failed; PDF chunk citations could all cite the wrong page; `chunk_size`/`chunk_overlap` were read by the GUI but had no effect on ingestion. All three are fixed, with automatic detection/recovery if a library's index goes stale (e.g. after a model or chunking config change).
- Added a FastAPI service layer (`python -m local_knowledge_library.api`) exposing library management, ingestion, model listing, and chat over HTTP, plus persistent per-library configuration.
- Fixed incremental ingestion and source change detection so unchanged documents are skipped and deleted content is cleaned up.
- Corrected `KnowledgeLibrary.open()` state loading and removed duplicate `remove_source()` behavior.

## Repository Layout

- `src/` — application code (backend: ingestion, chunking, embedding, retrieval, grounded QA, FastAPI service layer)
- `gui/` — the Tauri desktop GUI (plain HTML/JS/CSS frontend + Rust shell); see `gui/README.md`
- `tests/` — automated tests
- `docs/` — architecture and implementation notes
- `data/` — local library data (excluded from git)
- `.env.example` — safe example configuration
- `.gitignore` — excludes private library data, secrets, caches, and generated artifacts

## Data Security

This repository is public-source friendly. All user library contents, extracted text, embeddings, indexes, search caches, and secrets must remain outside version control.

**Never commit**:

- `data/`, `local_data/`, `libraries/`
- PDFs, books, notes, or source documents
- extracted text and chunk stores
- embeddings and vector indexes
- application databases and search indexes
- `.env` or local configuration with secrets
- logs, caches, or temporary ingestion files
- model artifacts or downloaded model caches

## Getting Started

1. Copy `.env.example` to `.env`.
2. Install dependencies:

```bash
python -m pip install -e ".[dev]"
```

3. Make sure [Ollama](https://ollama.com) is installed and running, with at least one chat model and one embedding-capable model pulled (e.g. `ollama pull qwen2:1.5b` and `ollama pull nomic-embed-text`). The app runs with fake/dummy responses if Ollama isn't available, which is fine for trying it out but not for real use.
4. Start the backend:

```bash
python -m local_knowledge_library.api
```

5. Either drive it via `curl`/the HTTP API (see `docs/how_to_use.md` §9), or run the desktop GUI — see `gui/README.md`.

## Architecture Overview

The system separates the following concerns behind well-defined interfaces:

- Document ingestion
- Document normalization
- Structure detection
- Chunking
- Embedding
- Metadata storage
- Vector storage
- Keyword search
- Retrieval
- Result fusion
- Reranking
- Query planning
- Context assembly
- LLM generation
- Citation management

## Supported MVP Formats

- PDF
- TXT
- Markdown

## Notes

The implementation uses provider abstractions so that Ollama/Qwen can be replaced with other local or cloud providers later. The current codebase provides local-first defaults and a safe example configuration.

## Live Usage Guide

See `docs/how_to_use.md` for an in-progress user guide that grows with the project.
