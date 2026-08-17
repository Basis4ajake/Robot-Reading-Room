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

- Added a FastAPI service layer (`python -m local_knowledge_library.api`) exposing library management, ingestion, model listing, and chat over HTTP, plus persistent per-library configuration — the foundation the planned GUI control center will sit on.
- Fixed incremental ingestion and source change detection so unchanged documents are skipped and deleted content is cleaned up.
- Corrected `KnowledgeLibrary.open()` state loading and removed duplicate `remove_source()` behavior.
- Added `src/local_knowledge_library/providers/__init__.py` for package imports.
- Updated packaging and `pytest` configuration for `src` package discovery.
- Cleaned Python cache artifacts and verified the repo with `7 passed` test results.

## Repository Layout

- `src/` — application code
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

3. Create or open a library in code or with future CLI support.

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
