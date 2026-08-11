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

## Data Directory

The default local library directory is configured via `LKL_DATA_DIR`, typically `./data/libraries`.

Each library is isolated under:

```
data/libraries/{library_id}/
  meta.json
  sources.json
  documents.json
  chunks.json
  indexes/
  source_files/
```

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
