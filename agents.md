# Robot Reading Room - Development Instructions

## Project

This repository is **Robot Reading Room**, a local-first modular
Retrieval-Augmented Generation (RAG) library.

The primary goal is to build a reusable **Local Knowledge Library**
that allows users to create isolated knowledge libraries from books,
PDFs, documents, research papers, notes, and other files.

Users should be able to query, analyze, compare, summarize, and
synthesize information from their local knowledge libraries using
generative AI.

The project specification and architecture documentation are located
under `docs/`.


## Current Architecture

The project is Python-based and uses a virtual environment.

The main package is:

    src/local_knowledge_library/

Current major components include:

- `models.py` - domain/data models
- `abstracts.py` - interfaces/abstractions
- `loaders.py` - document loading
- `chunkers.py` - document chunking
- `ingestion.py` - ingestion pipeline
- `storage.py` - persistence and storage
- `retrieval.py` - retrieval
- `query_planner.py` - query planning
- `qa.py` - question answering/generation
- `providers/` - external/local provider implementations
- `ollama_providers.py` - Ollama-specific implementations

Tests are located in:

    tests/


## Core Architectural Principles

Preserve clear separation between:

- document ingestion
- document extraction
- normalization
- chunking
- embedding
- storage
- retrieval
- reranking
- query planning
- context assembly
- LLM generation
- citation/provenance

Prefer interfaces and abstractions for components that may have
multiple implementations.

Do not tightly couple the core RAG system to Ollama or Qwen.

Ollama and Qwen are currently the primary local AI implementation,
but future providers should be possible without rewriting the core
application.


## Knowledge Libraries

Knowledge Libraries are the central domain concept.

Each library must remain isolated from other libraries.

Library data may include:

- source documents
- extracted text
- metadata
- chunks
- embeddings
- indexes
- ingestion state
- configuration

Do not assume that data from one library is available to another.


## Provenance and Citations

Preserve document provenance throughout the RAG pipeline.

Where available, retain:

- source
- document
- filename
- author
- title
- publisher
- publication date
- page
- chapter
- section
- paragraph/structural location
- chunk ID
- content hash

The application, not the LLM, is responsible for authoritative
citation metadata.

Never invent page numbers, chapters, authors, titles, or other
bibliographic information.

Retrieved chunks should remain traceable to their original source.


## Local-First

The project is intended to run locally.

Do not introduce cloud AI services or hosted infrastructure unless
explicitly requested.

Ollama is the current local LLM/provider mechanism.

Keep infrastructure simple and appropriate for a single-machine
local application.


## Privacy and Git Security

This repository may be public.

**Knowledge Library data and secrets must NEVER be committed to Git.**

Treat `.gitignore` as a security boundary.

Never commit:

- books
- PDFs
- source documents
- extracted document text
- chunks
- embeddings
- vector indexes
- library databases
- private library metadata
- conversation history
- generated research data
- credentials
- API keys
- passwords
- access tokens
- `.env` files
- private configuration
- user-specific logs
- caches containing user data
- temporary ingestion files

The `data/` directory contains local application/library data and
must remain excluded from version control.

Do not weaken `.gitignore` rules to make development easier.

If a file appears to contain user data or secrets, do not commit it.


## Python Environment

Use the project's `.venv` for Python development.

Do not install project dependencies globally.

Use `pyproject.toml` as the authoritative project configuration.

Run tests using the project's virtual environment.


## Testing

Tests are required for meaningful changes.

Before changing code:

1. Inspect the relevant implementation.
2. Inspect the existing tests.
3. Understand the current behavior.
4. Make the smallest appropriate change.
5. Add or update tests.
6. Run the relevant tests.

Important areas include:

- ingestion
- loaders
- chunking
- models
- storage
- retrieval
- query planning
- providers
- QA
- provenance
- citations
- library isolation
- incremental indexing


## Development Behavior

Act as a senior software engineer collaborating with the developer.

Before making substantial architectural changes:

1. Inspect the existing code.
2. Inspect related tests.
3. Identify existing abstractions.
4. Explain the proposed change.
5. Prefer extending existing architecture over creating parallel systems.

Do not rewrite working code unnecessarily.

Do not introduce dependencies or infrastructure without a clear reason.

Do not implement future features simply because they are possible.


## Git

Do not create commits unless explicitly requested.

Before recommending a commit:

- inspect `git status`
- inspect the diff
- check for secrets
- check for private library data
- verify tests where practical


## Important Rule

When uncertain, prioritize:

1. Privacy
2. Provenance
3. Modularity
4. Testability
5. Local operation
6. Simplicity

Ask before making a large architectural change.
