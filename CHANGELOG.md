# Changelog

All notable changes to this project are documented in this file.

## [Unreleased] - 2026-08-17

### Added
- FastAPI service layer (`src/local_knowledge_library/api/`) exposing the RAG core over HTTP: library CRUD, source management, ingestion, chat, Ollama model listing, and a health check. Run with `python -m local_knowledge_library.api`. This is the foundation the planned GUI control center will sit on.
- Persistent per-library configuration — `LibraryConfig` now includes `llm_model` and `embedding_model`, and `KnowledgeLibrary` writes/reads a `config.json` per library so model and chunk settings survive restarts instead of being re-supplied by the caller each run.
- `LibraryRegistry` (`registry.py`) for creating, listing, opening, updating, and deleting libraries under a data directory.
- Shared provider factory (`providers/factory.py`) with Ollama-with-dummy-fallback logic and `list_ollama_models()`, deduplicated out of the demo script.
- `tests/test_api.py` covering the new API end-to-end with dummy providers (no live Ollama required).

### Changed
- `scripts/demo_ingest_query.py` now uses the shared provider factory instead of its own copy of the fallback logic.
- README install instructions corrected to `pip install -e ".[dev]"` (previously referenced a nonexistent `requirements.txt`).
- `.gitignore` broadened to cover Python build/test artifacts (`.pytest_cache/`, `*.egg-info/`, coverage/lint caches) and future frontend/Tauri build output (`node_modules/`, `dist/`, `src-tauri/target/`, etc.).

## [0.1.0] - earlier history

- Initial project architecture, provider abstractions, and README.
- Incremental ingestion, storage cleanup, package imports fixed; `.gitignore` strengthened for secrets.
- SQLite vector store persistence added; Ollama provider wiring fixed.
- Demo ingest/query workflow (`scripts/demo_ingest_query.py`) and README added.
- Old README replaced; `agents.md` added for working with Qwen Code locally.
