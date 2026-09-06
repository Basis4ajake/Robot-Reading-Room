# Changelog

All notable changes to this project are documented in this file.

## [Unreleased] - 2026-09-05 (Phase 6, prototype)

### Added
- `recipe_extraction.py`: standalone recipe segmentation (`segment_recipes`) and per-recipe structured-fact extraction (`extract_recipe_facts`/`extract_all`) via the existing `LLMProvider` interface. Prototype for answering aggregate/superlative questions (e.g. "which recipe uses the fewest ingredients") that plain top-k vector search can't — not yet wired into `IngestionPipeline` or query handling. Verified against a real public-domain cookbook and live Ollama (`qwen2:1.5b`): correct structured JSON in 7-16s per recipe. 7 new tests (`tests/test_recipe_extraction.py`).

## [Unreleased] - 2026-09-05 (Phase 5)

### Added
- Loading feedback ("Creating...", "Saving...", "Deleting...") on the library Create/Save/Delete buttons while their requests are in flight, matching the pattern chat/ingest already used. Verified end-to-end with a real browser against the real backend.

### Changed
- Refreshed `README.md`, `gui/README.md` (was still the generic Tauri template), `docs/how_to_use.md`, and `docs/architecture.md` to reflect that the GUI exists and is the primary way to use the app, and to correct several stale/wrong claims (PDF ingestion and Ollama integration were described as stubs; they're both real; the documented `data/libraries/{id}/` layout was missing `state.json` and listed a `source_files/` directory that doesn't exist).

## [Unreleased] - 2026-09-05

### Fixed
- A library's index could silently go stale: switching `embedding_model` or `chunk_size`/`chunk_overlap` didn't change source content, so hash-based incremental ingestion kept skipping it forever, leaving old (wrong-dimension or wrong-size) vectors in place. `IngestionPipeline` now tracks `embedding_signature`/`chunking_signature` and forces a full re-embed/re-chunk when either changes or was never recorded on an existing index.
- PDF citations could cite the wrong page: page detection ran once per document via a single regex search, so every chunk in a book got the same page number. `ParagraphChunker` now attributes each chunk to its actual source page.
- A library created with no `embedding_model` set silently fell back to fake (`DummyEmbedder`) vectors with no warning. Defaulted `embedding_model` to `nomic-embed-text` everywhere it can be set (`LibraryConfig`, `LibraryRegistry.create_library`, `LibraryCreateRequest`), and `build_providers()` now prints a warning the moment it falls back to `DummyEmbedder` instead of doing so silently.

### Added
- `chunk_size`/`chunk_overlap` now actually do something: `ParagraphChunker` sub-splits any paragraph (page, for PDFs) longer than `chunk_size` words into overlapping word-windows. Previously these config fields were read by the GUI but ignored by ingestion. Overlap only applies within a paragraph/page, never across one, to keep page citations unambiguous.
- "Browse models ↗" links next to the LLM/embedding model fields in the GUI, pointing to Ollama's model catalog.

### Changed
- `chunk_size`/`chunk_overlap`/`top_k` backend defaults (`LibraryConfig`, `LibraryRegistry.create_library`, `LibraryCreateRequest`) realigned from `200`/`50`/`5` to `300`/`60`/`8`, matching the GUI form's own default (commit `81d61c1`) so direct API use gets the same recommendation. Already-created libraries keep their own persisted values.
- `OllamaQwenProvider.generate()` now sizes Ollama's `num_ctx` from the actual prompt at request time instead of relying on Ollama's undocumented 4096-token default, which silently drops old context (`--context-shift`) rather than erroring on an over-length prompt. The cap is now looked up per-model via `ollama.show()` instead of hardcoded to one model's max.

### Fixed (post-review)
- A transient Ollama outage could destroy a real index: `build_providers()` falling back to `DummyEmbedder` made the new signature-tracking treat it as a deliberate model change, force-reprocessing the library and overwriting good vectors with fake ones on the next `/ingest`. Ingestion now refuses outright instead.
- `chunk_overlap` >= `chunk_size` exploded chunk count (step collapsed to 1); a negative `chunk_overlap` silently dropped text (step overshot `chunk_size`). Both unvalidated anywhere in the stack - now clamped defensively in `ParagraphChunker`.
- `embedding_model: null` passed explicitly (not omitted) still bypassed the new default - `LibraryConfig.__post_init__` now normalizes it regardless of construction path.
- Added a runtime warning if `PdfLoader`'s and `ParagraphChunker`'s independent page/paragraph-count splits ever desync (currently kept in sync only by both using identical logic, with no shared code or assertion).

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
