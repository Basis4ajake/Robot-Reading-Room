# Changelog

All notable changes to this project are documented in this file.

## [Unreleased] - 2026-09-08 (eval-history: user-defined regression questions, run for real, kept as history)

### Added
- Per-library evaluation: `EvalCase` (a user-authored question + expected keyword), `EvalResult` (one case's outcome), and `EvalRun` (a full pass, persisted to history) in `models.py`; `evaluation.py`'s `run_evaluation()` runs every stored case through the real `GroundedQA.answer_query` pipeline - the same retrieval, reranking, and answer generation chat uses, not a separate code path like `scripts/eval_retrieval.py`'s raw-retrieval check.
- Each `EvalResult` records two independent pass/fail signals rather than one: `keyword_in_citations` (checked against the actually-retrieved evidence - the real regression signal, since retrieval is what has silently broken repeatedly in this project) and `keyword_in_answer` (checked against the free-form LLM answer text - a real quality signal, but too noisy to be the headline, since a correct answer can rephrase away the exact keyword).
- Each `EvalRun` stamps the config it executed under (`chunk_size`/`chunk_overlap`/`top_k`/`llm_model`/`embedding_model`), since two runs' pass counts are only comparable if you know whether settings changed between them - the whole point of keeping history instead of a single last-run result.
- New API endpoints under `/api/v1/libraries/{id}`: `GET`/`POST /eval-cases`, `DELETE /eval-cases/{eval_case_id}`, `POST /evaluate`, `GET /eval-runs` (newest first). `evaluate` holds `AppState.use_runtime()` for the whole run (can be minutes, not the single-question duration chat holds it for) so the config it stamps on the run is actually stable for the run's full duration, not a lie if a PATCH landed mid-run.
- History is capped at the 50 most recent runs per library (oldest trimmed first) so `eval_runs.json` can't grow unbounded from repeated manual runs.
- GUI: an "Evaluation" section per library (question/keyword form, case list, "Run Evaluation" button, and an expandable, newest-first run history table showing both pass signals and the full answer text per case).

### Fixed
- Real, previously-undiscovered bug found building this feature: `GroundedQA.answer_query`'s aggregate-query routing (`interpret_aggregate_query`) matched on bare superlative words ("longest", "highest", "most", "easiest", ...) with no check that the library was even a cookbook, so an ordinary question like "What is the longest river in the world?" against *any* library got silently misrouted into a recipe-only refusal ("This library doesn't have recipe data extracted yet...") instead of a real grounded answer - confirmed for real: it's exactly the kind of question `scripts/eval_retrieval.py`'s own test corpus uses, which only avoided the bug by calling `retriever.semantic_search()` directly instead of the real chat pipeline. Fixed by gating aggregate-query interception on `library.config.enable_recipe_extraction` - a library that never opted into recipe extraction has no aggregate facts to answer from in the first place, so it should never intercept a query at all. Verified two ways: a new `test_qa.py` regression test, and a real end-to-end run against live Ollama via the new eval feature itself (a "longest river" question against a non-recipe library now answers correctly instead of refusing). Checked against this machine's real persisted libraries (`003`, `sdf`, `demo-library`) - none currently have `enable_recipe_extraction` on or any recipe facts, so this fix changes no existing library's behavior today. Scope note: this closes cross-library leakage (a non-cookbook library never should have been intercepted at all) but does NOT fix bare-substring false positives *within* an opted-in cookbook library - `_MAX_KEYWORDS`/`_STEP_KEYWORDS` still contain words like "hardest"/"quick"/"easy" that can misfire on an ordinary in-book question once recipe extraction is enabled; that's a separate, still-open gap in `recipe_extraction.py`'s keyword lists, not touched here.

127/127 tests passing.

## [Unreleased] - 2026-09-06 (real reranker, RRF fusion, two documented gaps closed)

### Added
- `LexicalOverlapReranker`: replaces the never-wired `DummyReranker`/`None` in the default retrieval pipeline. Boosts candidates that literally contain the query's terms on top of whatever order semantic/hybrid search already produced, using a stable sort so ties (including zero overlap) keep their original relative order. Deliberately not an LLM call or a cross-encoder model - this project already measured a single grounded-QA request at ~2m20s on this CPU-only hardware, and a cross-encoder would pull in sentence-transformers/torch as new dependencies for a project that otherwise only talks to Ollama. Honest caveat: on the hybrid (default) path it reorders the same top-`k` that RRF's keyword leg already ranked using the identical term-overlap formula, so its marginal effect there is small - its main value is reordering `semantic_search()`'s output when used standalone (no keyword searcher / no RRF in play).
- `Retriever.hybrid_search` now fuses semantic and keyword result lists by real reciprocal rank fusion (each chunk scored by the sum of `1/(60 + rank)` across every list it appears in) instead of the previous "semantic first, keyword fills remaining slots" union, which could never let keyword evidence promote a chunk over a weaker semantic-only match. Pulls a 2x-`top_k` candidate pool from each source before fusing so there's actually something to fuse. Fusion uses the RAW (unreranked) semantic order for its semantic leg, not `semantic_search()`'s reranked output - caught before shipping: feeding a lexically-reranked list into RRF as "the semantic signal" silently turns "fuse semantic + lexical" into "fuse lexical + lexical," burying genuine paraphrase-style matches with zero literal query-term overlap. The reranker, when configured, is applied after RRF has already picked the final top_k set (reordering only, not re-selecting) rather than before or over the full candidate pool.
- `AppState.ingest_lock()`: a second concurrent `/ingest` call for the same library now gets a `409` instead of racing the first one on the same on-disk `chunks.json`/`vectors.db`. Not reachable through the GUI (its ingest button disables itself mid-request), only a direct API caller - but a real correctness gap, not a hypothetical one.
- `POST/PATCH .../embedding_model` now validates the model name against locally pulled Ollama models before saving, returning a clear `400` on a typo instead of silently accepting it and only surfacing the problem at the next ingest attempt (`build_providers()`'s loud DummyEmbedder-fallback warning still covers a model removed/unpulled *after* being saved - this is strictly earlier, not a replacement). Skipped entirely when Ollama isn't reachable, matching `/models`' own `force_dummy`/`OllamaUnavailableError` handling. Verified against the real local Ollama daemon, which caught a real bug before it shipped: `ollama.list()` reports fully-qualified names (`nomic-embed-text:latest`), so an exact-match-only check would have rejected this project's own untagged default (`nomic-embed-text`) the instant Ollama was actually reachable - fixed to also accept `name + ":latest"` without stripping tags generally (a mismatched explicit tag, e.g. `qwen3:4b` when only `qwen3:8b` is pulled, is still correctly rejected).

### Fixed
- `SimpleKeywordSearcher.search()` had no zero-score filter, so it always padded its result up to `top_k` with chunks that matched none of the query's terms - harmless when nothing used the list meaningfully, but a real regression risk now that it feeds `hybrid_search`'s RRF: an unfiltered zero-score chunk got a real reciprocal-rank score from its arbitrary position in a same-score stable sort, letting content matching nothing outrank a genuine semantic match. Now excludes zero-score chunks. Caught by a real `SimpleKeywordSearcher` + `InMemoryVectorStore` test (not a stub) using a query term present in none of the chunks.

114/114 tests passing.

## [Unreleased] - 2026-09-06 (answer_source + hybrid search wired in)

### Added
- `ChatResponse.answer_source`: `"llm"` (real Ollama answer), `"dummy"` (fallback), or `"computed"` (deterministic aggregate-query answer, no LLM call). Closes the long-standing "no field distinguishing a real answer from the dummy fallback" gap. GUI shows a small badge for anything other than `"llm"`.
- `SimpleKeywordSearcher`/hybrid search wired into the default retrieval pipeline: semantic results first, keyword results fill remaining `top_k` slots. Rewired `SimpleKeywordSearcher` to read chunks fresh on every search (a callable, not a cached list) since it lives inside `AppState`'s per-library runtime cache and a static snapshot would have gone stale after the next ingest - verified against the real `AppState` class, not just a unit test.

### Fixed
- `Retriever.hybrid_search` took separate `top_k_semantic`/`top_k_keyword` and unioned both result sets in full, which could silently return up to 2x `top_k` chunks. Now takes one `top_k` and caps the combined result at it. Not previously called by anything, so no external behavior changed.

100/100 tests passing.

## [Unreleased] - 2026-09-06 (second low-overhead sweep, fresh eyes)

### Fixed
- `SimpleKeywordSearcher.search()` scored every chunk identically - it counted query terms within the query text itself instead of the chunk text, so it never actually searched anything (would have silently done nothing the moment it was wired into hybrid search). Not currently used by the default pipeline, so dormant rather than user-visible, but had zero test coverage, which is exactly why it went unnoticed. Fixed and added `tests/test_keyword_searcher.py` (2 new tests, one of which would fail against the old code).

95/95 tests passing.

## [Unreleased] - 2026-09-06 (low-overhead bug sweep)

### Fixed
- `chunk_size`/`chunk_overlap`/`top_k` had no validation at the API boundary - `chunk_size <= 0` in particular was silently treated as "don't sub-split" rather than rejected (the GUI's own `min` attributes masked this for GUI users). Added Pydantic `Field(gt=0)`/`Field(ge=0)` constraints so the API now returns a clear `422` instead.
- GUI error messages showed raw JSON (`{"detail":"..."}`, or a JSON array for validation errors) instead of the actual message. Added `describeErrorBody()` to extract it properly.

## [Unreleased] - 2026-09-06 (AppState concurrency locking)

### Fixed
- A config `PATCH`/`DELETE` could close a library's vector-store connection while an in-flight `/ingest`, `/chat`, or source change was still using it (a pre-existing gap, now more likely to actually bite since recipe extraction can make an ingest run up to an hour). `AppState.exclusive()` now refuses config changes/deletion immediately (`409 Conflict`, not a block that could hang the UI for an hour) when the library is in use; `AppState.use_runtime()` marks chat/ingest/source-change requests as in-use so this can't race. `get_runtime()` removed in favor of `use_runtime()` everywhere. Verified with real concurrent threads (`tests/test_state.py`, 6 new) and a real HTTP-level 409 test driving the actual FastAPI app (`tests/test_api.py`). 90/90 tests passing.

## [Unreleased] - 2026-09-06 (full-book validation run: 2 real bugs fixed, 1 honesty improvement)

Ran the complete 211-recipe book (not an excerpt) through real ingestion with live Ollama: 222 segments found, 216 (97%) successfully extracted, in 41.6 minutes. Found and fixed two real bugs no unit test or small excerpt had caught:

### Fixed
- The persisted recipe name trusted the LLM's own self-reported `recipe_name` JSON field instead of the already-verified segment heading. Confirmed via the real run: the book has exactly one `"QUEEN'S SOUP"` heading, yet ingestion persisted a phantom second fact under `"Queen's Soup"` with different (wrong) counts - the LLM had misreported which recipe it was looking at. Now uses the segment's own heading (reliable by construction) for the name; the LLM is only trusted for ingredients/step_count.
- `citation_id`/`chunk_id` for a recipe citation were built from `document_id` + `recipe_name` alone, which collide when a book has two recipes sharing a title - confirmed real on this book (`"ROMAN FRY"` and `"BISCUIT"` each appear twice). Added `make_recipe_fact_citation_id()`, hashing in the source excerpt too so genuinely distinct recipes never collide even when same-titled.

### Changed
- Aggregate answers ("fewest ingredients", "simplest") now add an honest caveat when more than 3 recipes tie at the extreme value, rather than presenting it as precise. Real evidence: the full-book run produced an 8-way and a 16-way tie at the minimum, which is far more likely small-model under-extraction than genuine equality.

## [Unreleased] - 2026-09-06 (recipe extraction GUI checkbox)

### Added
- "Extract recipe facts" checkbox in the library create/edit forms, wiring up `enable_recipe_extraction` (previously API/config-only) so it's reachable from the app itself. Verified end-to-end with a real browser against the real backend: creating a library with it on, reopening to confirm it's checked, unchecking and saving, reopening to confirm it persisted off.

## [Unreleased] - 2026-09-06 (recipe segmentation hardened)

### Fixed
- `recipe_extraction.segment_recipes` found zero real recipes in a real Title-Case-headed cookbook (only recognized ALL-CAPS headings). Now case-agnostic: a heading is a short line (≤8 words) where every word is ALL-CAPS or Capitalized. Verified against a real book: "Doughnuts (Sour Milk)", "Spice Cakes", "Soft Gingerbread" now correctly found.
- Rejects single-word candidates under 5 characters with no vowel, eliminating the OCR-noise false positives ("WIA", "AAT", "DREN") found in the same real book's garbled front matter, and rejects candidates containing a period or digit (also kills byline/ingredient-line false positives as a side effect).
- Fixed a real regression the looser rule introduced: two different real patterns of adjacent heading-shaped lines (a title + parenthetical subtitle vs. a nested front-matter chain) need opposite tie-breaking (keep first vs. keep last name) - handled by treating parenthetical lines as pure separators that never rename the group. Caught by a test before shipping.
- 6 new tests (19/19 in `test_recipe_extraction.py`, 79/79 full suite).

### Known, not fixed
- The same real book's badly-OCR'd front matter still produces ~600 false-positive segments from longer garbled tokens (5+ characters, contains a vowel) that pass the new noise filter. Reliably telling real English words from OCR gibberish needs a dictionary or language-model check, not another regex - flagged for a scope decision rather than chased with more tuning.

## [Unreleased] - 2026-09-06 (retrieval regression check)

### Added
- `scripts/eval_retrieval.py` + `scripts/retrieval_eval_corpus.txt`: a small fixed-question retrieval regression check using real Ollama embeddings, for the failure class the fast dummy-provider `pytest` suite can't catch by design - a chunking/embedding change that runs without error but silently surfaces the wrong chunks (the actual root cause of three separate real bugs found by hand earlier this session). Refuses to run rather than report a false result if Ollama isn't reachable. Documented in `docs/how_to_use.md` §12.

## [Unreleased] - 2026-09-06 (PDF extraction)

### Changed
- `PyPDF2` → `pypdf` (the actively maintained fork; identical API) for PDF loading. Tested first rather than assumed: compared `PyPDF2`, `pypdf`, and `PyMuPDF` on the same real scanned public-domain cookbook PDF and got character-for-character identical output from all three, including the real OCR corruption already baked into that PDF's text layer ("V4 teaspoon soda" for "¼", "egos" for "eggs"). No PDF library swap fixes bad source OCR, so `PyMuPDF` was not adopted (no evidence it helps, and it carries AGPL licensing this project hasn't evaluated) - the `pypdf` swap is justified purely as removing a dependency on an abandoned package.

### Found (not yet fixed)
- The same real-cookbook test surfaced a genuine Phase 6 failure mode: `recipe_extraction.segment_recipes`'s ALL-CAPS-heading heuristic finds zero real recipes and 123 false-positive "recipes" from OCR-garbled front matter on a real Title-Case-headed cookbook. Confirmed the segmentation approach only works for the one ALL-CAPS-convention book tested so far - see the plan doc's "PDF extraction quality" section for detail. Not fixed; needs a scope decision before this feature is safe to recommend on an arbitrary cookbook.

## [Unreleased] - 2026-09-05 (Phase 6, wired)

### Added
- `LibraryConfig.enable_recipe_extraction` (opt-in per library, default off) wires the Phase 6 prototype into the real pipeline: `IngestionPipeline` now extracts and persists structured recipe facts (`RecipeFact`, `recipe_facts.json`) during ingestion when enabled, tracked with its own `recipe_extraction_signature` (same pattern as `embedding_signature`/`chunking_signature`, so toggling the flag on an already-ingested library actually takes effect on the next ingest instead of being silently skipped).
- `GroundedQA.answer_query` now answers superlative/aggregate recipe questions ("which recipe uses the fewest ingredients") from a real Python MIN/MAX over `RecipeFact` data instead of vector search — the actual fix for the gap `RuleBasedQueryPlanner`'s "comparison"/"synthesis" labels never closed. Cost/price questions get an explicit refusal instead of a fabricated answer, since cost isn't in the source text. Verified end-to-end against a real cookbook excerpt and live Ollama: correct answers, correct refusal, and correctly did NOT misroute a plain "what ingredients are in X" lookup question (a real false-positive caught and fixed during testing).
- 23 new tests across `test_recipe_extraction.py`, `test_ingestion.py`, `test_qa.py`, `test_models.py`, `test_storage.py`, and new `test_query_planner.py`. 74/74 passing.

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
