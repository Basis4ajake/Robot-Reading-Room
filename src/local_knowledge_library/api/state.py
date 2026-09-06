from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterator

from ..chunkers import ParagraphChunker
from ..ingestion import IngestionPipeline
from ..loaders import MarkdownLoader, PdfLoader, TextLoader
from ..providers import LexicalOverlapReranker, SimpleKeywordSearcher, SqliteVectorStore
from ..providers.factory import build_providers
from ..qa import GroundedQA
from ..query_planner import QueryPlanner
from ..registry import LibraryRegistry
from ..retrieval import Retriever


class LibraryBusyError(RuntimeError):
    """Raised when an operation can't proceed because a library's runtime is
    currently in use by another in-flight request.

    Two directions:
    - exclusive() (config update, delete) refuses immediately if a chat/
      ingest/source-change request is mid-flight, rather than blocking -
      an ingest with recipe extraction enabled can run for the better part
      of an hour, and blocking a settings save for that long would just
      move the "hangs" problem from "data silently corrupted" to "UI
      silently frozen," not actually fix anything.
    - use_runtime() (chat, ingest, source changes) refuses if a config
      update/delete is (briefly) in progress, for the same reason in
      reverse - it should be a rare, sub-second window in practice.
    - ingest_lock() refuses a second /ingest call for a library that's
      already mid-ingest. use_runtime() alone happily lets multiple
      callers hold a library's runtime concurrently (that's the whole
      point - chat shouldn't wait out a long ingest), which is fine for
      read-only chat but not for two ingests racing on the same on-disk
      state (chunks.json, vectors.db). Not reachable through the GUI
      today (its ingest button disables itself while a request is in
      flight), only a direct API caller.
    """


@dataclass
class LibraryRuntime:
    vector_store: SqliteVectorStore
    pipeline: IngestionPipeline
    retriever: Retriever
    qa: GroundedQA


class AppState:
    def __init__(self, data_dir: str, force_dummy: bool = False):
        self.registry = LibraryRegistry(data_dir)
        self.force_dummy = force_dummy
        self._runtimes: Dict[str, LibraryRuntime] = {}
        # Guards _runtimes and _active_uses only, held briefly (dict
        # mutation, never for the duration of an ingest/chat call itself).
        self._lock = threading.Lock()
        # Per library_id: 0 = idle, >0 = N in-flight use_runtime() callers,
        # -1 = an exclusive() (config update/delete) is in progress.
        self._active_uses: Dict[str, int] = {}
        # library_ids currently mid-/ingest - see ingest_lock().
        self._ingesting: set[str] = set()

    @contextmanager
    def use_runtime(self, library_id: str) -> Iterator[LibraryRuntime]:
        """Get a library's runtime for the duration of the `with` block.

        Marks the library as in-use so a concurrent exclusive() (PATCH/
        DELETE) refuses to close the runtime out from under this request
        instead of silently corrupting it - the actual bug this whole
        mechanism exists to prevent. Does NOT serialize concurrent
        use_runtime() callers against each other (e.g. chat during a long
        ingest is still allowed to proceed) - that's a separate, lesser
        concern than the invalidate-mid-use race this targets, and forcing
        a chat request to wait out a ~40-minute ingest would trade one bad
        UX for another.
        """
        with self._lock:
            if self._active_uses.get(library_id, 0) < 0:
                raise LibraryBusyError(
                    f"Library {library_id!r} is being updated right now - try again in a moment."
                )
            if library_id not in self._runtimes:
                self._runtimes[library_id] = self._build_runtime(library_id)
            runtime = self._runtimes[library_id]
            self._active_uses[library_id] = self._active_uses.get(library_id, 0) + 1
        try:
            yield runtime
        finally:
            with self._lock:
                self._active_uses[library_id] -= 1

    @contextmanager
    def exclusive(self, library_id: str) -> Iterator[None]:
        """Exclusive access for an operation that changes a library's config
        or deletes it. Raises LibraryBusyError immediately (never blocks)
        if a chat/ingest/source-change request currently holds the runtime -
        see LibraryBusyError's docstring for why this refuses rather than
        waits.
        """
        with self._lock:
            if self._active_uses.get(library_id, 0) != 0:
                raise LibraryBusyError(
                    f"Library {library_id!r} is currently in use (an ingest, chat, or source "
                    "change is in flight) - wait for it to finish and try again."
                )
            self._active_uses[library_id] = -1
        try:
            yield
        finally:
            with self._lock:
                self._active_uses[library_id] = 0

    @contextmanager
    def ingest_lock(self, library_id: str) -> Iterator[None]:
        """Refuses immediately (never blocks) if another /ingest call for
        this same library is already in flight - see LibraryBusyError's
        docstring for why this exists alongside use_runtime()/exclusive().
        """
        with self._lock:
            if library_id in self._ingesting:
                raise LibraryBusyError(
                    f"Library {library_id!r} is already being ingested - wait for it to finish "
                    "and try again."
                )
            self._ingesting.add(library_id)
        try:
            yield
        finally:
            with self._lock:
                self._ingesting.discard(library_id)

    def invalidate(self, library_id: str) -> None:
        """Close and drop the cached runtime, forcing a rebuild on next use.

        Callers that could race an in-flight request (a config PATCH or
        library DELETE) must call this from within an exclusive() block -
        invalidate() itself does not check busy status, since by the time
        it's reasonable to call it the exclusive() context has already
        established nothing else is using the runtime.
        """
        with self._lock:
            runtime = self._runtimes.pop(library_id, None)
        if runtime is not None:
            runtime.vector_store.close()

    def close(self) -> None:
        """App shutdown - force-close every runtime regardless of in-flight
        use tracking, since the process is exiting either way."""
        with self._lock:
            runtimes = list(self._runtimes.values())
            self._runtimes.clear()
            self._active_uses.clear()
            self._ingesting.clear()
        for runtime in runtimes:
            runtime.vector_store.close()

    def _build_runtime(self, library_id: str) -> LibraryRuntime:
        library = self.registry.get_library(library_id)
        embedder, llm_provider = build_providers(library.config, force_dummy=self.force_dummy)

        vector_db = library.index_dir / "vectors.db"
        vector_store = SqliteVectorStore(str(vector_db))

        loaders = [TextLoader(), MarkdownLoader(), PdfLoader()]
        chunker = ParagraphChunker(chunk_size=library.config.chunk_size, chunk_overlap=library.config.chunk_overlap)
        pipeline = IngestionPipeline(
            loaders, chunker, embedder, vector_store, debug=library.config.debug, llm_provider=llm_provider
        )
        # Reads library.chunks fresh from disk on every search (see
        # SimpleKeywordSearcher's docstring) rather than a snapshot from
        # this runtime's build time, which would go stale after the next
        # ingest since this runtime is cached and reused across requests.
        keyword_searcher = SimpleKeywordSearcher(
            lambda: self.registry.get_library(library_id).chunks.values()
        )
        retriever = Retriever(
            vector_store=vector_store,
            embedder=embedder,
            keyword_searcher=keyword_searcher,
            reranker=LexicalOverlapReranker(),
            debug=library.config.debug,
        )
        planner = QueryPlanner()
        qa = GroundedQA(retriever, planner, llm_provider, debug=library.config.debug)

        return LibraryRuntime(vector_store=vector_store, pipeline=pipeline, retriever=retriever, qa=qa)
