from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from ..chunkers import ParagraphChunker
from ..ingestion import IngestionPipeline
from ..loaders import MarkdownLoader, PdfLoader, TextLoader
from ..providers import SqliteVectorStore
from ..providers.factory import build_providers
from ..qa import GroundedQA
from ..query_planner import QueryPlanner
from ..registry import LibraryRegistry
from ..retrieval import Retriever


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

    def get_runtime(self, library_id: str) -> LibraryRuntime:
        if library_id not in self._runtimes:
            self._runtimes[library_id] = self._build_runtime(library_id)
        return self._runtimes[library_id]

    def invalidate(self, library_id: str) -> None:
        runtime = self._runtimes.pop(library_id, None)
        if runtime is not None:
            runtime.vector_store.close()

    def close(self) -> None:
        for library_id in list(self._runtimes.keys()):
            self.invalidate(library_id)

    def _build_runtime(self, library_id: str) -> LibraryRuntime:
        library = self.registry.get_library(library_id)
        embedder, llm_provider = build_providers(library.config, force_dummy=self.force_dummy)

        vector_db = library.index_dir / "vectors.db"
        vector_store = SqliteVectorStore(str(vector_db))

        loaders = [TextLoader(), MarkdownLoader(), PdfLoader()]
        chunker = ParagraphChunker()
        pipeline = IngestionPipeline(
            loaders, chunker, embedder, vector_store, debug=library.config.debug
        )
        retriever = Retriever(vector_store=vector_store, embedder=embedder, debug=library.config.debug)
        planner = QueryPlanner()
        qa = GroundedQA(retriever, planner, llm_provider, debug=library.config.debug)

        return LibraryRuntime(vector_store=vector_store, pipeline=pipeline, retriever=retriever, qa=qa)
