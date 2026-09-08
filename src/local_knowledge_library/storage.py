from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .models import (
    Citation,
    DocumentMetadata,
    EvalCase,
    EvalRun,
    IngestionState,
    LibraryConfig,
    LibraryMetadata,
    RecipeFact,
    SourceMetadata,
    Chunk,
    compute_path_hash,
    make_id,
    make_source_id,
)

# Oldest runs are trimmed once history exceeds this, so eval_runs.json can't
# grow unbounded from repeated manual "Run Evaluation" clicks.
_MAX_EVAL_RUNS = 50


class KnowledgeLibrary:
    def __init__(self, config: LibraryConfig, library_dir: str):
        self.config = config
        self.library_dir = Path(library_dir)
        self.library_dir.mkdir(parents=True, exist_ok=True)
        self.config_path = self.library_dir / "config.json"
        self.meta_path = self.library_dir / "meta.json"
        self.sources_path = self.library_dir / "sources.json"
        self.documents_path = self.library_dir / "documents.json"
        self.chunks_path = self.library_dir / "chunks.json"
        self.recipe_facts_path = self.library_dir / "recipe_facts.json"
        self.eval_cases_path = self.library_dir / "eval_cases.json"
        self.eval_runs_path = self.library_dir / "eval_runs.json"
        self.state_path = self.library_dir / "state.json"
        self.index_dir = self.library_dir / "indexes"
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.metadata = LibraryMetadata(
            library_id=self.config.library_id,
            name=self.config.name,
            description=self.config.description,
        )
        self.sources: Dict[str, SourceMetadata] = {}
        self.documents: Dict[str, DocumentMetadata] = {}
        self.chunks: Dict[str, Chunk] = {}
        self.recipe_facts: Dict[str, List[RecipeFact]] = {}
        self.eval_cases: Dict[str, EvalCase] = {}
        # Chronological (oldest first) on disk and in memory; list_eval_runs()
        # reverses for display, since the GUI wants newest-first history.
        self.eval_runs: List[EvalRun] = []
        self.state = IngestionState(library_id=self.config.library_id)

    def remove_document(self, document_id: str) -> None:
        document = self.documents.pop(document_id, None)
        if document is None:
            return
        self.state.documents.pop(document_id, None)
        chunk_ids = [cid for cid, chunk in self.chunks.items() if chunk.document_id == document_id]
        for chunk_id in chunk_ids:
            self.chunks.pop(chunk_id, None)
        self.recipe_facts.pop(document_id, None)
        self.metadata.document_count = len(self.documents)
        self.metadata.chunk_count = len(self.chunks)

    def remove_source(self, source_id: str) -> None:
        if source_id not in self.sources:
            return
        source = self.sources.pop(source_id)
        self.metadata.source_count = len(self.sources)
        to_remove = [doc_id for doc_id, doc in self.documents.items() if doc.source_id == source_id]
        for doc_id in to_remove:
            self.remove_document(doc_id)
        self.state.sources.pop(source.source_path, None)
        self.persist()

        self.load_state()
        self.load_sources()
        self.load_documents()
        self.load_chunks()
        self.load_recipe_facts()
        self.load_meta()

    @staticmethod
    def library_path(config: LibraryConfig) -> Path:
        base = Path(config.data_dir)
        return base / config.library_id

    @classmethod
    def create(cls, config: LibraryConfig) -> "KnowledgeLibrary":
        library_dir = cls.library_path(config)
        if library_dir.exists():
            raise FileExistsError(f"Library {config.library_id} already exists at {library_dir}")
        return cls(config, str(library_dir))

    @classmethod
    def open(cls, config: LibraryConfig) -> "KnowledgeLibrary":
        library_dir = cls.library_path(config)
        if not library_dir.exists():
            raise FileNotFoundError(f"Library {config.library_id} does not exist at {library_dir}")
        config_path = library_dir / "config.json"
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as handle:
                config = LibraryConfig.from_dict(json.load(handle))
        library = cls(config, str(library_dir))
        library.load_meta()
        library.load_sources()
        library.load_documents()
        library.load_chunks()
        library.load_recipe_facts()
        library.load_eval_cases()
        library.load_eval_runs()
        library.load_state()
        return library

    @classmethod
    def open_by_id(cls, library_id: str, data_dir: str) -> "KnowledgeLibrary":
        placeholder = LibraryConfig(library_id=library_id, name=library_id, data_dir=data_dir)
        return cls.open(placeholder)

    def load_json(self, path: Path, default):
        if not path.exists():
            return default
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def save_json(self, path: Path, payload) -> None:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    def load_meta(self) -> None:
        data = self.load_json(self.meta_path, None)
        if data:
            self.metadata = LibraryMetadata.from_dict(data)

    def load_sources(self) -> None:
        data = self.load_json(self.sources_path, [])
        for entry in data:
            source = SourceMetadata.from_dict(entry)
            self.sources[source.source_id] = source

    def load_documents(self) -> None:
        data = self.load_json(self.documents_path, {})
        for document_id, entry in data.items():
            self.documents[document_id] = DocumentMetadata.from_dict(entry)

    def load_chunks(self) -> None:
        data = self.load_json(self.chunks_path, {})
        for chunk_id, entry in data.items():
            self.chunks[chunk_id] = Chunk.from_dict(entry)

    def load_recipe_facts(self) -> None:
        data = self.load_json(self.recipe_facts_path, {})
        for document_id, entries in data.items():
            self.recipe_facts[document_id] = [RecipeFact.from_dict(entry) for entry in entries]

    def load_eval_cases(self) -> None:
        data = self.load_json(self.eval_cases_path, {})
        for eval_case_id, entry in data.items():
            self.eval_cases[eval_case_id] = EvalCase.from_dict(entry)

    def load_eval_runs(self) -> None:
        data = self.load_json(self.eval_runs_path, [])
        self.eval_runs = [EvalRun.from_dict(entry) for entry in data]

    def load_state(self) -> None:
        data = self.load_json(self.state_path, None)
        if data:
            self.state = IngestionState.from_dict(data)

    def load_config(self) -> None:
        data = self.load_json(self.config_path, None)
        if data:
            self.config = LibraryConfig.from_dict(data)

    def persist(self) -> None:
        self.save_json(self.config_path, self.config.to_dict())
        self.save_json(self.meta_path, self.metadata.to_dict())
        self.save_json(self.sources_path, [s.to_dict() for s in self.sources.values()])
        self.save_json(self.documents_path, {did: d.to_dict() for did, d in self.documents.items()})
        self.save_json(self.chunks_path, {cid: c.to_dict() for cid, c in self.chunks.items()})
        self.save_json(
            self.recipe_facts_path,
            {did: [f.to_dict() for f in facts] for did, facts in self.recipe_facts.items()},
        )
        self.save_json(
            self.eval_cases_path,
            {eid: c.to_dict() for eid, c in self.eval_cases.items()},
        )
        self.save_json(self.eval_runs_path, [run.to_dict() for run in self.eval_runs])
        self.save_json(self.state_path, self.state.to_dict())

    def add_source(self, source_path: str) -> SourceMetadata:
        normalized = str(Path(source_path).resolve())
        if not Path(normalized).exists():
            raise FileNotFoundError(f"Source path does not exist: {normalized}")
        source_id = make_source_id(normalized)
        filename = Path(normalized).name
        extension = Path(normalized).suffix.lower().lstrip(".")
        content_hash = compute_path_hash(normalized)
        source = SourceMetadata(
            source_id=source_id,
            library_id=self.config.library_id,
            source_path=normalized,
            filename=filename,
            file_type=extension,
            content_hash=content_hash,
        )
        self.sources[source_id] = source
        self.metadata.source_count = len(self.sources)
        self.persist()
        return source

    def list_sources(self) -> List[SourceMetadata]:
        return list(self.sources.values())

    def register_document(self, document: DocumentMetadata) -> None:
        self.documents[document.document_id] = document
        self.state.documents[document.document_id] = document.content_hash
        self.metadata.document_count = len(self.documents)

    def register_chunk(self, chunk: Chunk) -> None:
        self.chunks[chunk.chunk_id] = chunk
        self.metadata.chunk_count = len(self.chunks)

    def register_recipe_facts(self, document_id: str, facts: List[RecipeFact]) -> None:
        self.recipe_facts[document_id] = facts

    def list_recipe_facts(self) -> List[RecipeFact]:
        return [fact for facts in self.recipe_facts.values() for fact in facts]

    def add_eval_case(self, question: str, expected_keyword: str) -> EvalCase:
        case = EvalCase(
            eval_case_id=make_id("eval-case"),
            library_id=self.config.library_id,
            question=question,
            expected_keyword=expected_keyword,
        )
        self.eval_cases[case.eval_case_id] = case
        self.persist()
        return case

    def list_eval_cases(self) -> List[EvalCase]:
        return list(self.eval_cases.values())

    def remove_eval_case(self, eval_case_id: str) -> None:
        if self.eval_cases.pop(eval_case_id, None) is not None:
            self.persist()

    def register_eval_run(self, run: EvalRun) -> None:
        self.eval_runs.append(run)
        if len(self.eval_runs) > _MAX_EVAL_RUNS:
            self.eval_runs = self.eval_runs[-_MAX_EVAL_RUNS:]
        self.persist()

    def list_eval_runs(self) -> List[EvalRun]:
        return list(reversed(self.eval_runs))

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        return self.chunks.get(chunk_id)

    def get_citation(self, chunk: Chunk) -> Citation:
        return Citation(
            citation_id=chunk.citation_id,
            library_id=chunk.library_id,
            source_id=chunk.source_id,
            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id,
            page_number=chunk.metadata.get("page_number"),
            chapter=chunk.metadata.get("chapter"),
            section=chunk.metadata.get("section"),
            paragraph=chunk.metadata.get("paragraph"),
            filename=self.sources[chunk.source_id].filename if chunk.source_id in self.sources else None,
            file_type=self.sources[chunk.source_id].file_type if chunk.source_id in self.sources else None,
        )

    def find_documents_by_source(self, source_id: str) -> Iterable[DocumentMetadata]:
        return [doc for doc in self.documents.values() if doc.source_id == source_id]

    def find_chunks_by_document(self, document_id: str) -> Iterable[Chunk]:
        return [chunk for chunk in self.chunks.values() if chunk.document_id == document_id]
