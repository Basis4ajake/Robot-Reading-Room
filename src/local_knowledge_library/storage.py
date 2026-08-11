from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .models import (
    Citation,
    DocumentMetadata,
    IngestionState,
    LibraryConfig,
    LibraryMetadata,
    SourceMetadata,
    Chunk,
    compute_content_hash,
    make_source_id,
)


class KnowledgeLibrary:
    def __init__(self, config: LibraryConfig, library_dir: str):
        self.config = config
        self.library_dir = Path(library_dir)
        self.library_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.library_dir / "meta.json"
        self.sources_path = self.library_dir / "sources.json"
        self.documents_path = self.library_dir / "documents.json"
        self.chunks_path = self.library_dir / "chunks.json"
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
        self.state = IngestionState(library_id=self.config.library_id)

        self.load_state()
        self.load_sources()
        self.load_documents()
        self.load_chunks()
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
        return cls(config, str(library_dir))

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

    def load_state(self) -> None:
        data = self.load_json(self.state_path, None)
        if data:
            self.state = IngestionState.from_dict(data)

    def persist(self) -> None:
        self.save_json(self.meta_path, self.metadata.to_dict())
        self.save_json(self.sources_path, [s.to_dict() for s in self.sources.values()])
        self.save_json(self.documents_path, {did: d.to_dict() for did, d in self.documents.items()})
        self.save_json(self.chunks_path, {cid: c.to_dict() for cid, c in self.chunks.items()})
        self.save_json(self.state_path, self.state.to_dict())

    def add_source(self, source_path: str) -> SourceMetadata:
        normalized = str(Path(source_path).resolve())
        if not Path(normalized).exists():
            raise FileNotFoundError(f"Source path does not exist: {normalized}")
        source_id = make_source_id(normalized)
        filename = Path(normalized).name
        extension = Path(normalized).suffix.lower().lstrip(".")
        content_hash = compute_content_hash(normalized)
        source = SourceMetadata(
            source_id=source_id,
            library_id=self.config.library_id,
            source_path=normalized,
            filename=filename,
            file_type=extension,
            content_hash=content_hash,
        )
        self.sources[source_id] = source
        self.state.sources[normalized] = content_hash
        self.metadata.source_count = len(self.sources)
        self.persist()
        return source

    def remove_source(self, source_id: str) -> None:
        if source_id not in self.sources:
            return
        self.sources.pop(source_id)
        self.metadata.source_count = len(self.sources)
        to_remove = [doc_id for doc_id, doc in self.documents.items() if doc.source_id == source_id]
        for doc_id in to_remove:
            self.documents.pop(doc_id, None)
        self.persist()

    def list_sources(self) -> List[SourceMetadata]:
        return list(self.sources.values())

    def register_document(self, document: DocumentMetadata) -> None:
        self.documents[document.document_id] = document
        self.state.documents[document.document_id] = document.content_hash
        self.metadata.document_count = len(self.documents)

    def register_chunk(self, chunk: Chunk) -> None:
        self.chunks[chunk.chunk_id] = chunk
        self.metadata.chunk_count = len(self.chunks)

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
