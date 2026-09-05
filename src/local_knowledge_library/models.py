from __future__ import annotations

import dataclasses
import hashlib
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def _make_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


@dataclass
class StructureMetadata:
    page_number: Optional[int] = None
    chapter: Optional[str] = None
    section: Optional[str] = None
    paragraph: Optional[str] = None
    extra: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Optional[str]]:
        return {
            "page_number": self.page_number,
            "chapter": self.chapter,
            "section": self.section,
            "paragraph": self.paragraph,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "StructureMetadata":
        return cls(
            page_number=data.get("page_number"),
            chapter=data.get("chapter"),
            section=data.get("section"),
            paragraph=data.get("paragraph"),
            extra=data.get("extra", {}),
        )


@dataclass
class LibraryConfig:
    library_id: str
    name: str
    description: str = ""
    data_dir: str = "./data/libraries"
    chunk_size: int = 200
    chunk_overlap: int = 50
    top_k: int = 5
    debug: bool = False
    llm_model: str = "qwen2:1.5b"
    # Must be a real embedding-capable model, never None: an unset
    # embedding_model makes OllamaQwenProvider default to llm_model, which
    # for a chat-only model like qwen2:1.5b always fails and silently falls
    # back to fake DummyEmbedder vectors (see project_embedding_bug_2026_09_05
    # in memory). nomic-embed-text matches the GUI's own default.
    embedding_model: Optional[str] = "nomic-embed-text"

    def to_dict(self) -> Dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "LibraryConfig":
        return cls(**data)


@dataclass
class LibraryMetadata:
    library_id: str
    name: str
    description: str
    source_count: int = 0
    document_count: int = 0
    chunk_count: int = 0

    def to_dict(self) -> Dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "LibraryMetadata":
        return cls(**data)


@dataclass
class SourceMetadata:
    source_id: str
    library_id: str
    source_path: str
    filename: str
    file_type: str
    content_hash: str

    def to_dict(self) -> Dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "SourceMetadata":
        return cls(**data)


@dataclass
class DocumentMetadata:
    source_id: str
    library_id: str
    document_id: str
    source_path: str
    filename: str
    file_type: str
    title: Optional[str]
    author: Optional[str]
    publisher: Optional[str]
    publication_date: Optional[str]
    edition: Optional[str]
    isbn: Optional[str]
    content_hash: str
    structure: StructureMetadata = field(default_factory=StructureMetadata)
    text: Optional[str] = None
    # PDF only: non-empty-paragraph count per page, in reading order, so a
    # chunker can attribute each chunk to its real page instead of the one
    # page number `detect_structure` finds via a single whole-document regex
    # search. Populated fresh by PdfLoader on every load; deliberately not
    # persisted (excluded from to_dict/from_dict) since it's only needed
    # within the ingest pass that just loaded this document.
    page_boundaries: Optional[List[int]] = None

    def to_dict(self, include_text: bool = False) -> Dict:
        result = {
            "source_id": self.source_id,
            "library_id": self.library_id,
            "document_id": self.document_id,
            "source_path": self.source_path,
            "filename": self.filename,
            "file_type": self.file_type,
            "title": self.title,
            "author": self.author,
            "publisher": self.publisher,
            "publication_date": self.publication_date,
            "edition": self.edition,
            "isbn": self.isbn,
            "content_hash": self.content_hash,
            "structure": self.structure.to_dict(),
        }
        if include_text:
            result["text"] = self.text
        return result

    @classmethod
    def from_dict(cls, data: Dict) -> "DocumentMetadata":
        return cls(
            source_id=data["source_id"],
            library_id=data["library_id"],
            document_id=data["document_id"],
            source_path=data["source_path"],
            filename=data["filename"],
            file_type=data["file_type"],
            title=data.get("title"),
            author=data.get("author"),
            publisher=data.get("publisher"),
            publication_date=data.get("publication_date"),
            edition=data.get("edition"),
            isbn=data.get("isbn"),
            content_hash=data["content_hash"],
            structure=StructureMetadata.from_dict(data.get("structure", {})),
            text=data.get("text"),
        )


@dataclass
class Chunk:
    chunk_id: str
    library_id: str
    source_id: str
    document_id: str
    text: str
    metadata: Dict[str, str]
    citation_id: str

    def to_dict(self) -> Dict:
        return {
            "chunk_id": self.chunk_id,
            "library_id": self.library_id,
            "source_id": self.source_id,
            "document_id": self.document_id,
            "text": self.text,
            "metadata": self.metadata,
            "citation_id": self.citation_id,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Chunk":
        return cls(
            chunk_id=data["chunk_id"],
            library_id=data["library_id"],
            source_id=data["source_id"],
            document_id=data["document_id"],
            text=data["text"],
            metadata=data.get("metadata", {}),
            citation_id=data["citation_id"],
        )


@dataclass
class Citation:
    citation_id: str
    library_id: str
    source_id: str
    document_id: str
    chunk_id: str
    page_number: Optional[int] = None
    chapter: Optional[str] = None
    section: Optional[str] = None
    paragraph: Optional[str] = None
    filename: Optional[str] = None
    file_type: Optional[str] = None

    def to_dict(self) -> Dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "Citation":
        return cls(**data)


@dataclass
class IngestionState:
    library_id: str
    sources: Dict[str, str] = field(default_factory=dict)
    documents: Dict[str, str] = field(default_factory=dict)
    embedding_signature: Optional[str] = None
    # Same idea as embedding_signature, for the chunker: chunk_size/overlap
    # only take effect on documents that actually get (re-)processed, so a
    # config change with no content change would otherwise be silently
    # ignored by incremental (hash-based) ingestion forever.
    chunking_signature: Optional[str] = None

    def to_dict(self) -> Dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "IngestionState":
        return cls(
            library_id=data["library_id"],
            sources=data.get("sources", {}),
            documents=data.get("documents", {}),
            embedding_signature=data.get("embedding_signature"),
            chunking_signature=data.get("chunking_signature"),
        )


def compute_content_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def compute_path_hash(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8192):
            digest.update(chunk)
    return digest.hexdigest()


def make_document_id(source_id: str, filename: str) -> str:
    normalized = f"{source_id}:{filename}"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def make_chunk_id(source_id: str = "", document_id: str = "", offset: int = 0) -> str:
    normalized = f"{source_id}:{document_id}:{offset}"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def make_source_id(source_path: str) -> str:
    normalized = Path(source_path).resolve().as_posix()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
