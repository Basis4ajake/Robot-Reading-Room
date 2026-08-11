from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .abstracts import Chunker, DocumentLoader, Embedder, VectorStore
from .models import (
    Chunk,
    DocumentMetadata,
    IngestionState,
    LibraryConfig,
    LibraryMetadata,
    SourceMetadata,
    compute_content_hash,
    make_chunk_id,
    make_document_id,
    make_source_id,
    StructureMetadata,
)
from .storage import KnowledgeLibrary


class IngestionPipeline:
    def __init__(
        self,
        loaders: Iterable[DocumentLoader],
        chunker: Chunker,
        embedder: Embedder,
        vector_store: VectorStore,
        debug: bool = False,
    ):
        self.loaders = {ext: loader for loader in loaders for ext in loader.supported_extensions()}
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.debug = debug

    def get_loader_for_source(self, source_path: str) -> DocumentLoader:
        extension = Path(source_path).suffix.lower().lstrip(".")
        loader = self.loaders.get(extension)
        if loader is None:
            raise ValueError(f"No loader registered for extension: {extension}")
        return loader

    def ingest(self, library: KnowledgeLibrary) -> Dict[str, List[str]]:
        processed: List[str] = []
        skipped: List[str] = []
        removed: List[str] = []
        existing_sources = {src.source_path for src in library.list_sources()}

        for source in library.list_sources():
            if not Path(source.source_path).exists():
                removed.append(source.source_id)
                library.remove_source(source.source_id)
                continue
            loader = self.get_loader_for_source(source.source_path)
            for document in loader.load(source.source_path):
                document.source_id = source.source_id
                document.library_id = library.config.library_id
                document.source_path = source.source_path
                document.document_id = make_document_id(source.source_id, document.filename)
                document.content_hash = compute_content_hash(document.text or "")
                document.structure = self.detect_structure(document)
                previous_hash = library.state.documents.get(document.document_id)
                if previous_hash == document.content_hash:
                    skipped.append(document.document_id)
                    continue
                chunks = self.chunker.chunk(document)
                embeddings = self.embedder.embed_text([chunk.text for chunk in chunks])
                self.vector_store.add(chunks, embeddings)
                library.register_document(document)
                for chunk in chunks:
                    library.register_chunk(chunk)
                processed.append(document.document_id)
                if self.debug:
                    print(f"[Ingestion] ingested document {document.document_id} ({len(chunks)} chunks)")

        library.persist()
        return {
            "processed": processed,
            "skipped": skipped,
            "removed": removed,
        }

    def detect_structure(self, document: DocumentMetadata) -> StructureMetadata:
        if document.file_type in {"md", "markdown"} and document.text:
            section = self._detect_markdown_heading(document.text)
            return StructureMetadata(section=section)
        if document.file_type == "pdf" and document.text:
            page_number = self._detect_pdf_page_number(document.text)
            return StructureMetadata(page_number=page_number)
        return StructureMetadata()

    def _detect_markdown_heading(self, text: str) -> Optional[str]:
        for line in text.splitlines():
            if line.startswith("#"):
                return line.lstrip("# ").strip()
        return None

    def _detect_pdf_page_number(self, text: str) -> Optional[int]:
        match = re.search(r"page\s*(\d+)", text, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None
