from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .abstracts import Chunker, DocumentLoader, Embedder, LLMProvider, VectorStore
from .models import (
    Chunk,
    DocumentMetadata,
    LibraryConfig,
    SourceMetadata,
    compute_content_hash,
    compute_path_hash,
    make_document_id,
    StructureMetadata,
)
from .providers.ollama_providers import DummyEmbedder
from .recipe_extraction import extract_recipe_facts, segment_recipes, to_recipe_fact
from .storage import KnowledgeLibrary


def _embedder_signature(embedder: Embedder) -> str:
    """Identify which model actually produced embeddings, not which was configured.

    Falls back to the class name (e.g. "DummyEmbedder") when a provider has no
    named model, so a silent dummy-embedder fallback is itself a distinct
    signature and triggers the same re-embed path as an explicit model switch.
    """
    return getattr(embedder, "embedding_model", None) or getattr(embedder, "model_name", None) or type(embedder).__name__


def _chunker_signature(chunker: Chunker) -> str:
    """Identify how documents were actually split. A chunk_size/overlap
    change needs the same force-reprocess treatment as an embedding model
    change, or hash-based incremental ingestion would silently ignore it
    forever (the source/document content didn't change, only how it gets cut
    up) - the exact same failure shape _embedder_signature exists to catch.
    """
    chunk_size = getattr(chunker, "chunk_size", None)
    chunk_overlap = getattr(chunker, "chunk_overlap", None)
    return f"{type(chunker).__name__}:{chunk_size}:{chunk_overlap}"


def _recipe_extraction_signature(enabled: bool) -> str:
    """Same idea as _embedder_signature/_chunker_signature: toggling
    enable_recipe_extraction doesn't change source content, so it needs its
    own signature or hash-based incremental ingestion would silently ignore
    the change forever. Versioned so a future change to the extraction logic
    itself can also force reprocessing by bumping this string.
    """
    return "recipe_extraction:v1" if enabled else "disabled"


class IngestionPipeline:
    def __init__(
        self,
        loaders: Iterable[DocumentLoader],
        chunker: Chunker,
        embedder: Embedder,
        vector_store: VectorStore,
        debug: bool = False,
        llm_provider: Optional[LLMProvider] = None,
    ):
        self.loaders = {ext: loader for loader in loaders for ext in loader.supported_extensions()}
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.debug = debug
        # Only needed for recipe extraction (LibraryConfig.enable_recipe_extraction) -
        # every other pipeline stage uses embedder/vector_store only. Optional so
        # existing call sites that don't use that feature are unaffected.
        self.llm_provider = llm_provider

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

        current_embedding_signature = _embedder_signature(self.embedder)
        current_chunking_signature = _chunker_signature(self.chunker)
        current_recipe_extraction_signature = _recipe_extraction_signature(library.config.enable_recipe_extraction)

        # A transient Ollama outage (or any other embed failure) makes
        # build_providers() hand us a DummyEmbedder for THIS runtime - not a
        # deliberate model change. Treating that the same as a real model
        # switch would force-reprocess the whole library below and overwrite
        # good vectors with fake ones: actively destructive, not just
        # unhelpful, and worse than the pre-signature-tracking behavior this
        # was meant to improve on. Refuse outright instead.
        was_previously_real = library.state.embedding_signature not in (None, DummyEmbedder.__name__)
        if isinstance(self.embedder, DummyEmbedder) and was_previously_real:
            raise RuntimeError(
                "Embedding is currently falling back to DummyEmbedder (Ollama unavailable, or "
                f"the configured embedding model can't produce embeddings), but this library was "
                f"previously indexed with a real model ({library.state.embedding_signature!r}). "
                "Refusing to re-embed with fake vectors - fix Ollama/the embedding model and retry."
            )

        embedding_changed = (
            library.state.embedding_signature is not None
            and library.state.embedding_signature != current_embedding_signature
        )
        # A library indexed before this tracking existed has no recorded
        # signature at all; if it already has documents, treat that as
        # unverified rather than assume it matches, since it may be exactly
        # the case this is meant to catch (e.g. a prior silent DummyEmbedder
        # fallback that produced fake 16-dim vectors, as later confirmed).
        embedding_unverified = library.state.embedding_signature is None and bool(library.state.documents)
        chunking_changed = (
            library.state.chunking_signature is not None
            and library.state.chunking_signature != current_chunking_signature
        )
        chunking_unverified = library.state.chunking_signature is None and bool(library.state.documents)
        # Unlike embedding/chunking, "no recorded signature" here is the normal
        # state for every library that has never touched this opt-in feature -
        # NOT an unverified/unsafe state - so this only forces reprocessing on
        # an actual change (including the initial off -> on transition), never
        # merely because the library predates this tracking.
        recipe_extraction_changed = (
            (library.state.recipe_extraction_signature or "disabled") != current_recipe_extraction_signature
        )
        force_reprocess = (
            embedding_changed
            or embedding_unverified
            or chunking_changed
            or chunking_unverified
            or recipe_extraction_changed
        )

        if force_reprocess and self.debug:
            print(
                "[Ingestion] embedding/chunking/recipe-extraction config changed or unverified "
                f"(embedding: {library.state.embedding_signature!r} -> {current_embedding_signature!r}, "
                f"chunking: {library.state.chunking_signature!r} -> {current_chunking_signature!r}, "
                f"recipe_extraction: {library.state.recipe_extraction_signature!r} -> {current_recipe_extraction_signature!r}); "
                "forcing full re-embed/re-chunk"
            )

        for source in library.list_sources():
            if not Path(source.source_path).exists():
                removed.append(source.source_id)
                library.remove_source(source.source_id)
                continue

            current_source_hash = compute_path_hash(source.source_path)
            previous_source_hash = library.state.sources.get(source.source_path)
            if previous_source_hash == current_source_hash and not force_reprocess:
                if self.debug:
                    print(f"[Ingestion] no changes detected for source {source.source_path}")
                skipped.extend(doc.document_id for doc in library.find_documents_by_source(source.source_id))
                continue

            loader = self.get_loader_for_source(source.source_path)
            existing_docs = {doc.document_id for doc in library.find_documents_by_source(source.source_id)}
            seen_docs: set[str] = set()
            for document in loader.load(source.source_path):
                document.source_id = source.source_id
                document.library_id = library.config.library_id
                document.source_path = source.source_path
                document.document_id = make_document_id(source.source_id, document.filename)
                document.content_hash = compute_content_hash(document.text or "")
                document.structure = self.detect_structure(document)
                seen_docs.add(document.document_id)
                previous_hash = library.state.documents.get(document.document_id)
                if previous_hash == document.content_hash and not force_reprocess:
                    skipped.append(document.document_id)
                    continue

                # Explicitly remove whatever chunks this document had before
                # adding its newly (re-)chunked set, rather than relying on
                # SQLite REPLACE-by-id alone - a chunk_size/overlap change can
                # produce a different chunk count than before, so keeping
                # previous_hash available here (instead of wiping ingestion
                # state wholesale) is what lets stale higher- or lower-index
                # chunk rows actually get cleaned up instead of orphaned.
                if previous_hash is not None:
                    old_chunk_ids = [chunk.chunk_id for chunk in library.find_chunks_by_document(document.document_id)]
                    library.remove_document(document.document_id)
                    if hasattr(self.vector_store, "remove"):
                        self.vector_store.remove(old_chunk_ids)

                chunks = self.chunker.chunk(document)
                embeddings = self.embedder.embed_text([chunk.text for chunk in chunks])
                self.vector_store.add(chunks, embeddings)
                library.register_document(document)
                for chunk in chunks:
                    library.register_chunk(chunk)

                if library.config.enable_recipe_extraction and self.llm_provider is not None:
                    facts = self._extract_recipe_facts(document)
                    library.register_recipe_facts(document.document_id, facts)
                    if self.debug:
                        print(f"[Ingestion] extracted {len(facts)} recipe(s) from {document.document_id}")
                elif library.config.enable_recipe_extraction:
                    library.register_recipe_facts(document.document_id, [])
                    if self.debug:
                        print(
                            f"[Ingestion] recipe extraction enabled but no llm_provider configured - "
                            f"skipping for {document.document_id}"
                        )

                processed.append(document.document_id)
                if self.debug:
                    print(f"[Ingestion] ingested document {document.document_id} ({len(chunks)} chunks)")

            stale_doc_ids = existing_docs - seen_docs
            for stale_doc_id in stale_doc_ids:
                old_chunk_ids = [chunk.chunk_id for chunk in library.find_chunks_by_document(stale_doc_id)]
                library.remove_document(stale_doc_id)
                if hasattr(self.vector_store, "remove"):
                    self.vector_store.remove(old_chunk_ids)

            library.sources[source.source_id].content_hash = current_source_hash
            library.state.sources[source.source_path] = current_source_hash

        library.state.embedding_signature = current_embedding_signature
        library.state.chunking_signature = current_chunking_signature
        library.state.recipe_extraction_signature = current_recipe_extraction_signature
        library.persist()
        return {
            "processed": processed,
            "skipped": skipped,
            "removed": removed,
        }

    def _extract_recipe_facts(self, document: DocumentMetadata) -> List:
        segments = segment_recipes(document.text or "")
        facts = []
        for segment in segments:
            extracted = extract_recipe_facts(segment, self.llm_provider)
            if extracted is None:
                continue
            facts.append(
                to_recipe_fact(
                    segment,
                    extracted,
                    library_id=document.library_id,
                    source_id=document.source_id,
                    document_id=document.document_id,
                    page_number=document.structure.page_number,
                )
            )
        return facts

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
