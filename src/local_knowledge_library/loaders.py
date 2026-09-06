from __future__ import annotations

from pathlib import Path
from typing import Iterable, List


from .abstracts import DocumentLoader
from .models import DocumentMetadata, StructureMetadata


class TextLoader(DocumentLoader):
    def supported_extensions(self) -> List[str]:
        return ["txt"]

    def load(self, source_path: str) -> Iterable[DocumentMetadata]:
        path = Path(source_path)
        text = path.read_text(encoding="utf-8")
        return [DocumentMetadata(
            source_id="",
            library_id="",
            document_id="",
            source_path=str(path.resolve()),
            filename=path.name,
            file_type="txt",
            title=path.stem,
            author=None,
            publisher=None,
            publication_date=None,
            edition=None,
            isbn=None,
            content_hash="",
            structure=StructureMetadata(),
            text=text,
        )]


class MarkdownLoader(DocumentLoader):
    def supported_extensions(self) -> List[str]:
        return ["md", "markdown"]

    def load(self, source_path: str) -> Iterable[DocumentMetadata]:
        path = Path(source_path)
        text = path.read_text(encoding="utf-8")
        return [DocumentMetadata(
            source_id="",
            library_id="",
            document_id="",
            source_path=str(path.resolve()),
            filename=path.name,
            file_type="markdown",
            title=path.stem,
            author=None,
            publisher=None,
            publication_date=None,
            edition=None,
            isbn=None,
            content_hash="",
            structure=StructureMetadata(),
            text=text,
        )]


class PdfLoader(DocumentLoader):
    def supported_extensions(self) -> List[str]:
        return ["pdf"]

    def load(self, source_path: str) -> Iterable[DocumentMetadata]:
        path = Path(source_path)
        try:
            from pypdf import PdfReader
        except Exception:
            raise RuntimeError("pypdf is required to load PDF files. Install it or use a non-PDF source.")
        reader = PdfReader(str(path))
        text_parts = []
        page_boundaries = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
            # Mirrors ParagraphChunker's own split/filter exactly, so summing
            # these counts up to a given chunk index tells you which page
            # that chunk came from. Splitting a "\n\n"-joined string on
            # "\n\n" is equivalent to concatenating each part's own split
            # (occurrences of the separator are found the same way either
            # way), so this per-page count lines up with the chunks
            # ParagraphChunker will later produce from the joined raw_text.
            page_boundaries.append(len([p for p in page_text.split("\n\n") if p.strip()]))
        raw_text = "\n\n".join(text_parts)
        return [DocumentMetadata(
            source_id="",
            library_id="",
            document_id="",
            source_path=str(path.resolve()),
            filename=path.name,
            file_type="pdf",
            title=path.stem,
            author=None,
            publisher=None,
            publication_date=None,
            edition=None,
            isbn=None,
            content_hash="",
            structure=StructureMetadata(),
            text=raw_text,
            page_boundaries=page_boundaries,
        )]
