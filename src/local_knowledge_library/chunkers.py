from __future__ import annotations

from typing import List

from .abstracts import Chunker
from .models import Chunk
from .models import make_chunk_id


class FixedSizeChunker(Chunker):
    def __init__(self, chunk_size: int = 200, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document) -> List[Chunk]:
        text = document.text or ""
        chunks: List[Chunk] = []
        start = 0
        offset = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            piece = text[start:end].strip()
            if piece:
                chunk = Chunk(
                    chunk_id=make_chunk_id(document.source_id, document.document_id, offset),
                    library_id=document.library_id,
                    source_id=document.source_id,
                    document_id=document.document_id,
                    text=piece,
                    metadata={
                        "offset": str(start),
                        "page_number": str(document.structure.page_number or ""),
                        "section": document.structure.section or "",
                    },
                    citation_id=f"cite-{document.document_id}-{offset}",
                )
                chunks.append(chunk)
                offset += 1
            start += self.chunk_size - self.overlap
        return chunks


class ParagraphChunker(Chunker):
    def chunk(self, document) -> List[Chunk]:
        text = document.text or ""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: List[Chunk] = []
        for index, paragraph in enumerate(paragraphs):
            chunks.append(Chunk(
                chunk_id=make_chunk_id(document.source_id, document.document_id, index),
                library_id=document.library_id,
                source_id=document.source_id,
                document_id=document.document_id,
                text=paragraph,
                metadata={
                    "paragraph": str(index + 1),
                    "page_number": str(document.structure.page_number or ""),
                    "section": document.structure.section or "",
                },
                citation_id=f"cite-{document.document_id}-{index}",
            ))
        return chunks
