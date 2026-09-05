from __future__ import annotations

import bisect
from typing import List, Optional, Sequence

from .abstracts import Chunker
from .models import Chunk
from .models import make_chunk_id


def _page_lookup(page_boundaries: Optional[Sequence[int]]):
    """Build a paragraph-index -> 1-indexed page number lookup from per-page
    paragraph counts, or return None if there's nothing to look up (e.g. a
    non-PDF document, or a loader that doesn't populate page_boundaries)."""
    if not page_boundaries:
        return None
    cumulative: List[int] = []
    total = 0
    for count in page_boundaries:
        total += count
        cumulative.append(total)
    if total == 0:
        return None

    def lookup(paragraph_index: int) -> int:
        page = bisect.bisect_right(cumulative, paragraph_index) + 1
        return min(page, len(cumulative))

    return lookup


class FixedSizeChunker(Chunker):
    """Unused (nothing wires this in). Also: chunk_size/overlap here are
    CHARACTER counts, but the GUI's tooltip for those fields says "measured
    in words" - ParagraphChunker below is the word-based one that actually
    honors that contract. Left as-is rather than fixed, since fixing an
    unused class's behavior with no caller to verify it against isn't worth
    the risk of a silent latent bug; don't wire this in without addressing
    the unit mismatch first.

    Also note: this stores overlap as `self.overlap`, not `self.chunk_overlap`
    - ingestion.py's `_chunker_signature` reads `chunk_overlap` via getattr,
    so wiring this in as-is would make an overlap-only config change on this
    chunker silently invisible to the force-reprocess mechanism, reproducing
    the exact stale-index bug class that mechanism exists to close. Rename
    the attribute (or teach _chunker_signature both names) if this is ever
    wired in.
    """

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
    """Splits on paragraph breaks (for PDFs, effectively page breaks - see
    PdfLoader), then sub-splits any paragraph longer than chunk_size words
    into overlapping word windows. chunk_size/chunk_overlap are word counts,
    matching the GUI's tooltip contract. chunk_size=None (the default)
    preserves the old behavior: one chunk per paragraph, no splitting.

    Overlap only ever applies WITHIN a paragraph/page, never across one -
    a chunk straddling a page boundary would have no single correct page
    number to cite, and correct page citations are worth more here than
    seamless cross-page continuity. See docs/review-repository-propose-
    enhancements-snug-sloth.md for the tradeoff writeup.
    """

    def __init__(self, chunk_size: Optional[int] = None, chunk_overlap: int = 0):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document) -> List[Chunk]:
        text = document.text or ""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        # When we know per-page paragraph counts (PDFs), attribute each chunk
        # to the page it actually came from instead of the single page number
        # detect_structure found via one regex search over the whole document.
        page_boundaries = getattr(document, "page_boundaries", None)
        if page_boundaries and sum(page_boundaries) != len(paragraphs):
            # This invariant (PdfLoader's per-page split exactly mirroring
            # this method's own split) is only enforced by the two staying in
            # sync by inspection, not by any shared code or runtime check. A
            # future edit to either could silently desync them - _page_lookup
            # clamps any out-of-range index to the last page rather than
            # erroring, so a desync would otherwise misattribute citations to
            # the wrong page with no indication anything was wrong.
            print(
                f"[Chunker] WARNING: page_boundaries sum ({sum(page_boundaries)}) != "
                f"paragraph count ({len(paragraphs)}) for document {document.document_id!r} - "
                "page citations for this document may be misattributed."
            )
        lookup = _page_lookup(page_boundaries)
        chunks: List[Chunk] = []
        index = 0
        for paragraph_index, paragraph in enumerate(paragraphs):
            page_number = lookup(paragraph_index) if lookup else document.structure.page_number
            for piece in self._split_to_size(paragraph):
                chunks.append(Chunk(
                    chunk_id=make_chunk_id(document.source_id, document.document_id, index),
                    library_id=document.library_id,
                    source_id=document.source_id,
                    document_id=document.document_id,
                    text=piece,
                    metadata={
                        "paragraph": str(paragraph_index + 1),
                        "page_number": str(page_number or ""),
                        "section": document.structure.section or "",
                    },
                    citation_id=f"cite-{document.document_id}-{index}",
                ))
                index += 1
        return chunks

    def _split_to_size(self, paragraph: str) -> List[str]:
        if not self.chunk_size or self.chunk_size <= 0:
            return [paragraph]
        words = paragraph.split()
        if len(words) <= self.chunk_size:
            return [paragraph]
        # Nothing validates chunk_overlap upstream (API schema, GUI, direct
        # construction), so clamp defensively: negative overlap would make
        # step overshoot chunk_size and silently skip whole spans of text
        # (never embedded, never retrievable, no error anywhere). Capping at
        # chunk_size - 1 alone isn't enough either - that still allows
        # step=1, exploding a paragraph into thousands of near-duplicate
        # one-word-shifted chunks. Cap at half of chunk_size instead: real
        # overlap is meant to preserve continuity across a boundary, not
        # approach a full duplicate of the previous chunk.
        effective_overlap = max(0, min(self.chunk_overlap, self.chunk_size // 2))
        step = self.chunk_size - effective_overlap
        pieces: List[str] = []
        start = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            pieces.append(" ".join(words[start:end]))
            if end >= len(words):
                break
            start += step
        return pieces
