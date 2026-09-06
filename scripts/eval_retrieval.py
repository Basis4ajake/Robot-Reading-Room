#!/usr/bin/env python3
"""Retrieval regression check: does semantic search still surface the right
chunk for a fixed set of unambiguous questions?

This is the pathway that has silently broken three times this session
already (fake vectors, wrong page citations, an inert chunk_size setting) -
each time by hand, after the fact, against real library data. This script
is the cheap, repeatable version: run it after touching chunking, embedding
config, or retrieval code, before trusting the change.

Needs a live Ollama (real embeddings) to mean anything - DummyEmbedder
produces vectors with no real semantic content, so a pass/fail against it
would be meaningless. Refuses to run rather than report a false result if
Ollama isn't reachable.

Usage:
    python scripts/eval_retrieval.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from local_knowledge_library.chunkers import ParagraphChunker
from local_knowledge_library.ingestion import IngestionPipeline
from local_knowledge_library.loaders import TextLoader
from local_knowledge_library.models import LibraryConfig
from local_knowledge_library.providers import SqliteVectorStore
from local_knowledge_library.providers.factory import build_providers
from local_knowledge_library.providers.ollama_providers import DummyEmbedder
from local_knowledge_library.retrieval import Retriever
from local_knowledge_library.storage import KnowledgeLibrary

CORPUS_PATH = Path(__file__).with_name("retrieval_eval_corpus.txt")

# (question, keyword expected somewhere in the top-k retrieved chunks' text)
# Each question maps to exactly one fact in retrieval_eval_corpus.txt, chosen
# to be unambiguous - a passing check means real semantic retrieval, not luck.
CASES = [
    ("What is the longest river in the world?", "nile"),
    ("Which mountain is the highest in Africa?", "kilimanjaro"),
    ("What is the largest hot desert?", "sahara"),
    ("How much oxygen does the Amazon rainforest produce?", "amazon"),
    ("What is the longest continental mountain range?", "andes"),
    ("What is the deepest freshwater lake?", "baikal"),
    ("What coral reef system is visible from outer space?", "great barrier reef"),
]


def main() -> int:
    data_dir = Path(tempfile.mkdtemp(prefix="lkl-retrieval-eval-"))
    try:
        config = LibraryConfig(library_id="retrieval-eval", name="Retrieval Eval", data_dir=str(data_dir))
        library = KnowledgeLibrary.create(config)
        library.add_source(str(CORPUS_PATH))

        embedder, _ = build_providers(config)
        if isinstance(embedder, DummyEmbedder):
            print(
                "Ollama isn't reachable (or the embedding model failed), so this would only "
                "be testing DummyEmbedder's fake vectors - refusing to run rather than report "
                "a meaningless result. Start Ollama and pull nomic-embed-text, then retry."
            )
            return 2

        with SqliteVectorStore(str(library.index_dir / "vectors.db")) as vector_store:
            chunker = ParagraphChunker(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
            pipeline = IngestionPipeline([TextLoader()], chunker, embedder, vector_store)
            pipeline.ingest(library)

            retriever = Retriever(vector_store=vector_store, embedder=embedder)

            failures = []
            for question, expected_keyword in CASES:
                chunks = retriever.semantic_search(question, top_k=3)
                found = any(expected_keyword in chunk.text.lower() for chunk in chunks)
                status = "PASS" if found else "FAIL"
                print(f"[{status}] {question!r} -> expected {expected_keyword!r} in top-3")
                if not found:
                    failures.append(question)
                    for chunk in chunks:
                        print(f"    retrieved: {chunk.text.strip()[:100]}...")

        print(f"\n{len(CASES) - len(failures)}/{len(CASES)} passed")
        return 1 if failures else 0
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
