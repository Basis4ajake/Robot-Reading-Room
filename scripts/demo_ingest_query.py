#!/usr/bin/env python3
"""Demo: ingest a single file into a KnowledgeLibrary and run a query.

This script defaults to using dummy providers when `--force-dummy` is supplied
so the demo can run without Ollama or other external services.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


# Ensure `src` is on path for local imports when running from repo root
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from local_knowledge_library.models import LibraryConfig
from local_knowledge_library.storage import KnowledgeLibrary
from local_knowledge_library.loaders import TextLoader, MarkdownLoader, PdfLoader
from local_knowledge_library.chunkers import ParagraphChunker
from local_knowledge_library.ingestion import IngestionPipeline
from local_knowledge_library.retrieval import Retriever
from local_knowledge_library.query_planner import QueryPlanner
from local_knowledge_library.qa import GroundedQA
from local_knowledge_library.providers import (
    OllamaQwenProvider,
    DummyEmbedder,
    DummyLLMProvider,
    SqliteVectorStore,
)


def choose_embedder_and_llm(force_dummy: bool):
    if force_dummy:
        return DummyEmbedder(), DummyLLMProvider()

    try:
        # Try Ollama for both LLM and embeddings
        ollama_provider = OllamaQwenProvider()
        # quick check: try embedding a tiny string to validate embedding support
        try:
            _ = ollama_provider.embed_text(["ping"])
        except Exception:
            # fall back to dummy embedder while keeping Ollama as LLM if generate works
            try:
                _ = ollama_provider.generate("hello")
                return DummyEmbedder(), ollama_provider
            except Exception:
                return DummyEmbedder(), DummyLLMProvider()
        return ollama_provider, ollama_provider
    except Exception:
        return DummyEmbedder(), DummyLLMProvider()


def ensure_library(config: LibraryConfig) -> KnowledgeLibrary:
    try:
        return KnowledgeLibrary.open(config)
    except FileNotFoundError:
        return KnowledgeLibrary.create(config)


def main():
    parser = argparse.ArgumentParser(description="Demo ingest + query for Local Knowledge Library")
    parser.add_argument("--source", required=True, help="Path to a file to ingest (txt/md/pdf)")
    parser.add_argument("--library-id", default="demo-library", help="Library id to use/create")
    parser.add_argument("--query", default="Summarize the document.", help="Query to ask after ingest")
    parser.add_argument("--force-dummy", action="store_true", help="Force dummy providers for offline demo")
    args = parser.parse_args()

    config = LibraryConfig(library_id=args.library_id, name="Demo Library", data_dir="./data/libraries")
    library = ensure_library(config)

    print(f"Using library at: {library.library_dir}")

    # Add source
    source_path = Path(args.source).resolve()
    print(f"Adding source: {source_path}")
    source = library.add_source(str(source_path))

    # Choose providers
    embedder, llm_provider = choose_embedder_and_llm(args.force_dummy)

    # Vector DB path (inside library index dir)
    vector_db = library.index_dir / "vectors.db"

    # Build pipeline
    loaders = [TextLoader(), MarkdownLoader(), PdfLoader()]
    chunker = ParagraphChunker()

    with SqliteVectorStore(str(vector_db)) as vector_store:
        pipeline = IngestionPipeline(loaders, chunker, embedder, vector_store, debug=True)
        results = pipeline.ingest(library)
        print("Ingestion results:", results)

        # Retrieval + QA
        retriever = Retriever(vector_store=vector_store, embedder=embedder, debug=True)
        planner = QueryPlanner()
        qa = GroundedQA(retriever, planner, llm_provider, debug=True)
        answer = qa.answer_query(args.query, library, top_k=config.top_k)

        print("\n=== Answer ===")
        print(answer["answer"])
        print("\n=== Citations ===")
        for c in answer["citations"]:
            print(c)


if __name__ == "__main__":
    main()
