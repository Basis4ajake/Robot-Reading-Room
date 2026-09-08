"""Retrieval/answer regression harness, run on demand against a real library.

This is the GUI-facing generalization of scripts/eval_retrieval.py's fixed
geography-corpus check: instead of one hardcoded question set against a
synthetic corpus, a library owner defines their own (question, expected
keyword) cases and runs them against that library's real content, through
the real chat pipeline (GroundedQA.answer_query) - the same retrieval,
reranking, and answer generation chat uses, not a separate code path.
"""
from __future__ import annotations

from datetime import datetime, timezone

from .models import EvalCase, EvalResult, EvalRun, make_id
from .qa import GroundedQA
from .storage import KnowledgeLibrary


def _run_one_case(qa: GroundedQA, library: KnowledgeLibrary, case: EvalCase) -> EvalResult:
    result = qa.answer_query(case.question, library, top_k=library.config.top_k)
    keyword = case.expected_keyword.strip().lower()

    chunk_text = " ".join(chunk.get("text", "") for chunk in result["chunks"]).lower()
    answer_text = result["answer"].lower()

    return EvalResult(
        eval_case_id=case.eval_case_id,
        question=case.question,
        expected_keyword=case.expected_keyword,
        keyword_in_citations=keyword in chunk_text,
        keyword_in_answer=keyword in answer_text,
        answer=result["answer"],
        answer_source=result["answer_source"],
        citation_count=len(result["citations"]),
    )


def run_evaluation(library: KnowledgeLibrary, qa: GroundedQA) -> EvalRun:
    """Runs every eval case currently stored for `library` and returns the
    resulting EvalRun. Does not persist it - callers append it to the
    library themselves (see KnowledgeLibrary.register_eval_run) so this
    function stays a pure "run and report," like recipe_extraction's
    extraction functions.
    """
    config = library.config
    results = [_run_one_case(qa, library, case) for case in library.list_eval_cases()]

    return EvalRun(
        eval_run_id=make_id("eval-run"),
        library_id=config.library_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        top_k=config.top_k,
        llm_model=config.llm_model,
        embedding_model=config.embedding_model,
        results=results,
    )
