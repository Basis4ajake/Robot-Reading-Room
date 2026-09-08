from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ...evaluation import run_evaluation
from ...models import EvalRun
from ...registry import LibraryNotFoundError
from ..dependencies import get_app_state
from ..schemas import (
    EvalCaseCreateRequest,
    EvalCaseResponse,
    EvalResultResponse,
    EvalRunResponse,
)
from ..state import AppState, LibraryBusyError

router = APIRouter(prefix="/api/v1/libraries/{library_id}", tags=["eval"])


def _get_library(state: AppState, library_id: str):
    try:
        return state.registry.get_library(library_id)
    except LibraryNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Library {library_id} not found") from exc


def _eval_run_response(run: EvalRun) -> EvalRunResponse:
    return EvalRunResponse(
        eval_run_id=run.eval_run_id,
        timestamp=run.timestamp,
        chunk_size=run.chunk_size,
        chunk_overlap=run.chunk_overlap,
        top_k=run.top_k,
        llm_model=run.llm_model,
        embedding_model=run.embedding_model,
        passed_count=run.passed_count,
        total_count=run.total_count,
        results=[EvalResultResponse(**result.to_dict()) for result in run.results],
    )


@router.get("/eval-cases", response_model=list[EvalCaseResponse])
def list_eval_cases(library_id: str, state: AppState = Depends(get_app_state)):
    library = _get_library(state, library_id)
    return [
        EvalCaseResponse(eval_case_id=case.eval_case_id, question=case.question, expected_keyword=case.expected_keyword)
        for case in library.list_eval_cases()
    ]


@router.post("/eval-cases", response_model=EvalCaseResponse, status_code=201)
def add_eval_case(library_id: str, payload: EvalCaseCreateRequest, state: AppState = Depends(get_app_state)):
    library = _get_library(state, library_id)
    case = library.add_eval_case(payload.question, payload.expected_keyword)
    return EvalCaseResponse(eval_case_id=case.eval_case_id, question=case.question, expected_keyword=case.expected_keyword)


@router.delete("/eval-cases/{eval_case_id}", status_code=204)
def remove_eval_case(library_id: str, eval_case_id: str, state: AppState = Depends(get_app_state)):
    library = _get_library(state, library_id)
    if eval_case_id not in library.eval_cases:
        raise HTTPException(status_code=404, detail=f"Eval case {eval_case_id} not found")
    library.remove_eval_case(eval_case_id)


@router.post("/evaluate", response_model=EvalRunResponse)
def evaluate(library_id: str, state: AppState = Depends(get_app_state)):
    library = _get_library(state, library_id)
    try:
        # Held for the whole run, not per-case like chat holds it for one
        # question - a run over several cases can take minutes on this
        # project's CPU-only hardware, and the run needs the config it
        # stamps on the EvalRun to actually be stable for its full duration
        # (a config PATCH mid-run would make the recorded chunk_size/top_k/
        # model a lie about what some of the cases actually ran under).
        with state.use_runtime(library_id) as runtime:
            run = run_evaluation(library, runtime.qa)
    except LibraryBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    library.register_eval_run(run)
    return _eval_run_response(run)


@router.get("/eval-runs", response_model=list[EvalRunResponse])
def list_eval_runs(library_id: str, state: AppState = Depends(get_app_state)):
    library = _get_library(state, library_id)
    return [_eval_run_response(run) for run in library.list_eval_runs()]
