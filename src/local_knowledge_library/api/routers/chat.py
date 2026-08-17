from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ...registry import LibraryNotFoundError
from ..dependencies import get_app_state
from ..schemas import ChatRequest, ChatResponse
from ..state import AppState

router = APIRouter(prefix="/api/v1/libraries/{library_id}", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(library_id: str, payload: ChatRequest, state: AppState = Depends(get_app_state)):
    try:
        library = state.registry.get_library(library_id)
    except LibraryNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Library {library_id} not found") from exc

    runtime = state.get_runtime(library_id)
    top_k = payload.top_k or library.config.top_k
    result = runtime.qa.answer_query(payload.query, library, top_k=top_k)
    return ChatResponse(**result)
