from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ...registry import LibraryNotFoundError
from ...storage import KnowledgeLibrary
from ..dependencies import get_app_state
from ..schemas import LibraryCreateRequest, LibraryResponse, LibraryUpdateRequest
from ..state import AppState

router = APIRouter(prefix="/api/v1/libraries", tags=["libraries"])


def _to_response(library: KnowledgeLibrary) -> LibraryResponse:
    return LibraryResponse(
        library_id=library.metadata.library_id,
        name=library.metadata.name,
        description=library.metadata.description,
        source_count=library.metadata.source_count,
        document_count=library.metadata.document_count,
        chunk_count=library.metadata.chunk_count,
        chunk_size=library.config.chunk_size,
        chunk_overlap=library.config.chunk_overlap,
        top_k=library.config.top_k,
        llm_model=library.config.llm_model,
        embedding_model=library.config.embedding_model,
    )


@router.get("", response_model=list[LibraryResponse])
def list_libraries(state: AppState = Depends(get_app_state)):
    libraries = [state.registry.get_library(meta.library_id) for meta in state.registry.list_libraries()]
    return [_to_response(library) for library in libraries]


@router.post("", response_model=LibraryResponse, status_code=201)
def create_library(payload: LibraryCreateRequest, state: AppState = Depends(get_app_state)):
    try:
        library = state.registry.create_library(**payload.model_dump())
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _to_response(library)


@router.get("/{library_id}", response_model=LibraryResponse)
def get_library(library_id: str, state: AppState = Depends(get_app_state)):
    try:
        library = state.registry.get_library(library_id)
    except LibraryNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Library {library_id} not found") from exc
    return _to_response(library)


@router.patch("/{library_id}", response_model=LibraryResponse)
def update_library(library_id: str, payload: LibraryUpdateRequest, state: AppState = Depends(get_app_state)):
    try:
        library = state.registry.update_config(library_id, **payload.model_dump())
    except LibraryNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Library {library_id} not found") from exc
    state.invalidate(library_id)
    return _to_response(library)


@router.delete("/{library_id}", status_code=204)
def delete_library(library_id: str, confirm: bool = False, state: AppState = Depends(get_app_state)):
    if not confirm:
        raise HTTPException(status_code=400, detail="Pass confirm=true to permanently delete this library")
    try:
        state.registry.delete_library(library_id)
    except LibraryNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Library {library_id} not found") from exc
    state.invalidate(library_id)
