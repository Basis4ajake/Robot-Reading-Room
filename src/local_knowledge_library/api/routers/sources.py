from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ...registry import LibraryNotFoundError
from ..dependencies import get_app_state
from ..schemas import IngestResponse, SourceAddRequest, SourceResponse
from ..state import AppState

router = APIRouter(prefix="/api/v1/libraries/{library_id}", tags=["sources"])


def _get_library(state: AppState, library_id: str):
    try:
        return state.registry.get_library(library_id)
    except LibraryNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Library {library_id} not found") from exc


@router.get("/sources", response_model=list[SourceResponse])
def list_sources(library_id: str, state: AppState = Depends(get_app_state)):
    library = _get_library(state, library_id)
    return [
        SourceResponse(
            source_id=source.source_id,
            source_path=source.source_path,
            filename=source.filename,
            file_type=source.file_type,
        )
        for source in library.list_sources()
    ]


@router.post("/sources", response_model=SourceResponse, status_code=201)
def add_source(library_id: str, payload: SourceAddRequest, state: AppState = Depends(get_app_state)):
    library = _get_library(state, library_id)
    try:
        source = library.add_source(payload.source_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return SourceResponse(
        source_id=source.source_id,
        source_path=source.source_path,
        filename=source.filename,
        file_type=source.file_type,
    )


@router.delete("/sources/{source_id}", status_code=204)
def remove_source(library_id: str, source_id: str, state: AppState = Depends(get_app_state)):
    library = _get_library(state, library_id)
    if source_id not in library.sources:
        raise HTTPException(status_code=404, detail=f"Source {source_id} not found")
    chunk_ids = [chunk.chunk_id for chunk in library.chunks.values() if chunk.source_id == source_id]
    runtime = state.get_runtime(library_id)
    library.remove_source(source_id)
    if chunk_ids:
        runtime.vector_store.remove(chunk_ids)


@router.post("/ingest", response_model=IngestResponse)
def ingest(library_id: str, state: AppState = Depends(get_app_state)):
    library = _get_library(state, library_id)
    runtime = state.get_runtime(library_id)
    result = runtime.pipeline.ingest(library)
    return IngestResponse(**result)
