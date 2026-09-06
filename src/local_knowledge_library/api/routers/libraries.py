from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from ...providers.factory import OllamaUnavailableError, list_ollama_models
from ...registry import LibraryNotFoundError
from ...storage import KnowledgeLibrary
from ..dependencies import get_app_state
from ..schemas import LibraryCreateRequest, LibraryResponse, LibraryUpdateRequest
from ..state import AppState, LibraryBusyError

router = APIRouter(prefix="/api/v1/libraries", tags=["libraries"])


def _validate_embedding_model(state: AppState, embedding_model: Optional[str]) -> None:
    """Reject an embedding_model that isn't among locally pulled Ollama
    models, so a typo in the GUI's free-text field (or a direct API call)
    fails fast at save time. This is strictly earlier, not a replacement
    for build_providers()'s own loud DummyEmbedder-fallback warning - that
    still covers a model that gets removed/unpulled after being saved.

    None means "leave unchanged" (PATCH) or "use the default" (POST with
    an explicit null) - registry.update_config()/LibraryConfig already
    handle those cases without validation, same as before. Skipped
    entirely when Ollama isn't reachable (force_dummy, or a real
    OllamaUnavailableError) - can't validate against a daemon that isn't
    there, and blocking config changes because Ollama happens to be down
    would be a worse failure mode than the typo this is meant to catch.

    Real `ollama.list()` returns fully-qualified names (confirmed against
    the real local daemon: an untagged pull like `nomic-embed-text` comes
    back as `nomic-embed-text:latest`, while `ollama pull model:tag`
    keeps its explicit tag, e.g. `qwen2:1.5b`) - so an exact-match-only
    check would reject this project's own untagged default the moment a
    real daemon is reachable. Accepts either the exact name or `name +
    ":latest"`, but does NOT strip/ignore tags generally - a caller
    asking for `qwen3:4b` when only `qwen3:8b` is pulled must still be
    rejected, since those are genuinely different models.
    """
    if not embedding_model or state.force_dummy:
        return
    try:
        available = {model["name"] for model in list_ollama_models() if model["name"]}
    except OllamaUnavailableError:
        return
    if embedding_model not in available and f"{embedding_model}:latest" not in available:
        raise HTTPException(
            status_code=400,
            detail=(
                f"'{embedding_model}' is not a locally pulled Ollama model. "
                f"Pulled models: {', '.join(sorted(available)) or '(none)'}"
            ),
        )


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
        enable_recipe_extraction=library.config.enable_recipe_extraction,
    )


@router.get("", response_model=list[LibraryResponse])
def list_libraries(state: AppState = Depends(get_app_state)):
    libraries = [state.registry.get_library(meta.library_id) for meta in state.registry.list_libraries()]
    return [_to_response(library) for library in libraries]


@router.post("", response_model=LibraryResponse, status_code=201)
def create_library(payload: LibraryCreateRequest, state: AppState = Depends(get_app_state)):
    _validate_embedding_model(state, payload.embedding_model)
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
    _validate_embedding_model(state, payload.embedding_model)
    try:
        with state.exclusive(library_id):
            try:
                library = state.registry.update_config(library_id, **payload.model_dump())
            except LibraryNotFoundError as exc:
                raise HTTPException(status_code=404, detail=f"Library {library_id} not found") from exc
            state.invalidate(library_id)
    except LibraryBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _to_response(library)


@router.delete("/{library_id}", status_code=204)
def delete_library(library_id: str, confirm: bool = False, state: AppState = Depends(get_app_state)):
    if not confirm:
        raise HTTPException(status_code=400, detail="Pass confirm=true to permanently delete this library")
    try:
        with state.exclusive(library_id):
            try:
                state.registry.delete_library(library_id)
            except LibraryNotFoundError as exc:
                raise HTTPException(status_code=404, detail=f"Library {library_id} not found") from exc
            state.invalidate(library_id)
    except LibraryBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
