from __future__ import annotations

from fastapi import APIRouter, Depends

from ...providers.factory import OllamaUnavailableError, list_ollama_models
from ..dependencies import get_app_state
from ..schemas import HealthResponse
from ..state import AppState

router = APIRouter(prefix="/api/v1", tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(state: AppState = Depends(get_app_state)):
    ollama_available = False
    if not state.force_dummy:
        try:
            list_ollama_models()
            ollama_available = True
        except OllamaUnavailableError:
            ollama_available = False
    return HealthResponse(status="ok", data_dir=str(state.registry.data_dir), ollama_available=ollama_available)
