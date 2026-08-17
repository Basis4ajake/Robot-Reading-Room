from __future__ import annotations

from fastapi import APIRouter, Depends

from ...providers.factory import OllamaUnavailableError, list_ollama_models
from ..dependencies import get_app_state
from ..schemas import ModelsResponse
from ..state import AppState

router = APIRouter(prefix="/api/v1/models", tags=["models"])


@router.get("", response_model=ModelsResponse)
def get_models(state: AppState = Depends(get_app_state)):
    if state.force_dummy:
        return ModelsResponse(available=False, models=[])
    try:
        models = list_ollama_models()
    except OllamaUnavailableError:
        return ModelsResponse(available=False, models=[])
    return ModelsResponse(available=True, models=models)
