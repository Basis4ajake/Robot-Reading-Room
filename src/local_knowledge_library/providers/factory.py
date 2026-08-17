from __future__ import annotations

from typing import Dict, List, Tuple

from ..abstracts import Embedder, LLMProvider
from ..models import LibraryConfig
from .ollama_providers import DummyEmbedder, DummyLLMProvider, OllamaQwenProvider


class OllamaUnavailableError(RuntimeError):
    pass


def build_providers(config: LibraryConfig, force_dummy: bool = False) -> Tuple[Embedder, LLMProvider]:
    if force_dummy:
        return DummyEmbedder(), DummyLLMProvider()

    provider = OllamaQwenProvider(model_name=config.llm_model, embedding_model=config.embedding_model)
    try:
        provider.embed_text(["ping"])
    except Exception:
        try:
            provider.generate("hello")
            return DummyEmbedder(), provider
        except Exception:
            return DummyEmbedder(), DummyLLMProvider()
    return provider, provider


def list_ollama_models() -> List[Dict]:
    try:
        import ollama
    except ImportError as exc:
        raise OllamaUnavailableError("Ollama SDK is not installed") from exc
    try:
        response = ollama.list()
    except Exception as exc:
        raise OllamaUnavailableError("Could not reach the local Ollama daemon") from exc

    models = getattr(response, "models", None)
    if models is None and isinstance(response, dict):
        models = response.get("models", [])
    models = models or []

    result = []
    for item in models:
        name = getattr(item, "model", None) or (item.get("model") if isinstance(item, dict) else None)
        size = getattr(item, "size", None) or (item.get("size") if isinstance(item, dict) else None)
        details = getattr(item, "details", None) or (item.get("details") if isinstance(item, dict) else None)
        quantization = None
        if details is not None:
            quantization = getattr(details, "quantization_level", None) or (
                details.get("quantization_level") if isinstance(details, dict) else None
            )
        result.append({"name": name, "size": size, "quantization": quantization})
    return result
