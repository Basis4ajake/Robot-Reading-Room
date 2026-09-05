import sys
import types

from local_knowledge_library.models import LibraryConfig
from local_knowledge_library.providers.factory import build_providers
from local_knowledge_library.providers.ollama_providers import DummyEmbedder, DummyLLMProvider, OllamaQwenProvider


def _fake_ollama_module(*, embed_ok: bool, generate_ok: bool):
    module = types.SimpleNamespace()

    def fake_embed(model, input):
        if not embed_ok:
            raise RuntimeError("This server does not support embeddings")
        return types.SimpleNamespace(embeddings=[[0.1, 0.2, 0.3] for _ in input])

    def fake_generate(model, prompt, options):
        if not generate_ok:
            raise RuntimeError("model not found")
        return types.SimpleNamespace(response="ok")

    module.embed = fake_embed
    module.generate = fake_generate
    return module


def test_build_providers_returns_real_provider_when_embedding_works(monkeypatch):
    monkeypatch.setitem(sys.modules, "ollama", _fake_ollama_module(embed_ok=True, generate_ok=True))
    config = LibraryConfig(library_id="lib", name="Lib", embedding_model="nomic-embed-text")

    embedder, llm = build_providers(config)

    assert isinstance(embedder, OllamaQwenProvider)
    assert isinstance(llm, OllamaQwenProvider)


def test_build_providers_falls_back_to_dummy_embedder_and_warns(monkeypatch, capsys):
    # Reproduces the real bug: embedding_model resolves to a chat-only model
    # with no embedding head, so embed_text always fails but generate works.
    monkeypatch.setitem(sys.modules, "ollama", _fake_ollama_module(embed_ok=False, generate_ok=True))
    config = LibraryConfig(library_id="lib", name="Lib", embedding_model=None, llm_model="qwen2:1.5b")

    embedder, llm = build_providers(config)

    assert isinstance(embedder, DummyEmbedder)
    assert isinstance(llm, OllamaQwenProvider)
    warning = capsys.readouterr().out
    assert "DummyEmbedder" in warning
    assert "meaningless" in warning


def test_build_providers_falls_back_fully_when_ollama_totally_unavailable(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "ollama", _fake_ollama_module(embed_ok=False, generate_ok=False))
    config = LibraryConfig(library_id="lib", name="Lib", embedding_model="nomic-embed-text")

    embedder, llm = build_providers(config)

    assert isinstance(embedder, DummyEmbedder)
    assert isinstance(llm, DummyLLMProvider)
    assert "DummyEmbedder" in capsys.readouterr().out


def test_build_providers_force_dummy_skips_ollama_entirely():
    config = LibraryConfig(library_id="lib", name="Lib")

    embedder, llm = build_providers(config, force_dummy=True)

    assert isinstance(embedder, DummyEmbedder)
    assert isinstance(llm, DummyLLMProvider)


def test_library_config_defaults_embedding_model_to_a_real_model():
    # A null embedding_model makes OllamaQwenProvider fall back to the LLM's
    # own model name, which is exactly the trap that caused a real library's
    # index to silently fill with fake vectors.
    config = LibraryConfig(library_id="lib", name="Lib")
    assert config.embedding_model == "nomic-embed-text"
