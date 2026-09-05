import sys
import types

from local_knowledge_library.providers import OllamaQwenProvider


def test_ollama_qwen_provider_generate_and_embed(monkeypatch):
    ollama_module = types.SimpleNamespace()

    def fake_generate(model: str, prompt: str, options: dict):
        assert model == "qwen2:1.5b"
        assert prompt == "Hello"
        assert options == {"num_predict": 16}
        return types.SimpleNamespace(response="generated response")

    def fake_embed(model: str, input):
        assert model == "qwen2:1.5b"
        assert input == ["Hello"]
        return types.SimpleNamespace(embeddings=[[0.1, 0.2, 0.3]])

    ollama_module.generate = fake_generate
    ollama_module.embed = fake_embed
    monkeypatch.setitem(sys.modules, "ollama", ollama_module)

    provider = OllamaQwenProvider()
    assert provider.generate("Hello", max_tokens=16) == "generated response"
    assert provider.embed_text(["Hello"]) == [[0.1, 0.2, 0.3]]
