import sys
import types

from local_knowledge_library.providers import OllamaQwenProvider
from local_knowledge_library.providers.ollama_providers import _estimate_num_ctx


def test_ollama_qwen_provider_generate_and_embed(monkeypatch):
    ollama_module = types.SimpleNamespace()

    def fake_generate(model: str, prompt: str, options: dict):
        assert model == "qwen2:1.5b"
        assert prompt == "Hello"
        assert options["num_predict"] == 16
        assert options["num_ctx"] >= 2048  # sized from the prompt, not hardcoded
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


def test_estimate_num_ctx_scales_with_prompt_and_output_length():
    small = _estimate_num_ctx("hi", max_tokens=16)
    large = _estimate_num_ctx("x" * 20000, max_tokens=16)
    assert small < large
    assert large >= 20000 // 2  # never smaller than the estimated input alone


def test_estimate_num_ctx_never_returns_less_than_the_smallest_bucket():
    assert _estimate_num_ctx("", max_tokens=16) == 2048


def test_estimate_num_ctx_caps_at_model_architecture_max():
    # qwen2:1.5b's real max per `ollama show` - must not request more than
    # the model can actually use, however long the prompt is.
    assert _estimate_num_ctx("x" * 200000, max_tokens=512) == 32768


def test_estimate_num_ctx_matches_real_measured_rag_prompt_density():
    # Regression guard for the 2 chars/token conservatism: a real ~10k-char,
    # 3787-token RAG prompt against this project's own chemistry-heavy
    # content (top_k=8 on the "003" library) must land comfortably inside
    # the 8192 bucket, not silently get bucketed at 4096 and risk
    # --context-shift truncating some of the retrieved evidence.
    real_prompt_chars = 10053
    assert _estimate_num_ctx("x" * real_prompt_chars, max_tokens=512) == 8192
