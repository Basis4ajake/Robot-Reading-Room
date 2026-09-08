import sys
import threading
import types

import pytest
from fastapi.testclient import TestClient

from local_knowledge_library.api.app import create_app


@pytest.fixture
def client(tmp_path):
    app = create_app(data_dir=str(tmp_path / "libraries"), force_dummy=True)
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["ollama_available"] is False


def test_models_unavailable_when_force_dummy(client):
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    assert response.json() == {"available": False, "models": []}


def test_library_lifecycle(client):
    create_response = client.post(
        "/api/v1/libraries",
        json={"library_id": "test-lib", "name": "Test Library", "description": "for tests"},
    )
    assert create_response.status_code == 201
    created = create_response.json()
    assert created["library_id"] == "test-lib"
    assert created["llm_model"] == "qwen2:1.5b"

    list_response = client.get("/api/v1/libraries")
    assert list_response.status_code == 200
    assert [lib["library_id"] for lib in list_response.json()] == ["test-lib"]

    get_response = client.get("/api/v1/libraries/test-lib")
    assert get_response.status_code == 200
    assert get_response.json()["name"] == "Test Library"

    update_response = client.patch("/api/v1/libraries/test-lib", json={"top_k": 3})
    assert update_response.status_code == 200
    assert update_response.json()["top_k"] == 3

    rename_response = client.patch(
        "/api/v1/libraries/test-lib", json={"name": "Renamed Library", "description": "updated"}
    )
    assert rename_response.status_code == 200
    assert rename_response.json()["name"] == "Renamed Library"
    assert rename_response.json()["description"] == "updated"

    reget_response = client.get("/api/v1/libraries/test-lib")
    assert reget_response.json()["name"] == "Renamed Library"

    relist_response = client.get("/api/v1/libraries")
    assert relist_response.json()[0]["name"] == "Renamed Library"

    missing_response = client.get("/api/v1/libraries/does-not-exist")
    assert missing_response.status_code == 404

    duplicate_response = client.post(
        "/api/v1/libraries", json={"library_id": "test-lib", "name": "Test Library"}
    )
    assert duplicate_response.status_code == 409


def test_create_conflicts_and_delete_requires_confirm(client):
    client.post("/api/v1/libraries", json={"library_id": "lib-a", "name": "Lib A"})

    no_confirm = client.delete("/api/v1/libraries/lib-a")
    assert no_confirm.status_code == 400

    confirmed = client.delete("/api/v1/libraries/lib-a?confirm=true")
    assert confirmed.status_code == 204

    assert client.get("/api/v1/libraries/lib-a").status_code == 404


def test_source_ingest_and_chat_end_to_end(client, tmp_path):
    client.post("/api/v1/libraries", json={"library_id": "rag-lib", "name": "RAG Library"})

    source_file = tmp_path / "example.txt"
    source_file.write_text("Hello world. This is a simple knowledge base entry.", encoding="utf-8")

    add_response = client.post(
        "/api/v1/libraries/rag-lib/sources", json={"source_path": str(source_file)}
    )
    assert add_response.status_code == 201
    source = add_response.json()
    assert source["filename"] == "example.txt"

    ingest_response = client.post("/api/v1/libraries/rag-lib/ingest")
    assert ingest_response.status_code == 200
    assert len(ingest_response.json()["processed"]) == 1

    chat_response = client.post(
        "/api/v1/libraries/rag-lib/chat", json={"query": "What does the document say?"}
    )
    assert chat_response.status_code == 200
    body = chat_response.json()
    assert body["query"] == "What does the document say?"
    assert body["answer"]
    assert body["citations"]
    assert body["answer_source"] == "dummy"

    sources_response = client.get("/api/v1/libraries/rag-lib/sources")
    assert len(sources_response.json()) == 1

    remove_response = client.delete(f"/api/v1/libraries/rag-lib/sources/{source['source_id']}")
    assert remove_response.status_code == 204
    assert client.get("/api/v1/libraries/rag-lib/sources").json() == []


def test_eval_case_lifecycle_and_run_history(client, tmp_path):
    client.post("/api/v1/libraries", json={"library_id": "eval-lib", "name": "Eval Library"})

    source_file = tmp_path / "example.txt"
    source_file.write_text("Hello world. This is a simple knowledge base entry.", encoding="utf-8")
    client.post("/api/v1/libraries/eval-lib/sources", json={"source_path": str(source_file)})
    client.post("/api/v1/libraries/eval-lib/ingest")

    assert client.get("/api/v1/libraries/eval-lib/eval-cases").json() == []

    create_response = client.post(
        "/api/v1/libraries/eval-lib/eval-cases",
        json={"question": "What does the document say?", "expected_keyword": "hello"},
    )
    assert create_response.status_code == 201
    case = create_response.json()
    assert case["question"] == "What does the document say?"

    list_response = client.get("/api/v1/libraries/eval-lib/eval-cases")
    assert len(list_response.json()) == 1

    assert client.get("/api/v1/libraries/eval-lib/eval-runs").json() == []

    run_response = client.post("/api/v1/libraries/eval-lib/evaluate")
    assert run_response.status_code == 200
    run = run_response.json()
    assert run["total_count"] == 1
    assert run["llm_model"] == "qwen2:1.5b"
    assert run["chunk_size"] == 300
    assert len(run["results"]) == 1
    assert run["results"][0]["expected_keyword"] == "hello"
    # force_dummy fixture means DummyLLMProvider/DummyEmbedder, so this only
    # proves the plumbing works end-to-end - not that retrieval is accurate.
    assert run["results"][0]["answer_source"] == "dummy"

    history_response = client.get("/api/v1/libraries/eval-lib/eval-runs")
    assert history_response.status_code == 200
    history = history_response.json()
    assert len(history) == 1
    assert history[0]["eval_run_id"] == run["eval_run_id"]

    delete_response = client.delete(f"/api/v1/libraries/eval-lib/eval-cases/{case['eval_case_id']}")
    assert delete_response.status_code == 204
    assert client.get("/api/v1/libraries/eval-lib/eval-cases").json() == []
    # Removing a case doesn't retroactively erase past runs - history stores
    # question/expected_keyword by value, not a live reference to the case.
    assert len(client.get("/api/v1/libraries/eval-lib/eval-runs").json()) == 1


def test_evaluate_with_no_cases_returns_empty_run(client):
    client.post("/api/v1/libraries", json={"library_id": "eval-lib3", "name": "Eval Library 3"})

    response = client.post("/api/v1/libraries/eval-lib3/evaluate")

    assert response.status_code == 200
    body = response.json()
    assert body["total_count"] == 0
    assert body["passed_count"] == 0
    assert body["results"] == []


def test_evaluate_returns_404_for_unknown_library(client):
    response = client.post("/api/v1/libraries/nope/evaluate")
    assert response.status_code == 404


def test_remove_unknown_eval_case_returns_404(client):
    client.post("/api/v1/libraries", json={"library_id": "eval-lib2", "name": "Eval Library 2"})
    response = client.delete("/api/v1/libraries/eval-lib2/eval-cases/does-not-exist")
    assert response.status_code == 404


def test_create_rejects_non_positive_chunk_size(client):
    """chunk_size <= 0 used to be silently accepted and treated as "don't
    sub-split at all" (ParagraphChunker._split_to_size) rather than
    rejected - the GUI's min="1" caught this for GUI users, a direct API
    call had no such guard."""
    response = client.post(
        "/api/v1/libraries",
        json={"library_id": "bad-chunk-size", "name": "Bad", "chunk_size": 0},
    )
    assert response.status_code == 422

    response = client.post(
        "/api/v1/libraries",
        json={"library_id": "bad-chunk-size", "name": "Bad", "chunk_size": -5},
    )
    assert response.status_code == 422


def test_update_rejects_non_positive_chunk_size(client):
    client.post("/api/v1/libraries", json={"library_id": "test-lib", "name": "Test"})
    response = client.patch("/api/v1/libraries/test-lib", json={"chunk_size": 0})
    assert response.status_code == 422
    # A real, valid update still works after a rejected one.
    response = client.patch("/api/v1/libraries/test-lib", json={"chunk_size": 500})
    assert response.status_code == 200
    assert response.json()["chunk_size"] == 500


def test_chat_rejects_non_positive_top_k(client):
    client.post("/api/v1/libraries", json={"library_id": "test-lib", "name": "Test"})
    response = client.post("/api/v1/libraries/test-lib/chat", json={"query": "hi", "top_k": 0})
    assert response.status_code == 422


def test_patch_returns_409_while_an_ingest_is_in_flight(client):
    client.post("/api/v1/libraries", json={"library_id": "busy-lib", "name": "Busy Lib"})
    app_state = client.app.state.lkl

    entered = threading.Event()
    release = threading.Event()

    def hold_runtime():
        with app_state.use_runtime("busy-lib"):
            entered.set()
            release.wait(timeout=5)

    thread = threading.Thread(target=hold_runtime)
    thread.start()
    try:
        assert entered.wait(timeout=5), "use_runtime never entered"
        response = client.patch("/api/v1/libraries/busy-lib", json={"top_k": 3})
        assert response.status_code == 409
    finally:
        release.set()
        thread.join(timeout=5)

    # Busy only while the ingest actually held the runtime.
    response = client.patch("/api/v1/libraries/busy-lib", json={"top_k": 3})
    assert response.status_code == 200
    assert response.json()["top_k"] == 3


def test_second_concurrent_ingest_returns_409(client):
    """use_runtime() alone lets multiple callers hold a library's runtime
    at once (needed so chat isn't blocked by a long ingest) - a second
    /ingest for the SAME library must still be refused, since two ingests
    racing on the same on-disk chunks.json/vectors.db would corrupt it.
    Not reachable through the GUI (its ingest button disables itself
    mid-request), only a direct API caller."""
    client.post("/api/v1/libraries", json={"library_id": "busy-lib", "name": "Busy Lib"})
    app_state = client.app.state.lkl

    entered = threading.Event()
    release = threading.Event()

    def hold_ingest_lock():
        with app_state.ingest_lock("busy-lib"):
            entered.set()
            release.wait(timeout=5)

    thread = threading.Thread(target=hold_ingest_lock)
    thread.start()
    try:
        assert entered.wait(timeout=5), "ingest_lock never entered"
        response = client.post("/api/v1/libraries/busy-lib/ingest")
        assert response.status_code == 409
    finally:
        release.set()
        thread.join(timeout=5)

    # Free again once the in-flight ingest actually finishes.
    response = client.post("/api/v1/libraries/busy-lib/ingest")
    assert response.status_code == 200


def test_create_rejects_embedding_model_not_locally_pulled(tmp_path):
    # Real ollama.list() returns fully-qualified names, not bare ones -
    # confirmed against a real local daemon (an untagged pull comes back
    # as "nomic-embed-text:latest", not "nomic-embed-text").
    ollama_module = types.SimpleNamespace(list=lambda: {"models": [{"model": "nomic-embed-text:latest"}]})
    sys.modules["ollama"] = ollama_module
    try:
        app = create_app(data_dir=str(tmp_path / "libraries"), force_dummy=False)
        with TestClient(app) as ollama_client:
            response = ollama_client.post(
                "/api/v1/libraries",
                json={"library_id": "bad-embed", "name": "Bad", "embedding_model": "typo-model"},
            )
            assert response.status_code == 400
            assert "typo-model" in response.json()["detail"]

            response = ollama_client.post(
                "/api/v1/libraries",
                json={"library_id": "good-embed", "name": "Good", "embedding_model": "nomic-embed-text:latest"},
            )
            assert response.status_code == 201
    finally:
        del sys.modules["ollama"]


def test_create_accepts_untagged_default_against_a_latest_tagged_pull(tmp_path):
    """Regression test for a real bug caught before shipping: this
    project's own default embedding_model ("nomic-embed-text", untagged)
    would have been rejected by an exact-match-only check the moment a
    real Ollama daemon was reachable, since ollama.list() reports it as
    "nomic-embed-text:latest"."""
    ollama_module = types.SimpleNamespace(list=lambda: {"models": [{"model": "nomic-embed-text:latest"}]})
    sys.modules["ollama"] = ollama_module
    try:
        app = create_app(data_dir=str(tmp_path / "libraries"), force_dummy=False)
        with TestClient(app) as ollama_client:
            response = ollama_client.post(
                "/api/v1/libraries",
                json={"library_id": "lib", "name": "Lib", "embedding_model": "nomic-embed-text"},
            )
            assert response.status_code == 201
    finally:
        del sys.modules["ollama"]


def test_create_still_rejects_a_mismatched_explicit_tag(tmp_path):
    """The ":latest" fallback must not turn into stripping tags
    generally - asking for a specific tag that isn't pulled (only a
    different tag of the same model is) must still be rejected."""
    ollama_module = types.SimpleNamespace(list=lambda: {"models": [{"model": "qwen3:8b"}]})
    sys.modules["ollama"] = ollama_module
    try:
        app = create_app(data_dir=str(tmp_path / "libraries"), force_dummy=False)
        with TestClient(app) as ollama_client:
            response = ollama_client.post(
                "/api/v1/libraries",
                json={"library_id": "lib", "name": "Lib", "embedding_model": "qwen3:4b"},
            )
            assert response.status_code == 400
    finally:
        del sys.modules["ollama"]


def test_update_rejects_embedding_model_not_locally_pulled(tmp_path):
    ollama_module = types.SimpleNamespace(list=lambda: {"models": [{"model": "nomic-embed-text:latest"}]})
    sys.modules["ollama"] = ollama_module
    try:
        app = create_app(data_dir=str(tmp_path / "libraries"), force_dummy=False)
        with TestClient(app) as ollama_client:
            ollama_client.post("/api/v1/libraries", json={"library_id": "lib", "name": "Lib"})
            response = ollama_client.patch(
                "/api/v1/libraries/lib", json={"embedding_model": "typo-model"}
            )
            assert response.status_code == 400
    finally:
        del sys.modules["ollama"]


def test_embedding_model_validation_skipped_when_ollama_unreachable(tmp_path):
    """Can't validate a model name against a daemon that isn't there -
    must not block config changes just because Ollama happens to be
    down (same reasoning as /models' own OllamaUnavailableError
    handling)."""

    def fake_list():
        raise ConnectionError("no daemon")

    ollama_module = types.SimpleNamespace(list=fake_list)
    sys.modules["ollama"] = ollama_module
    try:
        app = create_app(data_dir=str(tmp_path / "libraries"), force_dummy=False)
        with TestClient(app) as ollama_client:
            response = ollama_client.post(
                "/api/v1/libraries",
                json={"library_id": "lib", "name": "Lib", "embedding_model": "anything-goes"},
            )
            assert response.status_code == 201
    finally:
        del sys.modules["ollama"]


def test_models_endpoint_reports_unavailable_when_ollama_missing(tmp_path):
    ollama_module = types.SimpleNamespace()

    def fake_list():
        raise ConnectionError("no daemon")

    ollama_module.list = fake_list
    sys.modules["ollama"] = ollama_module

    app = create_app(data_dir=str(tmp_path / "libraries"), force_dummy=False)
    with TestClient(app) as client:
        response = client.get("/api/v1/models")
        assert response.status_code == 200
        assert response.json() == {"available": False, "models": []}

    del sys.modules["ollama"]
