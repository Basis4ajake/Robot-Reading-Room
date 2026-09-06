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

    sources_response = client.get("/api/v1/libraries/rag-lib/sources")
    assert len(sources_response.json()) == 1

    remove_response = client.delete(f"/api/v1/libraries/rag-lib/sources/{source['source_id']}")
    assert remove_response.status_code == 204
    assert client.get("/api/v1/libraries/rag-lib/sources").json() == []


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
