"""Endpoint tests for the FastAPI server: fast, no LLM.

The lifespan builds the real graph (compile only — no model call happens
until invoke), then tests that need scripted behavior swap in a
conftest.fake_graph via app.state. The SqliteSaver deadlock guard for
pending enumeration lives in test_graph.py, next to pending_interrupts.
"""

from pathlib import Path

import pytest
from conftest import extraction_json, fake_graph
from fastapi.testclient import TestClient

from freightcase import server

FIXTURES = Path(__file__).parent / "fixtures" / "emails"
ROAD_EML = FIXTURES / "quote_road_plain_es.eml"
EMPTY_EML = FIXTURES / "not-an-email.eml"


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Keep checkpoint writes out of logs/: each test gets a throwaway DB.
    monkeypatch.setattr(server, "DB_PATH", str(tmp_path / "checkpoints.sqlite"))
    with TestClient(server.app) as client:
        yield client


def upload(
    client: TestClient,
    path: Path,
    *,
    content_type: str = "message/rfc822",
    filename: str | None = None,
):
    return client.post(
        "/ingest/",
        files={"file": (filename or path.name, path.read_bytes(), content_type)},
    )


@pytest.mark.parametrize(
    "content_type",
    ["message/rfc822", "application/octet-stream"],  # curl -F sends the latter
)
def test_ingest_pauses_with_confirmation_payload(client, content_type):
    server.app.state.graph, _ = fake_graph(extraction_json())

    response = upload(client, ROAD_EML, content_type=content_type)

    assert response.status_code == 200
    body = response.json()
    assert body["thread_id"]
    assert body["status"] == "awaiting_confirmation"
    assert body["payload"]["action"] == {
        "tool": "create_quote",
        "function": "quote_request",
    }


def test_ingest_dead_letter_reports_failed_not_500(client):
    server.app.state.graph, _ = fake_graph(
        classification='{"classification": "unknown", "reason": "marketing newsletter"}'
    )

    response = upload(client, ROAD_EML)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "failed"
    assert "No specialist" in body["error"]


def test_ingest_unparseable_eml_is_422(client):
    response = upload(client, EMPTY_EML)

    assert response.status_code == 422
    assert "Ingestion failed" in response.json()["detail"]


def test_ingest_rejects_non_eml_extension(client):
    response = upload(client, ROAD_EML, filename="quote.msg")

    assert response.status_code == 400
    assert "extension" in response.json()["detail"]


def test_ingest_rejects_wrong_mime_type(client):
    response = upload(client, ROAD_EML, content_type="text/plain")

    assert response.status_code == 400
    assert "Invalid MIME type" in response.json()["detail"]


def test_ingest_rejects_oversized_file(client, monkeypatch):
    monkeypatch.setattr(server, "MAX_FILE_SIZE", 10)

    response = upload(client, ROAD_EML)

    assert response.status_code == 400
    assert "maximum limit" in response.json()["detail"]


def test_confirm_approval_journey_executes_and_clears_pending(client):
    """Ingest pauses -> approve -> executed with the stub TMS reference.
    The decided thread then leaves /pending and cannot be confirmed again."""
    server.app.state.graph, _ = fake_graph(extraction_json())
    thread_id = upload(client, ROAD_EML).json()["thread_id"]

    response = client.post(f"/confirm/{thread_id}", json={"approved": True})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "executed"
    assert body["execution_ref"] == "Q-STUB-1"
    assert body["error"] is None

    assert client.get("/pending/").json() == []
    second = client.post(f"/confirm/{thread_id}", json={"approved": True})
    assert second.status_code == 404


def test_confirm_rejection_records_decline_without_executing(client):
    server.app.state.graph, _ = fake_graph(extraction_json())
    thread_id = upload(client, ROAD_EML).json()["thread_id"]

    response = client.post(f"/confirm/{thread_id}", json={"approved": False})

    body = response.json()
    assert body["status"] == "rejected"
    assert body["execution_ref"] is None


def test_confirm_bad_edit_reprompts_then_accepts_fix(client):
    """Approve with an invalid edit -> re-prompt in the same paused shape
    /ingest returns, problems naming the reason -> corrected edit executes."""
    server.app.state.graph, _ = fake_graph(extraction_json(exclude=("cargo.0.weight",)))
    ingested = upload(client, ROAD_EML).json()
    thread_id = ingested["thread_id"]
    assert ingested["payload"]["missing"] == ["cargo.0.weight"]

    reprompt = client.post(
        f"/confirm/{thread_id}",
        json={
            "approved": True,
            "edits": {"cargo.0.weight": {"value": 10, "unit": "quintales"}},
        },
    ).json()
    assert reprompt["status"] == "awaiting_confirmation"
    assert "quintales" in reprompt["payload"]["problems"][0]

    done = client.post(
        f"/confirm/{thread_id}",
        json={
            "approved": True,
            "edits": {"cargo.0.weight": {"value": 1.2, "unit": "toneladas"}},
        },
    ).json()
    assert done["status"] == "executed"


def test_confirm_unknown_thread_is_404(client):
    response = client.post("/confirm/no-such-thread", json={"approved": True})

    assert response.status_code == 404


def test_confirm_shape_error_is_rejected_at_the_boundary(client):
    """A body failing ConfirmationResume validation is a 422 from FastAPI:
    it never reaches the graph, so it cannot burn a re-prompt round. (The
    graph-level guard for malformed resumes is pinned in test_graph.py.)"""
    server.app.state.graph, _ = fake_graph(extraction_json())
    thread_id = upload(client, ROAD_EML).json()["thread_id"]

    response = client.post(f"/confirm/{thread_id}", json={"accepted": True})

    assert response.status_code == 422
    # The thread survives untouched: still pending, still confirmable.
    retry = client.post(f"/confirm/{thread_id}", json={"approved": True})
    assert retry.json()["status"] == "executed"


def test_pending_endpoint_lists_only_paused_threads(client):
    """One paused case, one dead letter, one IntakeError orphan: /pending
    must return exactly the paused one, in the same shape /ingest pauses
    with."""
    server.app.state.graph, _ = fake_graph(
        # Upload 1: routed to the specialist, pauses at confirm.
        extraction_json(),
        # Upload 2: dead-lettered, completes without pausing. The scripted
        # model serves calls in order: classify 1, extract 1, classify 2.
        '{"classification": "unknown", "reason": "marketing"}',
        classification='{"classification": "quote_request"}',
    )

    paused = upload(client, ROAD_EML).json()
    dead_lettered = upload(client, ROAD_EML).json()
    assert dead_lettered["status"] == "failed"
    assert upload(client, EMPTY_EML).status_code == 422  # orphaned thread

    body = client.get("/pending/").json()
    assert [case["thread_id"] for case in body] == [paused["thread_id"]]
    assert body[0]["status"] == "awaiting_confirmation"
    assert body[0]["payload"]["action"] == {
        "tool": "create_quote",
        "function": "quote_request",
    }
