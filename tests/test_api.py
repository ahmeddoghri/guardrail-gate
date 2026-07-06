from fastapi.testclient import TestClient

from app.main import app
from app.eval import SOURCE_DOCS

client = TestClient(app)


def test_healthz():
    resp = client.get("/healthz")
    assert resp.status_code == 200


def test_guard_redacts_pii_and_allows_grounded_text():
    resp = client.post("/v1/guard", json={
        "client_id": "test-client",
        "text": "Contact jane.doe@example.com. Your order ships within 3 to 5 business days.",
        "sources": SOURCE_DOCS,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["allowed"] is True
    assert "jane.doe@example.com" not in body["redacted_text"]
    assert any(m["kind"] == "email" for m in body["pii_found"])


def test_guard_blocks_hallucinated_text():
    resp = client.post("/v1/guard", json={
        "client_id": "test-client-2",
        "text": "Your order ships within 24 hours guaranteed, no exceptions ever.",
        "sources": SOURCE_DOCS,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["allowed"] is False
    assert "low_grounding" in body["warnings"]


def test_guard_without_sources_only_does_pii():
    resp = client.post("/v1/guard", json={
        "client_id": "test-client-3",
        "text": "Just a plain message, nothing special here.",
    })
    assert resp.status_code == 200
    assert resp.json()["allowed"] is True
