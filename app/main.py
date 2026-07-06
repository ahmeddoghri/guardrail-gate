"""FastAPI guardrail service: PII redaction + citation grounding + rate
limiting for outbound LLM text. Run locally with
`uvicorn app.main:app --reload`.
"""
from __future__ import annotations

from typing import Optional

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import security
from .config import settings
from .gate import GuardrailGate
from .security import require_api_key

app = FastAPI(title="guardrail-gate", version="0.1.0")
security.install(app)
_gate = GuardrailGate()


class GuardRequest(BaseModel):
    client_id: str = Field(min_length=1, max_length=settings.max_client_id_chars)
    text: str = Field(min_length=1, max_length=settings.max_text_chars)
    sources: Optional[list[str]] = Field(default=None, max_length=settings.max_sources)


class PIIMatchOut(BaseModel):
    kind: str
    text: str
    start: int
    end: int


class GuardResponse(BaseModel):
    allowed: bool
    redacted_text: str
    pii_found: list[PIIMatchOut]
    grounded_fraction: Optional[float] = None
    warnings: list[str]


@app.get("/healthz")
def healthz() -> dict:
    """Liveness probe: the process is up."""
    return {"status": "ok"}


@app.get("/readyz")
def readyz() -> dict:
    """Readiness probe: the gate is wired and the service can serve."""
    return {"status": "ready"}


@app.post("/v1/guard", response_model=GuardResponse,
          dependencies=[Depends(require_api_key)])
def guard(req: GuardRequest):
    result = _gate.check(req.client_id, req.text, req.sources)
    if result.rate_limited:
        return JSONResponse(status_code=429, content={"detail": "rate limited"})
    return GuardResponse(
        allowed=result.allowed,
        redacted_text=result.redacted_text,
        pii_found=[PIIMatchOut(kind=m.kind, text=m.text, start=m.start, end=m.end)
                   for m in result.pii_found],
        grounded_fraction=result.grounding.grounded_fraction if result.grounding else None,
        warnings=result.warnings,
    )
