"""FastAPI guardrail service: PII redaction + citation grounding + rate
limiting for outbound LLM text. Run locally with
`uvicorn app.main:app --reload`.
"""
from __future__ import annotations

from typing import Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .gate import GuardrailGate

app = FastAPI(title="guardrail-gate", version="0.1.0")
_gate = GuardrailGate()


class GuardRequest(BaseModel):
    client_id: str
    text: str
    sources: Optional[list[str]] = None


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
    return {"status": "ok"}


@app.post("/v1/guard", response_model=GuardResponse)
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
