# guardrail-gate

![CI](https://github.com/ahmeddoghri/guardrail-gate/actions/workflows/ci.yml/badge.svg)
![tests](https://img.shields.io/badge/tests-15%20passing-brightgreen)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-black)

A guardrail service that sits between your LLM's output and your user. It
redacts PII, checks whether each claim in the response is actually supported
by the source documents you retrieved, and rate-limits abusive clients —
then gives you a single allowed/blocked decision with the reasons attached.

This is the same discipline I built into regulatory-document LLM pipelines:
you don't ship a response to a user (or a compliance reviewer) without
knowing whether it's grounded in something real, and you don't let a chat
transcript leak an SSN because nobody thought to check. The version here is
rebuilt from scratch to be small, runnable, and inspectable — no proprietary
code, same idea.

## Why two separate checks

PII and hallucination are different failure modes and conflating them
produces worse guardrails, not better ones:

- **PII redaction** doesn't care if the text is *true* — an email address is
  still an email address whether the surrounding claim is accurate or not.
- **Grounding** doesn't care if the text is *sensitive* — a hallucinated
  shipping date is a problem even if it contains zero PII.

Running them as one combined "safety score" tends to hide which one actually
triggered. This keeps them separate and reports both.

## The result, on labeled benchmarks

```bash
python -m app.eval
```
```
--- PII redaction benchmark ---
precision=100%  recall=100%  (n=8)

--- grounding / hallucination-flagging benchmark ---
accuracy=83%  (n=6)
```

The PII benchmark is regex-complete for structured PII (emails, phone
numbers, SSNs, credit cards, IPs) — it won't catch a name or an address
mentioned in prose, which needs real NER. The grounding benchmark is
lexical-overlap based, which reliably catches the common case (the model
asserting something the sources never said) but won't catch subtle factual
drift within an otherwise-grounded sentence. Both limitations are disclosed
here on purpose — a guardrail you don't understand the limits of is worse
than no guardrail, because it creates false confidence.

## Install & run

```bash
git clone https://github.com/ahmeddoghri/guardrail-gate
cd guardrail-gate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

Or with Docker:

```bash
docker build -t guardrail-gate .
docker run -p 8000:8000 guardrail-gate
```

Try it:

```bash
curl -X POST localhost:8000/v1/guard \
  -H "Content-Type: application/json" \
  -d '{
    "client_id": "demo",
    "text": "Contact jane.doe@example.com. Your order ships within 3 to 5 business days.",
    "sources": ["The product ships within 3 to 5 business days after order confirmation."]
  }'
```
```json
{
  "allowed": true,
  "redacted_text": "Contact [REDACTED_EMAIL]. Your order ships within 3 to 5 business days.",
  "pii_found": [{"kind": "email", "text": "jane.doe@example.com", "start": 8, "end": 28}],
  "grounded_fraction": 1.0,
  "warnings": ["pii_redacted:email"]
}
```

## How it decides

```
POST /v1/guard
  ├─ rate limit check (per client_id token bucket) -- 429 if exhausted
  ├─ PII detection + redaction (structured patterns: email, ssn, card, phone, ip)
  ├─ if sources provided:
  │     split response into sentences, score each against the source set
  │     grounded_fraction < threshold?  -> blocked, warning: low_grounding
  └─ return { allowed, redacted_text, pii_found, grounded_fraction, warnings }
```

## Bring your own PII/grounding backend

The regex detector and lexical grounding checker are both intentionally
swappable:

```python
class MyNERDetector:
    def detect(self, text: str) -> list[PIIMatch]: ...   # e.g. spaCy, a hosted NER API

GuardrailGate(pii_detector=MyNERDetector())
```

For grounding, plug a real embedding-similarity or NLI-based entailment check
in place of `check_grounding` if lexical overlap isn't precise enough for
your domain.

## Tests

```bash
pip install -r requirements-dev.txt && pytest -q      # 15 passing
```

## License

MIT © Ahmed Doghri
