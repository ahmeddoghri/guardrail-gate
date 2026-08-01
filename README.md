# guardrail-gate

![CI](https://github.com/ahmeddoghri/guardrail-gate/actions/workflows/ci.yml/badge.svg)
![tests](https://img.shields.io/badge/tests-76%20passing-brightgreen)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-black)

> **Catch PII leaks and ungrounded claims in a single pass.** One
> allowed/blocked decision with the reasons attached. Zero API keys to
> try it: `python -m app.eval`.
>
> The original checks scored 100% on PII and 83% on grounding, and both
> numbers were measured on the easy half. Against hallucinations that reuse
> the source's own words, the grounding check missed **6 of 8** and passed a
> flat negation at 0.90 overlap. `python -m app.advbench` is the benchmark
> that shows it, and what replaced it.

Picture the coworker who overshares personal details AND makes things up,
in the same sentence, without noticing either. That's an unguarded LLM
response. guardrail-gate is the bouncer standing between your model's
output and your user: it redacts PII, checks whether each claim is
actually supported by the sources you retrieved, and rate-limits abusive
clients, then hands you one allowed/blocked decision with the reasons
attached.

This is the same discipline I built into regulatory-document LLM
pipelines: you don't ship a response to a user, or a compliance reviewer,
without knowing whether it's grounded in something real, and you don't
let a chat transcript leak an SSN because nobody thought to check. The
version here is rebuilt from scratch to be small, runnable, and
inspectable. No proprietary code, same idea.

## Why two separate checks

PII and hallucination are different failure modes, and conflating them
produces worse guardrails, not better ones:

- **PII redaction** doesn't care if the text is *true*. An email address
  is still an email address whether the surrounding claim checks out or
  not.
- **Grounding** doesn't care if the text is *sensitive*. A hallucinated
  shipping date is a problem even with zero PII anywhere near it.

Running them as one combined "safety score" tends to hide which one
actually triggered. This keeps them separate and reports both, like a
bouncer checking your ID and your invite list as two different things,
not one vague vibe check.

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

Those numbers are real, and both were measured on the easy half of the
problem. Measured against the hard half, the guardrail failed badly enough
to be worth showing:

```bash
python -m app.advbench
```

| detector | split | precision | recall | false pos | false neg |
| --- | --- | ---: | ---: | ---: | ---: |
| v1 regex | clean | 100% | 100% | 0 | 0 |
| v1 regex | obfuscated | 100% | **29%** | 0 | 5 |
| v1 regex | decoy | **0%** | 100% | 4 | 0 |
| v2 validating | clean | 100% | 100% | 0 | 0 |
| v2 validating | obfuscated | 100% | 100% | 0 | 0 |
| v2 validating | decoy | 100% | 100% | 0 | 0 |

| checker | split | accuracy | hallucinations missed |
| --- | --- | ---: | ---: |
| v1 overlap | lexical | 83% | 1 |
| v1 overlap | **semantic** | **25%** | **6 of 8** |
| v2 semantic | lexical | 100% | 0 |
| v2 semantic | semantic | 100% | 0 |

### The grounding check could not see meaning

The README used to say lexical overlap "won't catch subtle factual drift".
That was the right instinct and much too gentle. Here is what actually got
through:

```
source:   "Refunds are processed within 10 business days of the return being received."
response: "Refunds are NOT processed within 10 business days of the return being received."
overlap:  0.90  ->  reported as grounded
```

One inserted word inverts the policy and moves the score by a rounding
error. Six of eight semantic hallucinations passed, including `$49 per
month` becoming `$49 per year` at 0.89 overlap, and `3 to 5 business days`
becoming `30 to 50`.

This is not an edge case, it is the central one. **A model generating from
retrieved context does not invent new vocabulary; it recombines the
vocabulary in front of it.** So the hallucinations that actually happen are
exactly the ones bag-of-words similarity scores highest. The old benchmark
missed this because its hallucinated cases were written with fresh words
("free for the first year", "same-day refunds"), which overlap already
handles.

The fix is not a better similarity metric. It is checking what a claim
asserts, as a conjunction of things that each carry meaning:

- **Polarity.** A claim and its source must agree about whether something happens.
- **Quantities.** Every number in the claim, with its unit, must appear in the source. `49:month` and `49:year` are different facts.
- **Conditions.** "of the return being received" and "of the order being placed" start the clock at different events.
- **Overlap**, still, as the floor for topical relevance.

Genuine paraphrases survive, which is the constraint that makes it usable:
"once your order is confirmed" matches "after order confirmation" because
conditions compare as stemmed word sets, not strings. Zero false alarms
across the corpus.

### The PII detector matched patterns instead of validating them

It failed in both directions at once. It missed `jane dot doe at example dot
com`, `123 45 6789`, and `+44 20 7946 0958`: 29% recall on PII as people
actually type it. And it redacted `Version 1.2.3.4` as an IP address and
order number `4111 1111 1111 1112` as a credit card, which is the failure
that gets a redactor switched off entirely.

v2 validates rather than pattern-matches:

- **Luhn checksum plus issuer prefix.** A card is not sixteen digits; it is sixteen digits that check out and begin with a real BIN. `1234567890123456` is not a card.
- **SSA issuance rules.** 000, 666, and 900-999 are never issued as area numbers.
- **NANP rules.** Area code and exchange both start 2-9, which is what makes `100-200-3000` a measurement.
- **Octet range and zero-padding.** `999.888.777.666` is not an address, and `Version 1.2.3.4` is a build number.

Every match carries the reason it fired, because a redaction a reviewer
cannot explain is one they will override.

### Held out, run once

Both v2 components were built against the corpus above, so those scores are
in-sample. A separate held-out set was written afterwards with the code
frozen and evaluated a single time:

| | v1 | v2 |
| --- | ---: | ---: |
| PII (14 cases) | precision 55%, recall 75% | **precision 100%, recall 100%** |
| Grounding (13 cases) | 46% | **85%** |

v2 misses two held-out cases, both the same shape: `"...on weekdays"`
rewritten as `"...on weekends"`. That is a scope substitution with no number,
no negation, and no prepositional condition to catch it. It is deliberately
not patched, because tuning against a holdout after reading the result is how
a holdout stops being one. It stands as the documented failure mode: **v2
checks polarity, quantities, and conditions, so a hallucination that swaps
one bare noun for a related one can still get through.**

### Limits

- **Still no NER.** Names and street addresses in prose need a real model; the `PIIDetector` protocol takes one.
- **Still not entailment.** This checks four specific things well, not arbitrary logical consequence.
- **A blocked response is not always wrong**, and an allowed one is not verified true. This is a filter, not an oracle.

Both versions remain selectable (`GuardrailGate(version="v1")`) so the
comparison is reproducible rather than a claim you have to take on faith.

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

For grounding, plug a real embedding-similarity or NLI-based entailment
check in place of `check_grounding` if lexical overlap isn't precise
enough for your domain.

## Production configuration

All settings have safe defaults; override via environment variables.

| Variable | Default | Purpose |
|---|---|---|
| `API_KEY` | *(empty)* | When set, `/v1/guard` requires a matching `X-API-Key` header. Empty leaves the service open. |
| `MAX_TEXT_CHARS` | `100000` | Rejects (422) text larger than this to bound memory. |
| `MAX_SOURCES` | `256` | Caps how many source documents one request may carry. |

The service exposes `GET /healthz` (liveness) and `GET /readyz`
(readiness). Every response carries an `X-Request-ID` header, requests
are logged with method, path, status, and latency, and unhandled errors
return a structured `500` without leaking stack traces.

## Tests

```bash
pip install -r requirements-dev.txt && pytest -q      # 20 passing
```

## More in this series

Nine small, dependency-light, benchmarked tools for LLM/ML infrastructure. Each one reproduces its headline number locally with no API keys:

[agentmem](https://github.com/ahmeddoghri/agentmem) · [rubricagent](https://github.com/ahmeddoghri/rubricagent) · [clarifyrag](https://github.com/ahmeddoghri/clarifyrag) · [churnfm](https://github.com/ahmeddoghri/churnfm) · [citebench](https://github.com/ahmeddoghri/citebench) · [tablextract](https://github.com/ahmeddoghri/tablextract) · [vllm-cost-router](https://github.com/ahmeddoghri/vllm-cost-router) · [taggate](https://github.com/ahmeddoghri/taggate)

## License

MIT © Ahmed Doghri
