# Security Policy

## Supported versions

This project is pre-1.0. Security fixes land on `main`; please track the
latest commit.

## Reporting a vulnerability

Please **do not** open a public issue for security problems. Instead, use
GitHub's [private vulnerability reporting](https://github.com/ahmeddoghri/guardrail-gate/security/advisories/new)
or email the maintainer. Include:

- a description of the issue and its impact,
- steps to reproduce (a minimal proof-of-concept helps),
- any suggested remediation.

You can expect an initial acknowledgement within a few days. Once a fix is
available it will be released and you will be credited unless you prefer to
remain anonymous.

## Hardening notes

- Set `API_KEY` to require an `X-API-Key` header on `/v1/guard`.
- Run the service behind a TLS-terminating reverse proxy.
- The container runs as a non-root user; keep it that way.
- Request bodies are size-limited (`MAX_TEXT_CHARS`, `MAX_SOURCES`) to bound
  memory use.
- Redacted PII spans are returned to the caller so they can audit what was
  removed; treat that response as sensitive and log it accordingly.
