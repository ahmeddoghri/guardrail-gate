"""Settings loaded from environment variables. Everything has a safe default
so the service runs (and CI passes) with zero external configuration.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # Optional API-key auth: when set, write endpoints require a matching
    # X-API-Key header. Empty (the default) leaves the service open.
    api_key: str = os.environ.get("API_KEY", "")
    # Request-size guards to keep a single caller from exhausting memory.
    max_text_chars: int = int(os.environ.get("MAX_TEXT_CHARS", "100000"))
    max_sources: int = int(os.environ.get("MAX_SOURCES", "256"))
    max_client_id_chars: int = int(os.environ.get("MAX_CLIENT_ID_CHARS", "256"))


settings = Settings()
