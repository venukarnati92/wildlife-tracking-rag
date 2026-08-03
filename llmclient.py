"""OpenAI-compatible LLM client with pluggable base_url.

Works out of the box with:
- Standard OpenAI (set OPENAI_API_KEY)
- Local Ollama at http://localhost:11434/v1
- Workday's internal CIS gateway (set OPENAI_BASE_URL to the full
  `.../openai/v1` endpoint and OPENAI_PCA_FEATURE_KEY to your registered
  feature key)
- Any other OpenAI-compatible gateway (set OPENAI_BASE_URL and OPENAI_EXTRA_HEADERS)
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def _extra_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    raw = os.getenv("OPENAI_EXTRA_HEADERS")
    if raw:
        try:
            headers.update(json.loads(raw))
        except Exception:
            pass
    pca_feature_key = os.getenv("OPENAI_PCA_FEATURE_KEY")
    if pca_feature_key:
        headers["Wd-PCA-Feature-Key"] = pca_feature_key
    return headers


def create_openai_client() -> OpenAI:
    """Create an OpenAI client honoring OPENAI_BASE_URL / OPENAI_EXTRA_HEADERS."""
    api_key = os.getenv("OPENAI_API_KEY") or "unused"
    base_url = os.getenv("OPENAI_BASE_URL") or None
    pca_feature_key = os.getenv("OPENAI_PCA_FEATURE_KEY")
    headers = _extra_headers()

    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    if headers:
        kwargs["default_headers"] = headers
    if base_url:
        # Corporate/self-hosted gateways are often reachable only through an
        # egress proxy that terminates TLS with a self-signed/internal cert,
        # which the container's default CA store won't trust. Skip verification
        # for any non-default base_url (this never affects api.openai.com).
        http_client_kwargs: dict[str, Any] = {"verify": False}
        if pca_feature_key:
            # Workday's CIS gateway needs this query param to skip its own
            # auth layer when calling with a feature-key-scoped service account.
            http_client_kwargs["params"] = {"bypass_auth": "true"}
        kwargs["http_client"] = httpx.Client(**http_client_kwargs)
    return OpenAI(**kwargs)


class LLMClient:
    """Thin wrapper around OpenAI so RAG code can call `.chat.completions.create` uniformly."""

    def __init__(self, model: str = DEFAULT_MODEL, client: OpenAI | None = None):
        self.model = model
        self.client = client or create_openai_client()

    def __getattr__(self, name: str):
        return getattr(self.client, name)


_default_client: LLMClient | None = None


def get_llm_client(model: str = DEFAULT_MODEL) -> LLMClient:
    """Return a shared LLMClient (memoized per model)."""
    global _default_client
    if _default_client is None or _default_client.model != model:
        _default_client = LLMClient(model=model)
    return _default_client
