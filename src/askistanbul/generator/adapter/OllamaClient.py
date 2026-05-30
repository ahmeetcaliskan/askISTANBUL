"""Ollama LLM client — local inference via /api/chat."""

from __future__ import annotations

import json
from typing import Any

import requests

from ...config import config
from ..port.BaseLLMClient import BaseLLMClient


class OllamaClient(BaseLLMClient):
    """Local Ollama LLM client.

    Defaults (model, base URL) are read from :data:`askistanbul.config.config`.
    Connection is verified lazily — pass ``verify=True`` to ping at construction,
    or call :meth:`ping` explicitly.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        verify: bool = False,
    ):
        super().__init__(model or config.ollama_model)
        self.base_url = (base_url or config.ollama_base_url).rstrip("/")
        if verify:
            self.ping()

    def ping(self) -> bool:
        """Check Ollama reachability and that ``self.model`` is available.

        Returns:
            True if reachable and the model is loaded; False otherwise.
            Diagnostics are printed; no exception is raised on failure.
        """
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"[OllamaClient] unreachable at {self.base_url}: {exc}")
            return False

        available = [m["name"] for m in resp.json().get("models", [])]
        if self.model not in available:
            print(
                f"[OllamaClient] model {self.model!r} not loaded. "
                f"Available: {available}"
            )
            return False
        print(f"[OllamaClient] OK — {self.base_url}, model={self.model}")
        return True

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_new_tokens: int = 2000,
    ) -> str:
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_new_tokens,
                },
            },
            timeout=180,
        )
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "")

    def chat_json(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_new_tokens: int = 2000,
    ) -> dict[str, Any]:
        """Use Ollama's native JSON mode (``format='json'``), then parse."""
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": temperature,
                    "num_predict": max_new_tokens,
                },
            },
            timeout=180,
        )
        resp.raise_for_status()
        raw = resp.json().get("message", {}).get("content", "")
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Failed to parse Ollama JSON response: {exc}\nRaw: {raw[:200]}"
            ) from exc
