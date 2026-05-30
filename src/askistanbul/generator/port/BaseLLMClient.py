"""Abstract LLM client (port in hexagonal architecture)."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any


class BaseLLMClient(ABC):
    """Abstract LLM client.

    Concrete adapters (OllamaClient, future OpenAIClient / AnthropicClient)
    must implement ``chat``. ``chat_json`` has a default implementation that
    calls ``chat`` and parses the response; adapters with native JSON support
    should override it.
    """

    def __init__(self, model: str):
        self.model: str = model

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_new_tokens: int = 2000,
    ) -> str:
        """Send chat messages and return the generated text.

        Args:
            messages: list of {"role": "system"|"user"|"assistant", "content": "..."}
            temperature: sampling temperature (default 0.3 — low for grounded RAG)
            max_new_tokens: hard cap on generated tokens (default 2000)

        Returns:
            Generated text string.
        """

    def chat_json(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_new_tokens: int = 2000,
    ) -> dict[str, Any]:
        """Send chat messages, parse the response as JSON, return the object.

        Default implementation calls ``self.chat()`` and parses with
        ``json.loads``. Adapters with native JSON modes (Ollama ``format=json``,
        OpenAI ``response_format``) should override.

        Raises:
            ValueError: if the response is not valid JSON.
        """
        response = self.chat(messages, temperature=temperature, max_new_tokens=max_new_tokens)
        try:
            return json.loads(response)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse LLM response as JSON: {exc}") from exc
