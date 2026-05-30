"""Factory for constructing concrete :class:`BaseLLMClient` instances."""

from __future__ import annotations

from ...config import Config
from ..port.BaseLLMClient import BaseLLMClient


class LLMClientFactory:
    """Pick the right LLM adapter based on a ``client_type`` string.

    Keeping this in one place lets the rest of the codebase stay decoupled
    from concrete LLM implementations — to add OpenAI / Anthropic / Bedrock
    later, you only edit this file and the corresponding adapter.
    """

    @staticmethod
    def create_llm_client(config: Config, client_type: str) -> BaseLLMClient:
        """Construct a client by name.

        Args:
            config: the askistanbul Config singleton (or a stand-in for tests).
            client_type: ``"ollama"`` | ``"openrouter"``.

        Raises:
            ValueError: if ``client_type`` isn't recognised.
        """
        client_type = client_type.lower().strip()

        if client_type == "ollama":
            from ..adapter.OllamaClient import OllamaClient
            return OllamaClient(model=config.ollama_model, base_url=config.ollama_base_url)

        if client_type == "openrouter":
            from ..adapter.OpenRouterClient import OpenRouterClient
            return OpenRouterClient(model=config.openrouter_model, api_key=config.openrouter_api_key)

        raise ValueError(
            f"unknown client_type {client_type!r}; expected one of: ollama, openrouter"
        )
