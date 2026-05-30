

from askistanbul.config import Config

from ..port.BaseLLMClient import BaseLLMClient


class LLMClientFactory:
    """
    Factory class for creating LLM client instances based on the configuration.
    This allows the rest of the code to be decoupled from specific LLM implementations, so they can be easily swapped out or modified without affecting the rest of the system.
    """

    @staticmethod
    def create_llm_client(config : Config, client_type: str) -> BaseLLMClient:
        """
        Create an LLM client instance based on the provided configuration.
        Currently, this factory only creates an OllamaClient, but it can be extended in the future to support other LLM clients (e.g., OpenAIClient, AnthropicClient) based on the presence of API keys or other config parameters.
        """

        from ..adapter.OllamaClient import OllamaClient

        # For now, we always return an OllamaClient since it's the primary local LLM we're using.
        # In the future, we could add logic here to check for API keys and return different clients accordingly.
        return OllamaClient(base_url=config.ollama_base_url, model=config.ollama_model)