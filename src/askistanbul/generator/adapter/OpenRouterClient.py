"""OpenRouter LLM client using the raw HTTP API."""

from __future__ import annotations

import json
from typing import Any, Iterator

import requests

from ...config import config
from ..port.BaseLLMClient import BaseLLMClient


class OpenRouterClient(BaseLLMClient):
    """Client for OpenRouter that calls the REST API directly via ``requests``."""

    BASE_URL = "https://openrouter.ai/api/v1"
    CHAT_COMPLETIONS_PATH = "/chat/completions"
    DEFAULT_TIMEOUT = 60

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        extra_headers: dict[str, str] | None = None,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        super().__init__(model or config.openrouter_model or "qwen/qwen-2.5-72b-instruct")

        api_key = api_key or config.openrouter_api_key
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required")

        self.api_key = api_key
        self.timeout = timeout
        self.url = f"{self.BASE_URL}{self.CHAT_COMPLETIONS_PATH}"

        self.headers: dict[str, str] = {
            "Authorization": f"Bearer {api_key}",
        }
        referer = config.openrouter_referer
        title = config.openrouter_title
        if referer:
            self.headers["HTTP-Referer"] = referer
        if title:
            self.headers["X-Title"] = title
        if extra_headers:
            self.headers.update(extra_headers)

        print(f"✓ OpenRouterClient initialized (model={self.model})")

    def _post(self, payload: dict, stream: bool = False) -> requests.Response:
        resp = requests.post(
            url=self.url,
            headers=self.headers,
            json=payload,
            stream=stream,
            timeout=self.timeout,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"OpenRouter HTTP {resp.status_code}: {resp.text}")
        return resp

    def _build_payload(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_new_tokens: int,
        json_mode: bool = False,
        stream: bool = False,
    ) -> dict:
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_new_tokens,
        }
        if stream:
            payload["stream"] = True
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_new_tokens: int = 2000,
    ) -> str:
        payload = self._build_payload(messages, temperature, max_new_tokens)
        resp = self._post(payload)
        data = resp.json()
        if "choices" not in data:
            # OpenRouter returns error/rate-limit responses without "choices"
            error_msg = data.get("error", {}).get("message", str(data))
            raise RuntimeError(f"OpenRouter response missing 'choices': {error_msg}")
        return data["choices"][0]["message"].get("content") or ""

    def chat_json(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_new_tokens: int = 2000,
    ) -> dict[str, Any]:
        """Use OpenRouter's native JSON mode (``response_format=json_object``), then parse."""
        payload = self._build_payload(
            messages, temperature, max_new_tokens, json_mode=True
        )
        resp = self._post(payload)
        data = resp.json()
        content = data["choices"][0]["message"].get("content") or ""
        if not content:
            raise ValueError(
                f"OpenRouter returned an empty response. Full body: {data}"
            )
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Failed to parse OpenRouter JSON response: {exc}\nRaw: {content[:200]}"
            ) from exc

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_new_tokens: int = 2000,
    ) -> Iterator[str]:
        payload = self._build_payload(
            messages, temperature, max_new_tokens, stream=True
        )
        resp = self._post(payload, stream=True)
        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line or not raw_line.startswith("data:"):
                continue
            chunk = raw_line[len("data:"):].strip()
            if chunk == "[DONE]":
                break
            try:
                event = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            choices = event.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            content = delta.get("content")
            if content:
                yield content
