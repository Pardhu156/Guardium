"""Ollama chat client for Stage 4.0."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import requests

from aegisvault.agent_runtime.exceptions import OllamaRuntimeError


@dataclass(slots=True)
class OllamaChatResult:
    payload: dict[str, Any]
    latency_ms: float


class OllamaChatClient:
    """Small HTTP client for Ollama's local chat API."""

    def __init__(
        self,
        *,
        model: str = "qwen3:4b-instruct",
        base_url: str = "http://localhost:11434",
        timeout_seconds: float = 60,
        temperature: float = 0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature

    def chat(self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> OllamaChatResult:
        started = time.perf_counter()
        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if tools:
            request["tools"] = tools
        try:
            response = requests.post(f"{self.base_url}/api/chat", json=request, timeout=self.timeout_seconds)
            response.raise_for_status()
            payload = response.json()
        except requests.Timeout as exc:
            raise OllamaRuntimeError(f"Ollama timed out after {self.timeout_seconds} seconds") from exc
        except requests.RequestException as exc:
            raise OllamaRuntimeError(f"Ollama chat request failed: {exc}") from exc
        except ValueError as exc:
            raise OllamaRuntimeError("Ollama returned non-JSON response") from exc
        return OllamaChatResult(payload=payload, latency_ms=(time.perf_counter() - started) * 1000)

    def list_models(self) -> list[str]:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout_seconds)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise OllamaRuntimeError(f"failed to list Ollama models: {exc}") from exc
        return [item.get("name", "") for item in payload.get("models", []) if item.get("name")]
