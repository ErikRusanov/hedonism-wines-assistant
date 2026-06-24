"""LLM access layer (OpenRouter-backed)."""

from hedonism_assistant.llm.openrouter import OpenRouterClient, get_openrouter_client

__all__ = ["OpenRouterClient", "get_openrouter_client"]
