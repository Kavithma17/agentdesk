"""Provider-agnostic LLM layer. The rest of the app talks to `LLMClient`, never to a vendor SDK."""

from typing import Protocol

from google import genai
from google.genai import types

from app.core.config import get_settings

# Free-tier models often return 429 (rate limit) or 503 (overloaded); these are worth retrying.
RETRYABLE_STATUS_CODES = [429, 500, 503]


class LLMClient(Protocol):
    def generate(self, prompt: str, system: str | None = None) -> str: ...


class GeminiClient:
    def __init__(self, api_key: str, model: str):
        retry = types.HttpRetryOptions(
            attempts=4, initial_delay=1.0, max_delay=20.0, http_status_codes=RETRYABLE_STATUS_CODES
        )
        self._client = genai.Client(
            api_key=api_key, http_options=types.HttpOptions(retry_options=retry)
        )
        self._model = model

    def generate(self, prompt: str, system: str | None = None) -> str:
        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=types.GenerateContentConfig(system_instruction=system),
        )
        return response.text or ""


def get_llm() -> LLMClient:
    settings = get_settings()
    if settings.llm_provider == "gemini":
        return GeminiClient(settings.gemini_api_key.get_secret_value(), settings.llm_model)
    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
