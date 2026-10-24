"""LLM provider abstraction.

* :class:`OpenAICompatibleProvider` – any hosted API exposing the OpenAI
  ``/chat/completions`` format (e.g. Groq, Google Gemini's OpenAI-compatible
  endpoint, OpenRouter, Together). Configured with ``LLM_BASE_URL``,
  ``LLM_MODEL`` and ``LLM_API_KEY``.
* :class:`OllamaProvider` – a locally running Ollama server.
* Offline demo mode has no provider: the response service returns retrieved
  passages verbatim and labels them as not AI-generated.

API keys are sent only in the Authorization header and never logged.
"""

from __future__ import annotations

import logging
import random
import time
from abc import ABC, abstractmethod
from typing import Callable

import requests

from config.settings import Settings

logger = logging.getLogger(__name__)
MAX_RETRY_AFTER_SECONDS = 20.0  # longest provider-requested wait before retrying


class LLMError(Exception):
    """Base class for provider errors. ``user_message`` is safe to display."""

    user_message = "The AI service failed unexpectedly. Please try again."


class MissingCredentialsError(LLMError):
    user_message = (
        "The AI provider is not configured (missing API key or model). Set LLM_API_KEY and "
        "LLM_MODEL in .env or Streamlit secrets, or switch to offline demo mode."
    )


class InvalidCredentialsError(MissingCredentialsError):
    user_message = (
        "The AI provider's API key is malformed: it contains spaces or non-ASCII characters "
        "(for example a pasted '…' or curly quote). Copy the key again into LLM_API_KEY."
    )


class AuthenticationError(LLMError):
    user_message = "The AI provider rejected the API key. Please check LLM_API_KEY."


class QuotaExceededError(LLMError):
    user_message = (
        "The AI provider's rate limit or free-tier quota has been reached. Please wait a "
        "minute and try again, or switch to offline demo mode."
    )
    retry_after: float | None = None  # seconds, from the provider's Retry-After header


class ProviderTimeoutError(LLMError):
    user_message = "The AI provider took too long to respond. Please try again."


class ProviderUnavailableError(LLMError):
    user_message = (
        "The AI provider is currently unreachable or having an outage. Please try again "
        "later, or switch to offline demo mode."
    )


class InvalidResponseError(LLMError):
    user_message = "The AI provider returned an unexpected response. Please try again."


Messages = list[dict[str, str]]


class LLMProvider(ABC):
    """Common interface for chat-completion providers."""

    name: str = "provider"
    model: str = ""

    @abstractmethod
    def generate(self, messages: Messages) -> str:
        """Return the assistant's reply text for ``messages``."""


class _HTTPProvider(LLMProvider):
    """Shared HTTP behaviour: timeouts, bounded retries and error mapping."""

    def __init__(
        self,
        timeout: float,
        max_retries: int,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.timeout = timeout
        self.max_retries = max(0, max_retries)
        self.session = session or requests.Session()
        self._sleep = sleep

    def _post(self, url: str, payload: dict, headers: dict[str, str]) -> dict:
        attempt = 0
        while True:
            try:
                resp = self.session.post(url, json=payload, headers=headers, timeout=self.timeout)
            except requests.Timeout as exc:
                error: LLMError = ProviderTimeoutError(str(exc))
            except requests.ConnectionError as exc:
                error = ProviderUnavailableError(f"connection failed: {exc.__class__.__name__}")
            else:
                if resp.status_code == 200:
                    try:
                        return resp.json()
                    except ValueError as exc:
                        raise InvalidResponseError("response was not JSON") from exc
                error = self._map_status(resp)
                if not isinstance(error, (QuotaExceededError, ProviderUnavailableError)):
                    raise error
            if attempt >= self.max_retries:
                raise error
            delay = min(8.0, (2**attempt) + random.uniform(0, 0.5))
            retry_after = getattr(error, "retry_after", None)
            if retry_after is not None:
                # Per-minute limits reset within seconds; a longer wait means a daily quota.
                if retry_after > MAX_RETRY_AFTER_SECONDS:
                    raise error
                delay = max(delay, retry_after)
            logger.warning("%s call failed (%s); retry %d in %.1fs", self.name, type(error).__name__, attempt + 1, delay)
            self._sleep(delay)
            attempt += 1

    @staticmethod
    def _map_status(resp: requests.Response) -> LLMError:
        code = resp.status_code
        detail = f"HTTP {code}"
        if code in (401, 403):
            return AuthenticationError(detail)
        if code == 429:
            quota = QuotaExceededError(detail)
            try:
                quota.retry_after = float(resp.headers.get("retry-after", ""))
            except (AttributeError, TypeError, ValueError):
                pass
            return quota
        if code in (404,):
            err = InvalidResponseError(detail + " (model or endpoint not found)")
            err.user_message = "The configured AI model or endpoint was not found. Check LLM_MODEL and LLM_BASE_URL."
            return err
        if code >= 500:
            return ProviderUnavailableError(detail)
        return InvalidResponseError(detail)


class OpenAICompatibleProvider(_HTTPProvider):
    """Hosted provider using the OpenAI chat-completions wire format."""

    name = "openai_compatible"

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        timeout: float = 45.0,
        max_retries: int = 2,
        temperature: float = 0.2,
        max_tokens: int = 700,
        **kwargs,
    ) -> None:
        api_key = (api_key or "").strip()
        if not api_key or not model:
            raise MissingCredentialsError("LLM_API_KEY and LLM_MODEL are required")
        # HTTP headers must be ASCII; a pasted ellipsis or quote would otherwise crash the request.
        if not api_key.isascii() or not api_key.isprintable() or any(c.isspace() for c in api_key):
            raise InvalidCredentialsError("LLM_API_KEY contains invalid characters")
        super().__init__(timeout, max_retries, **kwargs)
        self._api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens

    def __repr__(self) -> str:  # never expose the key
        return f"OpenAICompatibleProvider(model={self.model!r}, base_url={self.base_url!r})"

    def generate(self, messages: Messages) -> str:
        data = self._post(
            f"{self.base_url}/chat/completions",
            {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
            {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
        )
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise InvalidResponseError("missing choices[0].message.content") from exc
        if not isinstance(content, str) or not content.strip():
            raise InvalidResponseError("empty completion")
        return content.strip()


class OllamaProvider(_HTTPProvider):
    """Local Ollama server (``ollama serve``) using ``/api/chat``."""

    name = "ollama"

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout: float = 300.0,
        max_retries: int = 1,
        temperature: float = 0.2,
        max_tokens: int = 700,
        num_ctx: int = 8192,
        **kwargs,
    ) -> None:
        if not model:
            raise MissingCredentialsError("OLLAMA_MODEL is required")
        super().__init__(timeout, max_retries, **kwargs)
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Set explicitly: when a prompt exceeds Ollama's context window the
        # oldest tokens (our system rules) are dropped silently.
        self.num_ctx = num_ctx

    def generate(self, messages: Messages) -> str:
        options = {"temperature": self.temperature, "num_predict": self.max_tokens, "num_ctx": self.num_ctx}
        try:
            data = self._post(
                f"{self.base_url}/api/chat",
                {"model": self.model, "messages": messages, "stream": False, "options": options},
                {"Content-Type": "application/json"},
            )
        except ProviderUnavailableError as exc:
            exc.user_message = (
                f"Cannot reach Ollama at {self.base_url}. Start it with `ollama serve` and "
                f"pull the model with `ollama pull {self.model}`."
            )
            raise
        prompt_tokens = data.get("prompt_eval_count") if isinstance(data, dict) else None
        if isinstance(prompt_tokens, int) and prompt_tokens >= self.num_ctx - self.max_tokens:
            logger.warning(
                "Ollama prompt used %d of %d context tokens; it may have been truncated. Increase OLLAMA_NUM_CTX.",
                prompt_tokens, self.num_ctx,
            )
        try:
            content = data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise InvalidResponseError("missing message.content") from exc
        if not isinstance(content, str) or not content.strip():
            raise InvalidResponseError("empty completion")
        return content.strip()


def create_provider(settings: Settings) -> LLMProvider | None:
    """Create the configured provider, or ``None`` for offline demo mode.

    Raises :class:`MissingCredentialsError` if a hosted provider is selected
    without credentials.
    """
    if settings.llm_provider == "offline":
        return None
    if settings.llm_provider == "ollama":
        return OllamaProvider(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            timeout=settings.ollama_timeout_seconds,
            max_retries=min(settings.llm_max_retries, 1),
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            num_ctx=settings.ollama_num_ctx,
        )
    return OpenAICompatibleProvider(
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
