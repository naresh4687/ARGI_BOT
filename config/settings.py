"""Application configuration.

Values are read (in priority order) from:
1. Real environment variables.
2. A local ``.env`` file (development only, never committed).
3. Streamlit secrets (``.streamlit/secrets.toml`` locally or the hosting
   platform's secrets manager).
4. The defaults defined below.

Secrets are never logged. Use :meth:`Settings.redacted` when printing config.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:  # optional dependency for local development
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)
except ImportError:  # pragma: no cover
    pass

VALID_PROVIDERS = ("openai_compatible", "ollama", "offline")
_SECRET_FIELDS = {"llm_api_key"}


def _streamlit_secret(name: str) -> Any:
    """Return a Streamlit secret if Streamlit and a secrets file are available."""
    try:
        import streamlit as st

        if name in st.secrets:
            return st.secrets[name]
    except Exception:  # no secrets file, or not running under Streamlit
        return None
    return None


def _get(name: str, default: Any) -> Any:
    value = os.environ.get(name)
    if value is None or value == "":
        value = _streamlit_secret(name)
    if value is None or value == "":
        return default
    if isinstance(default, bool):
        return str(value).strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return str(value)


@dataclass(frozen=True)
class Settings:
    """Typed, immutable application settings."""

    # --- LLM provider -------------------------------------------------------
    llm_provider: str = "offline"
    llm_api_key: str = ""
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = ""
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gemma3:4b"
    ollama_num_ctx: int = 8192
    ollama_timeout_seconds: float = 300.0  # CPU-only generation can be slow
    llm_timeout_seconds: float = 45.0
    llm_max_retries: int = 2
    llm_temperature: float = 0.2
    llm_max_tokens: int = 700

    # --- Embeddings & retrieval ---------------------------------------------
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    top_k: int = 4
    # Hybrid retrieval (see chatbot/retriever.py and evaluation/results):
    # score = dense_weight * centred_dense + (1 - dense_weight) * lexical
    dense_weight_en: float = 0.5
    dense_weight_ta: float = 0.2
    min_relevance_en: float = 0.25
    min_relevance_ta: float = 0.18
    # Passages from other documents must score >= this fraction of the best hit.
    relative_cutoff: float = 0.85
    chunk_size: int = 400  # characters, English/Latin text
    chunk_size_ta: int = 300  # Tamil needs ~1.4x more tokens per character
    chunk_overlap: int = 70

    # --- Paths ----------------------------------------------------------------
    raw_data_dir: Path = field(default=PROJECT_ROOT / "data" / "raw")
    # Download the manifest sources when data/raw is empty (cloud deployments).
    auto_fetch_sources: bool = False
    index_dir: Path = field(default=PROJECT_ROOT / "data" / "processed")

    # --- Input limits ----------------------------------------------------------
    max_question_chars: int = 1000
    min_question_chars: int = 3

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from environment variables / Streamlit secrets."""
        d = cls()
        provider = _get("LLM_PROVIDER", d.llm_provider).strip().lower()
        if provider not in VALID_PROVIDERS:
            logging.getLogger(__name__).warning(
                "Unknown LLM_PROVIDER %r; falling back to 'offline'.", provider
            )
            provider = "offline"
        return cls(
            llm_provider=provider,
            llm_api_key=_get("LLM_API_KEY", d.llm_api_key),
            llm_base_url=_get("LLM_BASE_URL", d.llm_base_url).rstrip("/"),
            llm_model=_get("LLM_MODEL", d.llm_model),
            ollama_base_url=_get("OLLAMA_BASE_URL", d.ollama_base_url).rstrip("/"),
            ollama_model=_get("OLLAMA_MODEL", d.ollama_model),
            ollama_num_ctx=_get("OLLAMA_NUM_CTX", d.ollama_num_ctx),
            ollama_timeout_seconds=_get("OLLAMA_TIMEOUT_SECONDS", d.ollama_timeout_seconds),
            llm_timeout_seconds=_get("LLM_TIMEOUT_SECONDS", d.llm_timeout_seconds),
            llm_max_retries=_get("LLM_MAX_RETRIES", d.llm_max_retries),
            llm_temperature=_get("LLM_TEMPERATURE", d.llm_temperature),
            llm_max_tokens=_get("LLM_MAX_TOKENS", d.llm_max_tokens),
            embedding_model=_get("EMBEDDING_MODEL", d.embedding_model),
            top_k=_get("TOP_K", d.top_k),
            dense_weight_en=_get("DENSE_WEIGHT_EN", d.dense_weight_en),
            dense_weight_ta=_get("DENSE_WEIGHT_TA", d.dense_weight_ta),
            min_relevance_en=_get("MIN_RELEVANCE_EN", d.min_relevance_en),
            min_relevance_ta=_get("MIN_RELEVANCE_TA", d.min_relevance_ta),
            relative_cutoff=_get("RELATIVE_CUTOFF", d.relative_cutoff),
            chunk_size=_get("CHUNK_SIZE", d.chunk_size),
            chunk_size_ta=_get("CHUNK_SIZE_TA", d.chunk_size_ta),
            chunk_overlap=_get("CHUNK_OVERLAP", d.chunk_overlap),
            raw_data_dir=Path(_get("RAW_DATA_DIR", str(d.raw_data_dir))),
            auto_fetch_sources=_get("AUTO_FETCH_SOURCES", d.auto_fetch_sources),
            index_dir=Path(_get("INDEX_DIR", str(d.index_dir))),
            max_question_chars=_get("MAX_QUESTION_CHARS", d.max_question_chars),
        )

    def redacted(self) -> dict[str, Any]:
        """Return settings as a dict with secrets masked (safe to log/display)."""
        data = asdict(self)
        for key in _SECRET_FIELDS:
            data[key] = "***set***" if data[key] else "(not set)"
        return {k: str(v) if isinstance(v, Path) else v for k, v in data.items()}


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging once. Never log secrets or full prompts."""
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    # Third-party HTTP loggers can print full URLs/headers at DEBUG level.
    for noisy in ("urllib3", "httpx", "httpx2", "httpcore", "sentence_transformers", "faiss.loader"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
