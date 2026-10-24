"""AgriBot AI – Streamlit entry point.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import replace

import streamlit as st

st.set_page_config(page_title="AgriBot AI", page_icon="🌾", layout="centered")

from chatbot.bootstrap import build_service, load_or_build_store  # noqa: E402
from chatbot.llm_client import MissingCredentialsError  # noqa: E402
from chatbot.response_service import ResponseService  # noqa: E402
from config.settings import Settings, configure_logging  # noqa: E402
from knowledge_base.embeddings import SentenceTransformerEmbedder  # noqa: E402
from knowledge_base.ingest import DocumentLoadError  # noqa: E402
from knowledge_base.vector_store import FaissVectorStore  # noqa: E402
from ui import chat, components, info_pages  # noqa: E402
from ui.sidebar import render_sidebar  # noqa: E402

configure_logging()
logger = logging.getLogger("agribot.app")


@st.cache_resource(show_spinner="Loading the multilingual embedding model…")
def get_embedder(model_name: str) -> SentenceTransformerEmbedder:
    embedder = SentenceTransformerEmbedder(model_name)
    embedder.embed(["warm-up"])  # load weights now, not on the first question
    return embedder


@st.cache_resource(show_spinner="Loading the agricultural knowledge base…")
def get_store(_settings: Settings, model_name: str, index_dir: str) -> FaissVectorStore:
    return load_or_build_store(_settings, get_embedder(model_name))


@st.cache_resource(show_spinner=False)
def get_service(_settings: Settings, signature: str, model_name: str, index_dir: str) -> ResponseService:
    return build_service(_settings, get_embedder(model_name), get_store(_settings, model_name, index_dir))


def provider_signature(s: Settings) -> str:
    """Cache key covering every provider setting, so changed secrets/env take
    effect. The API key is included only as a hash."""
    key_hash = hashlib.sha256(s.llm_api_key.encode()).hexdigest()[:12] if s.llm_api_key else ""
    parts = (
        s.llm_provider, s.llm_base_url, s.llm_model, key_hash, s.ollama_base_url, s.ollama_model, s.ollama_num_ctx, s.ollama_timeout_seconds,
        s.llm_timeout_seconds, s.llm_max_retries, s.llm_temperature, s.llm_max_tokens, s.top_k,
        s.dense_weight_en, s.dense_weight_ta, s.min_relevance_en, s.min_relevance_ta, s.max_question_chars,
    )
    return "|".join(map(str, parts))


def main() -> None:
    components.inject_styles()
    chat.init_state()
    settings = Settings.from_env()

    # ---- Knowledge base ---------------------------------------------------
    try:
        store = get_store(settings, settings.embedding_model, str(settings.index_dir))
    except DocumentLoadError as exc:
        components.header()
        st.error(
            "The knowledge base is empty, so AgriBot cannot answer yet.\n\n"
            f"Details: {exc}\n\nRun `python -m knowledge_base.fetch_sources --allow-insecure-tls` and then "
            "`python -m knowledge_base.ingest`, add verified documents to `data/raw/`, or set "
            "`AUTO_FETCH_SOURCES=1` to download the sources at start-up."
        )
        st.stop()
    except Exception as exc:  # model download failure, corrupt files, ...
        logger.exception("Knowledge base failed to load")
        components.header()
        st.error(f"The knowledge base could not be loaded: {exc.__class__.__name__}. See the server log.")
        st.stop()

    # ---- Provider (falls back to offline demo on missing credentials) -------
    provider_warning = None
    offline_forced = st.session_state.get("force_offline", False)
    effective = replace(settings, llm_provider="offline") if offline_forced else settings
    try:
        service = get_service(effective, provider_signature(effective), settings.embedding_model, str(settings.index_dir))
    except MissingCredentialsError as exc:
        provider_warning = exc.user_message
        effective = replace(settings, llm_provider="offline")
        service = get_service(effective, provider_signature(effective), settings.embedding_model, str(settings.index_dir))

    if service.provider is None:
        mode_label = "Offline demo (no AI generation)"
    else:
        mode_label = f"AI · {service.provider.name} · `{service.provider.model}`"
    docs = store.info.get("documents", "?")
    kb_summary = f"{docs} documents, {store.size} passages"

    state = render_sidebar(mode_label, kb_summary)
    unverified = store.info.get("tls_unverified_documents", 0)
    if unverified:
        st.sidebar.warning(
            f"{unverified} of {docs} source documents were downloaded without HTTPS certificate "
            "verification, because the TNAU server's certificate check failed. Their text could "
            "have been altered in transit. Check important advice against the linked TNAU pages.",
            icon="⚠️",
        )
    if settings.llm_provider != "offline" and provider_warning is None:
        st.sidebar.toggle(
            "Offline demo mode",
            key="force_offline",
            help="Show retrieved passages only. No request is sent to the AI provider.",
        )

    components.header()
    if provider_warning:
        st.warning(provider_warning + " Running in offline demo mode.", icon="🔌")

    if state.page == "chat":
        chat.render_chat(service.answer, state.language)
    elif state.page == "help":
        info_pages.render_help(state.language)
    elif state.page == "about":
        info_pages.render_about(state.language)
    else:
        view = effective.redacted()
        view["effective_mode"] = mode_label
        info_pages.render_project_info(view, store.info)


main()
