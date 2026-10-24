"""Wiring helpers that build the pipeline from settings."""

from __future__ import annotations

import logging

from chatbot.llm_client import create_provider
from chatbot.response_service import ResponseService
from chatbot.retriever import LanguageProfile, Retriever
from config.settings import Settings
from knowledge_base.embeddings import SentenceTransformerEmbedder
from knowledge_base.fetch_sources import fetch_all
from knowledge_base.ingest import DocumentLoadError, build_knowledge_base
from knowledge_base.vector_store import FaissVectorStore, IndexCorruptError, IndexNotFoundError

logger = logging.getLogger(__name__)


def load_or_build_store(settings: Settings, embedder: SentenceTransformerEmbedder) -> FaissVectorStore:
    """Load the persisted index; rebuild it from ``data/raw`` if it is missing,
    corrupt or was built with a different embedding model.

    The source documents are not stored in the public repository (licensing).
    With ``AUTO_FETCH_SOURCES=1`` an empty ``data/raw`` is filled from the
    manifest first, which is how a fresh cloud deployment gets its corpus."""
    try:
        return FaissVectorStore.load(settings.index_dir, expected_model=settings.embedding_model)
    except (IndexNotFoundError, IndexCorruptError) as exc:
        logger.warning("%s - rebuilding the index from %s", exc, settings.raw_data_dir)
    try:
        build_knowledge_base(settings, embedder)
    except DocumentLoadError:
        if not settings.auto_fetch_sources:
            raise
        logger.warning("No source documents - downloading the starter corpus (AUTO_FETCH_SOURCES=1)")
        # The TNAU server intermittently serves an expired certificate; affected
        # files are marked tls_verified=false in their sidecar metadata.
        fetch_all(settings.raw_data_dir, allow_insecure_tls=True)
        build_knowledge_base(settings, embedder)
    return FaissVectorStore.load(settings.index_dir, expected_model=settings.embedding_model)


def build_service(
    settings: Settings,
    embedder: SentenceTransformerEmbedder | None = None,
    store: FaissVectorStore | None = None,
) -> ResponseService:
    """Create a ready-to-use :class:`ResponseService`.

    May raise ``MissingCredentialsError`` if a hosted provider is configured
    without credentials; callers decide whether to fall back to offline mode.
    """
    embedder = embedder or SentenceTransformerEmbedder(settings.embedding_model)
    store = store or load_or_build_store(settings, embedder)
    profiles = {
        "en": LanguageProfile(settings.dense_weight_en, settings.min_relevance_en),
        "ta": LanguageProfile(settings.dense_weight_ta, settings.min_relevance_ta),
    }
    retriever = Retriever(
        embedder, store, top_k=settings.top_k, profiles=profiles, relative_cutoff=settings.relative_cutoff
    )
    return ResponseService(
        retriever,
        create_provider(settings),
        max_question_chars=settings.max_question_chars,
        min_question_chars=settings.min_question_chars,
    )
