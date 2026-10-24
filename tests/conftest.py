"""Shared test fixtures.

Unit tests use a deterministic hashing embedder so they run offline and fast.
Tests that need the real embedding model / built index are marked ``model``
and skip automatically when it is not available.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chatbot.retriever import LanguageProfile, RetrievedPassage, Retriever  # noqa: E402
from knowledge_base.ingest import Chunk  # noqa: E402
from knowledge_base.lexical import char_ngrams  # noqa: E402
from knowledge_base.vector_store import FaissVectorStore  # noqa: E402


class HashingEmbedder:
    """Bag of hashed character n-grams -> L2-normalised 128-d vector."""

    model_name = "test-hashing-embedder"

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for row, text in enumerate(texts):
            for gram in char_ngrams(text, 3):
                h = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16)
                out[row, h % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-12)


SAMPLE_CHUNKS = [
    Chunk("blast:p0:0", "rice-blast", "Diseases of Rice: Blast",
          "Rice blast symptoms: spindle shaped spots with grey centre and dark brown margin on leaves.",
          source_url="https://example.org/rice-blast", language="en"),
    Chunk("drip:p0:0", "drip", "Irrigation Management: Drip Irrigation",
          "Drip irrigation saves water and delivers water directly to the root zone of plants.",
          source_url="https://example.org/drip", language="en"),
    Chunk("drip:p0:1", "drip", "Irrigation Management: Drip Irrigation",
          "Drip irrigation reduces weed growth because only the root zone is wetted.",
          source_url="https://example.org/drip", language="en"),
    Chunk("drip:p0:2", "drip", "Irrigation Management: Drip Irrigation",
          "Drip irrigation systems need filters to prevent clogging of emitters.",
          source_url="https://example.org/drip", language="en"),
    Chunk("blast-ta:p0:0", "rice-blast-ta", "நெல் நோய்கள்: குலைநோய்",
          "குலைநோய் அறிகுறிகள்: இலைகளின் மேல் கண் வடிவ புள்ளிகள் காணப்படும்.",
          source_url="https://example.org/rice-blast-ta", language="ta"),
    Chunk("pdf:p3:0", "guide-pdf", "Natural Pest Control Guide",
          "Companion planting deters pests: onions planted beside carrots deter carrot flies.",
          source_url=None, page_number=3, language="en"),
]


@pytest.fixture(autouse=True)
def _ignore_local_streamlit_secrets(monkeypatch):
    """Tests configure the app through env vars only; a developer's local
    .streamlit/secrets.toml (which may hold a real key) must not leak in."""
    import config.settings

    monkeypatch.setattr(config.settings, "_streamlit_secret", lambda name: None)


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder()


@pytest.fixture
def store(embedder) -> FaissVectorStore:
    vectors = embedder.embed([c.text for c in SAMPLE_CHUNKS])
    return FaissVectorStore.build(vectors, SAMPLE_CHUNKS, info={"embedding_model": embedder.model_name})


@pytest.fixture
def retriever(embedder, store) -> Retriever:
    profiles = {"en": LanguageProfile(0.5, 0.15), "ta": LanguageProfile(0.2, 0.10)}
    return Retriever(embedder, store, top_k=3, profiles=profiles)


def make_passages(*chunks: Chunk) -> list[RetrievedPassage]:
    return [RetrievedPassage(label=f"S{i + 1}", chunk=c, score=0.6) for i, c in enumerate(chunks)]


class StubRetriever:
    """Returns a fixed list of passages and records queries."""

    def __init__(self, passages: list[RetrievedPassage]) -> None:
        self.passages = passages
        self.queries: list[str] = []

    def retrieve(self, query, language=None):
        self.queries.append(query)
        return list(self.passages)


class StubProvider:
    """Fake LLM provider: returns a canned reply or raises an error."""

    name = "stub"
    model = "stub-model"

    def __init__(self, reply: str = "", error: Exception | None = None) -> None:
        self.reply = reply
        self.error = error
        self.calls: list[list[dict]] = []

    def generate(self, messages):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return self.reply


def _real_index_available() -> bool:
    from config.settings import Settings

    s = Settings()
    if not FaissVectorStore.exists(s.index_dir):
        return False
    try:
        from huggingface_hub import try_to_load_from_cache

        return try_to_load_from_cache(s.embedding_model, "config.json") is not None
    except Exception:
        return False


def pytest_configure(config):
    config.addinivalue_line("markers", "model: needs the real embedding model and built index")
    config.addinivalue_line("markers", "integration: calls a real external LLM provider")


def pytest_collection_modifyitems(config, items):
    if _real_index_available():
        return
    skip = pytest.mark.skip(reason="real embedding model or built index not available")
    for item in items:
        if "model" in item.keywords:
            item.add_marker(skip)
