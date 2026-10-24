"""Embedding model wrappers."""

from __future__ import annotations

import logging
from typing import Protocol, Sequence

import numpy as np

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    """Anything that turns texts into L2-normalised float32 vectors."""

    model_name: str

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Multilingual sentence-transformers embedder (CPU by default).

    The model is loaded lazily on first use so that importing this module is
    cheap. Vectors are normalised so inner product equals cosine similarity.
    """

    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 32) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model %s on %s", self.model_name, self.device)
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype="float32")
        vectors = self._load().encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype="float32")
