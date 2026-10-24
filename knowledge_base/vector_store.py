"""FAISS vector store with persistence of vectors, chunk metadata and build info."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from knowledge_base.ingest import Chunk

logger = logging.getLogger(__name__)

INDEX_FILE = "index.faiss"
CHUNKS_FILE = "chunks.jsonl"
INFO_FILE = "index_info.json"
CENTROIDS_FILE = "language_centroids.npz"


class IndexNotFoundError(FileNotFoundError):
    """Raised when no persisted index exists in the given directory."""


class IndexCorruptError(RuntimeError):
    """Raised when the persisted index files are inconsistent or unreadable."""


@dataclass
class SearchHit:
    """One similarity-search result."""

    chunk: Chunk
    score: float  # cosine similarity in [-1, 1]


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


class FaissVectorStore:
    """Exact inner-product FAISS index over normalised embeddings.

    ``IndexFlatIP`` is exact and fast enough for a few thousand chunks, which
    keeps behaviour predictable for a small curated corpus.

    Per-language centering: the multilingual MiniLM model places almost all
    Tamil text in one narrow region, so unrelated Tamil sentences score ~0.8
    against each other. Subtracting each language's mean vector (from the
    corpus) before normalising removes this shared component; queries are
    centred with the mean of their own language via :meth:`transform_query`.
    """

    def __init__(
        self,
        index,
        chunks: list[Chunk],
        info: dict[str, Any] | None = None,
        centroids: dict[str, np.ndarray] | None = None,
    ) -> None:
        if index.ntotal != len(chunks):
            raise IndexCorruptError(
                f"index has {index.ntotal} vectors but {len(chunks)} chunks were provided"
            )
        self.index = index
        self.chunks = chunks
        self.info = info or {}
        self.centroids = centroids or {}

    @property
    def size(self) -> int:
        return len(self.chunks)

    @property
    def dimension(self) -> int:
        return int(self.index.d)

    def transform_query(self, vector: np.ndarray, language: str | None) -> np.ndarray:
        """Centre a raw query embedding the same way the corpus was centred."""
        vector = np.asarray(vector, dtype="float32")
        centroid = self.centroids.get(language or "") if self.centroids else None
        if centroid is None and self.centroids:
            centroid = self.centroids.get("en", next(iter(self.centroids.values())))
        if centroid is not None:
            vector = vector - centroid
        return _normalize(vector).astype("float32")

    @classmethod
    def build(
        cls,
        vectors: np.ndarray,
        chunks: Sequence[Chunk],
        info: dict[str, Any] | None = None,
        center_by_language: bool = True,
    ) -> "FaissVectorStore":
        """Create an index from vectors aligned with ``chunks``."""
        import faiss

        vectors = np.ascontiguousarray(vectors, dtype="float32")
        if vectors.ndim != 2 or len(vectors) != len(chunks) or len(chunks) == 0:
            raise ValueError("vectors must be a non-empty 2-D array aligned with chunks")
        centroids: dict[str, np.ndarray] = {}
        if center_by_language:
            langs = np.array([c.language or "unknown" for c in chunks])
            centroids = {str(l): vectors[langs == l].mean(axis=0) for l in sorted(set(langs))}
            vectors = np.ascontiguousarray(
                _normalize(vectors - np.stack([centroids[l] for l in langs])), dtype="float32"
            )
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        meta = dict(info or {})
        meta.update(
            {
                "dimension": int(vectors.shape[1]),
                "chunk_count": len(chunks),
                "centered_languages": sorted(centroids),
                "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        )
        return cls(index, list(chunks), meta, centroids)

    def all_scores(self, query_vector: np.ndarray) -> np.ndarray:
        """Similarity of a (transformed) query to every chunk, in chunk order."""
        vectors = self.index.reconstruct_n(0, self.size)
        return vectors @ np.asarray(query_vector, dtype="float32").reshape(-1)

    def search(self, query_vector: np.ndarray, k: int = 5) -> list[SearchHit]:
        """Return up to ``k`` hits sorted by descending cosine similarity."""
        if self.size == 0:
            return []
        query = np.ascontiguousarray(np.atleast_2d(query_vector), dtype="float32")
        if query.shape[1] != self.dimension:
            raise ValueError(
                f"query dimension {query.shape[1]} does not match index dimension {self.dimension}"
            )
        k = max(1, min(k, self.size))
        scores, ids = self.index.search(query, k)
        return [
            SearchHit(chunk=self.chunks[i], score=float(s))
            for s, i in zip(scores[0], ids[0])
            if i >= 0
        ]

    def save(self, directory: Path) -> None:
        """Persist index, chunk metadata and build info to ``directory``."""
        import faiss

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / INDEX_FILE))
        with open(directory / CHUNKS_FILE, "w", encoding="utf-8") as fh:
            for chunk in self.chunks:
                fh.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        (directory / INFO_FILE).write_text(json.dumps(self.info, indent=2), encoding="utf-8")
        np.savez(directory / CENTROIDS_FILE, **self.centroids)
        logger.info("Saved index with %d chunks to %s", self.size, directory)

    @classmethod
    def exists(cls, directory: Path) -> bool:
        directory = Path(directory)
        return all((directory / f).exists() for f in (INDEX_FILE, CHUNKS_FILE, INFO_FILE))

    @classmethod
    def load(cls, directory: Path, expected_model: str | None = None) -> "FaissVectorStore":
        """Load a persisted store. Raises IndexNotFoundError / IndexCorruptError."""
        import faiss

        directory = Path(directory)
        if not cls.exists(directory):
            raise IndexNotFoundError(f"No knowledge-base index found in {directory}")
        try:
            index = faiss.read_index(str(directory / INDEX_FILE))
            with open(directory / CHUNKS_FILE, encoding="utf-8") as fh:
                chunks = [Chunk.from_dict(json.loads(line)) for line in fh if line.strip()]
            info = json.loads((directory / INFO_FILE).read_text(encoding="utf-8"))
            centroids: dict[str, np.ndarray] = {}
            if info.get("centered_languages"):
                with np.load(directory / CENTROIDS_FILE) as data:
                    centroids = {k: data[k].astype("float32") for k in data.files}
        except (RuntimeError, OSError, ValueError, json.JSONDecodeError, TypeError) as exc:
            raise IndexCorruptError(f"Knowledge-base index is unreadable: {exc}") from exc
        if expected_model and info.get("embedding_model") not in (None, expected_model):
            raise IndexCorruptError(
                f"Index was built with {info.get('embedding_model')!r} but "
                f"{expected_model!r} is configured. Rebuild the index."
            )
        return cls(index, chunks, info, centroids)
