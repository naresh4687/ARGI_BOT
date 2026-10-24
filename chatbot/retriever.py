"""Hybrid semantic + lexical retrieval with a code-enforced relevance threshold."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from chatbot.language import detect_language
from knowledge_base.embeddings import Embedder
from knowledge_base.ingest import Chunk
from knowledge_base.lexical import LexicalIndex
from knowledge_base.vector_store import FaissVectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievedPassage:
    """A passage that passed the relevance threshold, with a citation label."""

    label: str  # "S1", "S2", ... used for in-answer citations
    chunk: Chunk
    score: float  # hybrid relevance score
    dense_score: float = 0.0
    lexical_score: float = 0.0


@dataclass(frozen=True)
class LanguageProfile:
    """Retrieval parameters for queries written in one language.

    ``dense_weight`` mixes the centred dense similarity with the lexical
    similarity; ``min_relevance`` is the threshold on the mixed score. Values
    were chosen from the measured sweep in ``evaluation/results``.
    """

    dense_weight: float
    min_relevance: float


DEFAULT_PROFILES = {
    "en": LanguageProfile(dense_weight=0.5, min_relevance=0.25),
    "ta": LanguageProfile(dense_weight=0.2, min_relevance=0.18),
}


class Retriever:
    """Embeds the query, scores every chunk with
    ``w * dense + (1 - w) * lexical`` and keeps only hits above the
    language's threshold. Passages below the threshold never reach the LLM.

    A passage from a different document than the best hit must also score at
    least ``relative_cutoff`` times the best score. Weaker neighbours are
    usually about another disease or practice, and small models tend to
    blend them into the answer (e.g. sheath blight symptoms reported as
    bacterial leaf blight symptoms).

    The profile is chosen from the language the *question* is written in
    (not the requested answer language), because that determines how the
    query text matches the corpus.
    """

    def __init__(
        self,
        embedder: Embedder,
        store: FaissVectorStore,
        top_k: int = 4,
        profiles: dict[str, LanguageProfile] | None = None,
        max_per_document: int = 2,
        relative_cutoff: float = 0.85,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.top_k = top_k
        self.profiles = profiles or DEFAULT_PROFILES
        self.max_per_document = max_per_document
        self.relative_cutoff = relative_cutoff
        self.lexical = LexicalIndex([c.search_text for c in store.chunks])

    def profile_for(self, query: str) -> tuple[str, LanguageProfile]:
        language = detect_language(query)
        language = language if language in self.profiles else "en"
        return language, self.profiles[language]

    def retrieve(self, query: str, language: str | None = None) -> list[RetrievedPassage]:
        """Return up to ``top_k`` relevant passages for ``query``.

        ``language`` is the requested answer language; it is only used to
        break near-ties in favour of passages already in that language.
        """
        query = query.strip()
        if not query or self.store.size == 0:
            return []
        query_lang, profile = self.profile_for(query)
        raw = self.embedder.embed([query])[0]
        dense = self.store.all_scores(self.store.transform_query(raw, query_lang))
        lexical = np.zeros(self.store.size, dtype="float32")
        for idx, score in self.lexical.scores(query).items():
            lexical[idx] = score
        hybrid = profile.dense_weight * dense + (1 - profile.dense_weight) * lexical

        order = np.argsort(-hybrid)
        selected: list[RetrievedPassage] = []
        per_doc: dict[str, int] = {}
        candidates = [i for i in order[: self.top_k * 6] if hybrid[i] >= profile.min_relevance]
        # Prefer the answer language among near-equal candidates (within 0.02).
        candidates.sort(
            key=lambda i: hybrid[i] + (0.02 if language and self.store.chunks[i].language == language else 0.0),
            reverse=True,
        )
        best = max((float(hybrid[i]) for i in candidates), default=0.0)
        for i in candidates:
            chunk = self.store.chunks[i]
            if per_doc.get(chunk.document_id, 0) >= self.max_per_document:
                continue
            if (
                selected
                and chunk.document_id != selected[0].chunk.document_id
                and hybrid[i] < self.relative_cutoff * best
            ):
                continue
            per_doc[chunk.document_id] = per_doc.get(chunk.document_id, 0) + 1
            selected.append(
                RetrievedPassage(
                    label=f"S{len(selected) + 1}",
                    chunk=chunk,
                    score=float(hybrid[i]),
                    dense_score=float(dense[i]),
                    lexical_score=float(lexical[i]),
                )
            )
            if len(selected) >= self.top_k:
                break
        logger.info(
            "query_lang=%s retrieved=%d best=%.3f threshold=%.2f",
            query_lang,
            len(selected),
            float(hybrid[order[0]]),
            profile.min_relevance,
        )
        return selected
