"""Character n-gram TF-IDF index used alongside dense retrieval.

Tamil is agglutinative (``பாசனத்தின்``, ``பாசனம்`` share the stem ``பாசன``),
so word-level matching misses many relevant passages. Character 4-grams taken
inside each word match shared stems in both Tamil and English without a
language-specific stemmer.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Sequence

_WORD = re.compile(r"[0-9A-Za-z஀-௿]+")
NGRAM = 4


def char_ngrams(text: str, n: int = NGRAM) -> list[str]:
    """Return within-word character n-grams with boundary markers."""
    grams: list[str] = []
    for word in _WORD.findall(text.lower()):
        word = f"#{word}#"
        if len(word) <= n:
            grams.append(word)
        else:
            grams.extend(word[i : i + n] for i in range(len(word) - n + 1))
    return grams


class LexicalIndex:
    """Sparse TF-IDF (log-scaled tf, smoothed idf) with cosine scoring."""

    def __init__(self, texts: Sequence[str]) -> None:
        counts = [Counter(char_ngrams(t)) for t in texts]
        df: Counter[str] = Counter()
        for c in counts:
            df.update(c.keys())
        n_docs = len(counts)
        self.idf = {g: math.log((n_docs + 1) / (n + 0.5)) for g, n in df.items()}
        self.postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for doc_id, c in enumerate(counts):
            for gram, weight in self._weights(c).items():
                self.postings[gram].append((doc_id, weight))
        self.size = n_docs

    def _weights(self, counts: Counter[str]) -> dict[str, float]:
        raw = {g: (1 + math.log(c)) * self.idf[g] for g, c in counts.items() if g in self.idf}
        norm = math.sqrt(sum(v * v for v in raw.values())) or 1.0
        return {g: v / norm for g, v in raw.items()}

    def scores(self, query: str) -> dict[int, float]:
        """Cosine similarity between ``query`` and every matching document."""
        result: dict[int, float] = defaultdict(float)
        for gram, q_weight in self._weights(Counter(char_ngrams(query))).items():
            for doc_id, d_weight in self.postings.get(gram, ()):
                result[doc_id] += q_weight * d_weight
        return dict(result)
