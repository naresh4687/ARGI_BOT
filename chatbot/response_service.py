"""End-to-end RAG pipeline: validate → retrieve → prompt → generate → verify.

UI code calls :meth:`ResponseService.answer` and renders the returned
:class:`ChatResponse`; no AI logic lives in the UI layer.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

from chatbot.language import detect_language, language_name, t
from chatbot.llm_client import LLMError, LLMProvider
from chatbot.prompts import build_messages
from chatbot.retriever import RetrievedPassage, Retriever
from chatbot.safety import is_high_risk, neutralize_passage, validate_citations, validate_question

logger = logging.getLogger(__name__)

# Statuses
ANSWERED = "answered"  # LLM answer with at least one valid citation
UNVERIFIED = "unverified"  # LLM answer without valid citations
INSUFFICIENT = "insufficient"  # nothing relevant enough was retrieved
OFFLINE = "offline"  # extractive offline demo answer
INVALID_INPUT = "invalid_input"
ERROR = "error"

_QUANTITY_RE = re.compile(
    r"\d+(?:[.,]\d+)?\s*(?:-|–|to)?\s*\d*(?:[.,]\d+)?\s*"
    r"(?:kg|g|gm|mg|ml|l|lit(?:re|er)s?|%|ppm|cm|mm|m|ha|acre|days?|கிலோ|கிராம்|மி\.?லி|லிட்டர்|நாட்கள்)\b",
    re.I,
)

# Area units for doses. A model can copy a number correctly but swap the unit
# ("5 ml per hectare" -> "per acre"), which the quantity check cannot see.
# Tamil entries are stems so that case endings (ஏக்கருக்கு, எக்டரில்) match.
_AREA_UNITS = {
    "acre": re.compile(r"\b(?:acres?|ac)\b|ஏக்கர", re.I),
    "hectare": re.compile(r"\b(?:hectares?|ha)\b|எக்டர|ஹெக்டேர|ஹெக்டர", re.I),
}


@dataclass
class SourceRef:
    """A source shown under an answer; built only from retrieved metadata."""

    label: str
    title: str
    url: str | None
    page_number: int | None
    language: str | None
    publisher: str | None
    score: float
    excerpt: str
    cited: bool


@dataclass
class ChatResponse:
    status: str
    answer: str
    language: str
    sources: list[SourceRef] = field(default_factory=list)
    notices: list[str] = field(default_factory=list)  # informational
    warnings: list[str] = field(default_factory=list)  # reliability caveats
    safety_note: bool = False
    mode: str = "offline"
    latency_ms: int = 0


class ResponseService:
    """Coordinates retrieval, generation and response validation."""

    def __init__(
        self,
        retriever: Retriever,
        provider: LLMProvider | None,
        max_question_chars: int = 1000,
        min_question_chars: int = 3,
    ) -> None:
        self.retriever = retriever
        self.provider = provider
        self.max_question_chars = max_question_chars
        self.min_question_chars = min_question_chars

    @property
    def mode(self) -> str:
        return "offline" if self.provider is None else self.provider.name

    def answer(self, question: str, language: str = "en") -> ChatResponse:
        """Answer ``question`` in ``language`` ("en" or "ta")."""
        start = time.perf_counter()
        language = language if language in ("en", "ta") else "en"
        response = self._answer(question, language)
        response.latency_ms = int((time.perf_counter() - start) * 1000)
        logger.info("status=%s mode=%s sources=%d latency=%dms", response.status, response.mode, len(response.sources), response.latency_ms)
        return response

    # ------------------------------------------------------------------ #
    def _answer(self, question: str, language: str) -> ChatResponse:
        check = validate_question(question, self.max_question_chars, self.min_question_chars)
        if not check.ok:
            msg = t(language, check.error_key or "empty", length=len(check.text), limit=self.max_question_chars)
            return ChatResponse(INVALID_INPUT, msg, language, mode=self.mode)

        notices = []
        if check.detected_language in ("en", "ta") and check.detected_language != language:
            notices.append(
                t(language, "mismatch", detected=language_name(check.detected_language), selected=language_name(language))
            )

        passages = self.retriever.retrieve(check.text, language)
        if not passages:
            return ChatResponse(INSUFFICIENT, t(language, "insufficient"), language, notices=notices, mode=self.mode)

        if self.provider is None:
            return self._offline_answer(check.text, passages, language, notices)

        messages = build_messages(check.text, passages, language)
        try:
            raw = self.provider.generate(messages)
        except LLMError as exc:
            logger.warning("LLM failure: %s", type(exc).__name__)
            return ChatResponse(
                ERROR,
                f"{t(language, 'error_prefix')} {exc.user_message}",
                language,
                sources=self._sources(passages, cited=set()),
                notices=notices,
                mode=self.mode,
            )
        return self._verify(check.text, raw, passages, language, notices)

    def _verify(
        self, question: str, raw: str, passages: list[RetrievedPassage], language: str, notices: list[str]
    ) -> ChatResponse:
        labels = {p.label for p in passages}
        urls = {p.chunk.source_url for p in passages if p.chunk.source_url}
        cit = validate_citations(raw, labels, urls)
        warnings: list[str] = []
        if cit.invalid_labels:
            warnings.append(f"Removed citations to non-existent sources: {', '.join(sorted(set(cit.invalid_labels)))}.")
        if cit.removed_urls:
            warnings.append("Removed links that were not part of the verified sources.")

        evidence = " ".join(p.chunk.text for p in passages)
        unsupported = [q for q in _QUANTITY_RE.findall(cit.text) if _norm(q) not in _norm(evidence)]
        if unsupported:
            warnings.append(
                "The answer contains quantities not found in the retrieved sources "
                f"({', '.join(sorted(set(u.strip() for u in unsupported))[:5])}). Do not rely on them; "
                "check the product label or your local agricultural officer."
            )

        swapped = [
            unit
            for unit, pattern in _AREA_UNITS.items()
            if pattern.search(cit.text) and not pattern.search(evidence)
        ]
        if swapped:
            warnings.append(
                f"The answer gives a dose per {' / '.join(swapped)}, but the sources do not use that unit. "
                "Check the unit in the source passages below."
            )

        status = ANSWERED if cit.cited_labels else UNVERIFIED
        if status == UNVERIFIED:
            warnings.append(
                "The AI answer did not cite any verified source, so it may not be supported by the "
                "knowledge base. Please read the source passages below."
            )
        if language == "ta" and detect_language(cit.text) != "ta":
            warnings.append("The AI model did not answer in Tamil reliably.")

        return ChatResponse(
            status,
            cit.text,
            language,
            sources=self._sources(passages, set(cit.cited_labels)),
            notices=notices,
            warnings=warnings,
            safety_note=is_high_risk(question, cit.text),
            mode=self.mode,
        )

    def _offline_answer(
        self, question: str, passages: list[RetrievedPassage], language: str, notices: list[str]
    ) -> ChatResponse:
        lines = [f"**{t(language, 'offline_label')}**", ""]
        for p in passages[:3]:
            text, _ = neutralize_passage(p.chunk.text)
            lines.append(f"- {text} [{p.label}]")
        if any(p.chunk.language != language for p in passages[:3]):
            notices.append(t(language, "offline_lang_note"))
        return ChatResponse(
            OFFLINE,
            "\n".join(lines),
            language,
            sources=self._sources(passages, {p.label for p in passages[:3]}),
            notices=notices,
            safety_note=is_high_risk(question, " ".join(p.chunk.text for p in passages[:3])),
            mode="offline",
        )

    @staticmethod
    def _sources(passages: list[RetrievedPassage], cited: set[str]) -> list[SourceRef]:
        refs = [
            SourceRef(
                label=p.label,
                title=p.chunk.source_title,
                url=p.chunk.source_url,
                page_number=p.chunk.page_number,
                language=p.chunk.language,
                publisher=p.chunk.publisher,
                score=round(p.score, 3),
                excerpt=p.chunk.text,
                cited=p.label in cited,
            )
            for p in passages
        ]
        return sorted(refs, key=lambda r: (not r.cited, -r.score))


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", text.lower()).replace("–", "-")
