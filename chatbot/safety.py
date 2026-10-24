"""Input validation, prompt-injection defence and citation validation.

These checks run in application code, independently of the system prompt, so
that safety does not rely on the model following instructions.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from chatbot.language import detect_language


# --------------------------------------------------------------------------- #
# User input
# --------------------------------------------------------------------------- #
@dataclass
class InputCheck:
    ok: bool
    text: str
    error_key: str | None = None  # key into chatbot.language.UI_TEXT
    detected_language: str = "unknown"


_CONTROL = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]")


def validate_question(raw: str | None, max_chars: int = 1000, min_chars: int = 3) -> InputCheck:
    """Normalise and validate a user question."""
    text = unicodedata.normalize("NFC", raw or "")
    text = _CONTROL.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return InputCheck(False, text, "empty")
    if len(text) > max_chars:
        return InputCheck(False, text, "too_long")
    if len(text) < min_chars:
        return InputCheck(False, text, "too_short")
    language = detect_language(text)
    if language == "other":
        return InputCheck(False, text, "unsupported_language", language)
    return InputCheck(True, text, None, language)


# --------------------------------------------------------------------------- #
# Retrieved content (untrusted evidence)
# --------------------------------------------------------------------------- #
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+|any\s+)?(the\s+)?(previous|prior|above|earlier)\s+(instructions|prompts?|rules)",
    r"disregard\s+(all\s+|the\s+)?(previous|prior|above|system)",
    r"forget\s+(all\s+|your\s+)?(previous\s+)?(instructions|rules)",
    r"you\s+are\s+now\s+",
    r"new\s+instructions?\s*:",
    r"system\s*prompt",
    r"^\s*(system|assistant|developer)\s*:",
    r"reveal\s+(your|the)\s+(prompt|instructions|api\s*key|secret)",
    r"</?\s*(passage|system|instructions?)\s*>",
    r"(api[_\s-]?key|password|secret\s+token)",
]
_INJECTION_RE = re.compile("|".join(f"(?:{p})" for p in _INJECTION_PATTERNS), re.I | re.M)


def neutralize_passage(text: str) -> tuple[str, bool]:
    """Remove instruction-like content from a retrieved passage.

    Returns (sanitised_text, was_modified). Sentences matching known injection
    patterns are replaced with a marker, and characters that could close the
    evidence delimiters are escaped. The passage remains evidence only.
    """
    modified = False
    parts = re.split(r"(?<=[.!?\n])\s+", text)
    kept = []
    for part in parts:
        if _INJECTION_RE.search(part):
            kept.append("[removed: instruction-like text in source]")
            modified = True
        else:
            kept.append(part)
    # Escape angle brackets so a passage cannot close its <passage> delimiter.
    cleaned = " ".join(kept).replace("<", "‹").replace(">", "›")
    return cleaned, modified


# --------------------------------------------------------------------------- #
# Model output
# --------------------------------------------------------------------------- #
_CITATION_RE = re.compile(r"\[\s*(S\d+)(?:\s*[,;]\s*S\d+)*\s*\]")
_LABEL_RE = re.compile(r"S\d+")
_URL_RE = re.compile(r"(?:https?://|www\.)[^\s)\]>\"']+", re.I)


@dataclass
class CitationCheck:
    text: str
    cited_labels: list[str]
    invalid_labels: list[str]
    removed_urls: list[str]


def validate_citations(answer: str, valid_labels: set[str], allowed_urls: set[str]) -> CitationCheck:
    """Keep only citations that refer to retrieved passages and strip any URL
    the model wrote that is not an actual retrieved source URL."""
    cited: list[str] = []
    invalid: list[str] = []

    def _fix(match: re.Match[str]) -> str:
        labels = _LABEL_RE.findall(match.group(0))
        good = [label for label in labels if label in valid_labels]
        invalid.extend(label for label in labels if label not in valid_labels)
        for label in good:
            if label not in cited:
                cited.append(label)
        return f"[{', '.join(good)}]" if good else ""

    text = _CITATION_RE.sub(_fix, answer)

    removed: list[str] = []

    def _url(match: re.Match[str]) -> str:
        url = match.group(0).rstrip(".,;")
        if url in allowed_urls:
            return match.group(0)
        removed.append(url)
        return "[link removed]"

    text = _URL_RE.sub(_url, text)
    text = re.sub(r"[ \t]+([.,;:])", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text).strip()
    return CitationCheck(text=text, cited_labels=cited, invalid_labels=invalid, removed_urls=removed)


# --------------------------------------------------------------------------- #
# High-risk topics
# --------------------------------------------------------------------------- #
_HIGH_RISK = re.compile(
    r"pesticide|insecticide|fungicide|herbicide|weedicide|chemical|spray|dose|dosage|"
    r"ml\s*/|g\s*/|kg\s*/|per\s+(acre|ha|hectare|litre|liter)|fertili[sz]er|urea|dap\b|potash|"
    r"poison|toxic|"
    r"பூச்சிக்கொல்லி|பூஞ்சாணக்கொல்லி|களைக்கொல்லி|மருந்து|தெளி|உரம்|யூரியா|அளவு|விஷம்|நச்சு",
    re.I,
)


def is_high_risk(*texts: str) -> bool:
    """True if the question or answer concerns agro-chemicals or quantities,
    in which case a label/expert-guidance note is attached."""
    return any(_HIGH_RISK.search(t or "") for t in texts)
