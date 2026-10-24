"""Text extraction helpers, cleaning and chunking."""

from __future__ import annotations

import html
import re
import unicodedata

_BLOCK_TAGS = re.compile(r"(?i)<\s*(br|p|tr|li|h[1-6]|div|table|ul|ol|dt|dd)\b[^>]*>")
_CELL_TAGS = re.compile(r"(?i)<\s*/?\s*(td|th)\b[^>]*>")
_SCRIPT_STYLE = re.compile(r"(?is)<(script|style|noscript)\b.*?</\1\s*>")
_COMMENTS = re.compile(r"(?s)<!--.*?-->")
_TAGS = re.compile(r"<[^>]+>")
_TITLE = re.compile(r"(?is)<title[^>]*>(.*?)</title>")

# Lines that are site navigation/footer rather than agricultural content.
_BOILERPLATE = [
    re.compile(r"©|all rights reserved", re.I),
    re.compile(r"^\s*(home|top|back|next|previous|disclaimer)\s*$", re.I),
    re.compile(r"^\s*TNAU Agritech Portal\s*::", re.I),
    re.compile(r"^\s*தவேப வேளாண் இணைய தளம்\s*::"),
]

# Zero-width and control characters except Tamil-relevant joiners (ZWJ/ZWNJ).
_INVISIBLE = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f­​⁠﻿]")
_SENTENCE_END = re.compile(r"(?<=[.!?।])\s+|\n+")


def html_title(raw_html: str) -> str | None:
    """Return the text of an HTML document's ``<title>`` element, if any."""
    match = _TITLE.search(raw_html)
    if not match:
        return None
    return re.sub(r"\s+", " ", html.unescape(match.group(1))).strip() or None


def _is_nav_line(line: str) -> bool:
    """Navigation rows such as "Home | About Us | Contact". Menus are often
    split over several lines, so any line made only of short "|"-separated
    items (or starting/ending with "|") is treated as navigation."""
    if "|" not in line:
        return False
    if line.count("|") >= 2 or line.startswith("|") or line.endswith("|"):
        return True
    return all(len(part.split()) <= 4 for part in line.split("|"))


def html_to_text(raw_html: str) -> str:
    """Convert an HTML page to plain text, keeping line structure and dropping
    scripts, styles, navigation bars and footers."""
    text = _COMMENTS.sub(" ", raw_html)
    text = _SCRIPT_STYLE.sub(" ", text)
    text = _TITLE.sub(" ", text)
    text = _BLOCK_TAGS.sub("\n", text)
    text = _CELL_TAGS.sub(" ", text)
    text = html.unescape(_TAGS.sub(" ", text))
    kept: list[str] = []
    for line in text.splitlines():
        line = re.sub(r"[ \t\xa0]+", " ", line).strip()
        if not line:
            continue
        if _is_nav_line(line):
            continue
        if any(p.search(line) for p in _BOILERPLATE):
            continue
        kept.append(line)
    return "\n".join(kept)


def clean_text(text: str) -> str:
    """Normalise Unicode (NFC, important for Tamil), remove invisible
    characters, repair hyphenated line breaks and collapse whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = _INVISIBLE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # de-hyphenate PDF line breaks
    text = re.sub(r"[ \t\xa0]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_units(text: str) -> list[str]:
    """Split text into sentence/line units for chunk packing."""
    return [u.strip() for u in _SENTENCE_END.split(text) if u and u.strip()]


def chunk_text(text: str, chunk_size: int = 450, overlap: int = 80) -> list[str]:
    """Split ``text`` into chunks of at most ``chunk_size`` characters.

    Chunks are packed from whole sentences/lines where possible; a sentence
    longer than ``chunk_size`` is hard-split. Consecutive chunks share roughly
    ``overlap`` characters of trailing context so that facts spanning a
    boundary remain retrievable.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")
    text = text.strip()
    if not text:
        return []

    units: list[str] = []
    for unit in _split_units(text):
        while len(unit) > chunk_size:
            cut = unit.rfind(" ", 0, chunk_size)
            cut = cut if cut > chunk_size // 2 else chunk_size
            units.append(unit[:cut].strip())
            unit = unit[cut:].strip()
        if unit:
            units.append(unit)

    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current} {unit}" if current else unit
        if current and len(candidate) > chunk_size:
            chunks.append(current)
            tail = _overlap_tail(current, overlap)
            current = f"{tail} {unit}" if tail and len(tail) + len(unit) + 1 <= chunk_size else unit
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def letter_ratio(text: str) -> float:
    """Share of non-space characters that are letters or combining marks
    (Tamil vowel signs are marks, not letters)."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(unicodedata.category(c)[0] in "LM" for c in chars) / len(chars)


def is_table_debris(text: str, min_letter_ratio: float = 0.5) -> bool:
    """True for chunks that are mostly numbers, e.g. a flattened HTML table
    whose rows and columns can no longer be read. Such chunks mislead the
    LLM, so they are left out of the index. Dosage tables (product names
    with quantities) stay well above the default ratio."""
    return letter_ratio(text) < min_letter_ratio


def _overlap_tail(chunk: str, overlap: int) -> str:
    """Return roughly the last ``overlap`` characters of ``chunk``, starting at
    a word boundary."""
    if overlap <= 0 or len(chunk) <= overlap:
        return "" if overlap <= 0 else chunk
    tail = chunk[-overlap:]
    space = tail.find(" ")
    return tail[space + 1 :] if 0 <= space < len(tail) - 1 else tail
