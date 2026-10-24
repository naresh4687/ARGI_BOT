"""Document loading and knowledge-base build pipeline.

Supported inputs inside ``data/raw`` (searched recursively):

* ``.pdf``  – text extracted per page with PyMuPDF (page numbers preserved)
* ``.txt`` / ``.md`` – plain text
* ``.json`` – either ``{"text": ...}``, ``{"passages": [{"text": ...}, ...]}``
  or a list of such passage objects

Each file may have a sidecar ``<filename>.meta.json`` holding verified source
metadata (``source_title``, ``source_url``, ``language``, ``published_date``,
...). Without a sidecar, the file name is used as the title and the URL is left
``None`` - metadata is never invented.

Run ``python -m knowledge_base.ingest`` to (re)build the FAISS index.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from knowledge_base.preprocess import chunk_text, clean_text, is_table_debris

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".json"}
META_SUFFIX = ".meta.json"
MAX_FILE_BYTES = 25 * 1024 * 1024  # refuse unexpectedly large files


@dataclass
class PageText:
    """Text from one logical page of a source document."""

    text: str
    page_number: int | None = None


@dataclass
class SourceDocument:
    """A loaded source document with verified metadata."""

    document_id: str
    source_title: str
    pages: list[PageText]
    source_url: str | None = None
    language: str | None = None
    published_date: str | None = None
    publisher: str | None = None
    topic: str | None = None
    file_path: str | None = None


@dataclass
class Chunk:
    """A retrievable passage with the metadata needed for citation."""

    chunk_id: str
    document_id: str
    source_title: str
    text: str
    source_url: str | None = None
    page_number: int | None = None
    language: str | None = None
    published_date: str | None = None
    publisher: str | None = None
    topic: str | None = None

    @property
    def search_text(self) -> str:
        """Text used for embedding and lexical matching: the curated document
        title is prepended so that every chunk carries its topic (e.g. "Drip
        Irrigation") even when the passage itself does not repeat it. Only
        ``text`` is ever shown to users or the LLM as evidence."""
        return f"{self.source_title}\n{self.text}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class IngestReport:
    """Summary of an ingestion run."""

    loaded: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    chunk_count: int = 0


class DocumentLoadError(Exception):
    """Raised when a document cannot be read or contains no usable text."""


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def _read_sidecar(path: Path) -> dict[str, Any]:
    meta_path = path.with_name(path.name + META_SUFFIX)
    if not meta_path.exists():
        return {}
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring unreadable metadata file %s: %s", meta_path.name, exc)
        return {}


def _detect_language(text: str) -> str:
    from chatbot.language import detect_language

    return detect_language(text)


def extract_pdf_pages(path: Path) -> list[PageText]:
    """Extract text from each page of a PDF. Raises DocumentLoadError if the
    file is corrupt, encrypted or has no extractable text (e.g. scanned)."""
    import pymupdf

    try:
        doc = pymupdf.open(path)
    except Exception as exc:  # PyMuPDF raises several types for bad files
        raise DocumentLoadError(f"cannot open PDF: {exc}") from exc
    try:
        if doc.needs_pass:
            raise DocumentLoadError("PDF is password protected")
        raw_pages = [(index + 1, page.get_text("text")) for index, page in enumerate(doc)]
    except DocumentLoadError:
        raise
    except Exception as exc:
        raise DocumentLoadError(f"cannot read PDF pages: {exc}") from exc
    finally:
        doc.close()
    repeated = _running_lines([text for _, text in raw_pages])
    pages = []
    for number, raw in raw_pages:
        lines = [
            line
            for line in raw.splitlines()
            if line.strip() not in repeated and not _PAGE_LABEL.match(line)
        ]
        text = clean_text("\n".join(lines))
        if text:
            pages.append(PageText(text=text, page_number=number))
    if not pages:
        raise DocumentLoadError("PDF has no extractable text (it may be a scanned image)")
    return pages


_PAGE_LABEL = re.compile(r"^\s*(page\s*)?\d{1,4}\s*(of\s*\d+)?\s*$", re.I)


def _running_lines(page_texts: list[str]) -> set[str]:
    """Lines that occur on more than half of the pages (running headers and
    footers). Only applied to documents with at least 3 pages."""
    if len(page_texts) < 3:
        return set()
    counts: dict[str, int] = {}
    for text in page_texts:
        for line in {l.strip() for l in text.splitlines() if l.strip()}:
            counts[line] = counts.get(line, 0) + 1
    return {line for line, n in counts.items() if n > len(page_texts) / 2}


def _json_passages(data: Any) -> list[PageText]:
    items = data.get("passages", [data]) if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise DocumentLoadError("unsupported JSON structure")
    pages = []
    for item in items:
        if isinstance(item, dict) and isinstance(item.get("text"), str):
            text = clean_text(item["text"])
            if text:
                page = item.get("page_number")
                pages.append(PageText(text=text, page_number=page if isinstance(page, int) else None))
    return pages


def load_document(path: Path) -> SourceDocument:
    """Load one supported file into a :class:`SourceDocument`."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES or path.name.endswith(META_SUFFIX):
        raise DocumentLoadError(f"unsupported file type: {path.suffix}")
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise DocumentLoadError(f"cannot read file: {exc}") from exc
    if size == 0:
        raise DocumentLoadError("file is empty")
    if size > MAX_FILE_BYTES:
        raise DocumentLoadError("file exceeds the 25 MB size limit")

    meta = _read_sidecar(path)
    if suffix == ".pdf":
        pages = extract_pdf_pages(path)
    else:
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise DocumentLoadError(f"cannot decode file as UTF-8: {exc}") from exc
        if suffix == ".json":
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise DocumentLoadError(f"invalid JSON: {exc}") from exc
            if isinstance(data, dict):
                meta = {**{k: v for k, v in data.items() if k not in {"text", "passages"}}, **meta}
            pages = _json_passages(data)
        else:
            text = clean_text(raw)
            pages = [PageText(text=text)] if text else []
    if not pages:
        raise DocumentLoadError("no usable text found")

    document_id = str(meta.get("document_id") or _slug(path.stem))
    language = meta.get("language") or _detect_language(" ".join(p.text for p in pages[:3]))
    return SourceDocument(
        document_id=document_id,
        source_title=str(meta.get("source_title") or path.stem.replace("_", " ")),
        source_url=meta.get("source_url") or None,
        language=language,
        published_date=meta.get("published_date") or None,
        publisher=meta.get("publisher") or None,
        topic=meta.get("topic") or None,
        pages=pages,
        file_path=str(path),
    )


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "document"


def discover_files(raw_dir: Path) -> list[Path]:
    """Return supported source files under ``raw_dir`` in a stable order."""
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        return []
    return sorted(
        p
        for p in raw_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES and not p.name.endswith(META_SUFFIX)
    )


def load_documents(raw_dir: Path, report: IngestReport | None = None) -> list[SourceDocument]:
    """Load every supported document, skipping (and reporting) bad files."""
    report = report if report is not None else IngestReport()
    docs: list[SourceDocument] = []
    seen_ids: set[str] = set()
    for path in discover_files(raw_dir):
        try:
            doc = load_document(path)
        except DocumentLoadError as exc:
            logger.warning("Skipping %s: %s", path.name, exc)
            report.skipped[path.name] = str(exc)
            continue
        if doc.document_id in seen_ids:
            report.skipped[path.name] = f"duplicate document_id {doc.document_id!r}"
            continue
        seen_ids.add(doc.document_id)
        docs.append(doc)
        report.loaded.append(path.name)
    return docs


def chunk_documents(
    docs: Iterable[SourceDocument],
    chunk_size: int = 400,
    overlap: int = 70,
    chunk_size_ta: int | None = None,
) -> list[Chunk]:
    """Split documents into chunks that each carry full source metadata.

    Tamil documents use ``chunk_size_ta`` (overlap scaled proportionally) so
    that chunks fit the embedding model's 128-token window.
    """
    chunks: list[Chunk] = []
    for doc in docs:
        size, ovl = chunk_size, overlap
        if doc.language == "ta" and chunk_size_ta:
            size, ovl = chunk_size_ta, max(0, round(overlap * chunk_size_ta / chunk_size))
        for page in doc.pages:
            for i, text in enumerate(chunk_text(page.text, size, ovl)):
                if is_table_debris(text):
                    logger.info("Skipping table debris in %s: %.60s", doc.document_id, text)
                    continue
                digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
                page_part = f"p{page.page_number}" if page.page_number else "p0"
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.document_id}:{page_part}:{i}:{digest}",
                        document_id=doc.document_id,
                        source_title=doc.source_title,
                        source_url=doc.source_url,
                        page_number=page.page_number,
                        language=doc.language,
                        published_date=doc.published_date,
                        publisher=doc.publisher,
                        topic=doc.topic,
                        text=text,
                    )
                )
    return chunks


def build_knowledge_base(settings=None, embedder=None) -> IngestReport:
    """Load, chunk, embed and persist the knowledge base. Returns a report."""
    from config.settings import Settings
    from knowledge_base.embeddings import SentenceTransformerEmbedder
    from knowledge_base.vector_store import FaissVectorStore

    settings = settings or Settings.from_env()
    report = IngestReport()
    docs = load_documents(settings.raw_data_dir, report)
    if not docs:
        raise DocumentLoadError(
            f"No usable documents in {settings.raw_data_dir}. "
            "Run `python -m knowledge_base.fetch_sources` or add files manually."
        )
    chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap, settings.chunk_size_ta)
    embedder = embedder or SentenceTransformerEmbedder(settings.embedding_model)
    vectors = embedder.embed([c.search_text for c in chunks])
    store = FaissVectorStore.build(
        vectors,
        chunks,
        info={
            "embedding_model": embedder.model_name,
            "chunk_size": settings.chunk_size,
            "chunk_size_ta": settings.chunk_size_ta,
            "chunk_overlap": settings.chunk_overlap,
            "documents": len(docs),
        },
    )
    store.save(settings.index_dir)
    report.chunk_count = len(chunks)
    return report


def main(argv: list[str] | None = None) -> int:
    from config.settings import configure_logging

    parser = argparse.ArgumentParser(description="Build the AgriBot AI FAISS index.")
    parser.parse_args(argv)
    configure_logging()
    try:
        report = build_knowledge_base()
    except DocumentLoadError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Loaded {len(report.loaded)} documents -> {report.chunk_count} chunks.")
    for name, reason in report.skipped.items():
        print(f"  skipped {name}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
