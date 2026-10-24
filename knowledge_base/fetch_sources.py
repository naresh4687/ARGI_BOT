"""Download the curated starter corpus listed in ``data/sources/manifest.json``.

Each HTML page is converted to cleaned text and saved as
``data/raw/<document_id>.txt``; PDFs are saved as ``<document_id>.pdf``. A
``.meta.json`` sidecar records the verified URL, title, publisher, language,
retrieval time and the TLS status of the download.

Usage::

    python -m knowledge_base.fetch_sources            # skip files already present
    python -m knowledge_base.fetch_sources --force    # re-download everything

Only the URLs in the manifest are fetched (no crawling). Requests are spaced
out politely and identify the client with a descriptive User-Agent.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from config.settings import PROJECT_ROOT, configure_logging
from knowledge_base.preprocess import clean_text, html_title, html_to_text

logger = logging.getLogger(__name__)

MANIFEST = PROJECT_ROOT / "data" / "sources" / "manifest.json"
USER_AGENT = "AgriBotAI-student-project/1.0 (educational; manifest-based fetch, no crawling)"
DELAY_SECONDS = 1.5
MIN_TEXT_CHARS = 200


def _decode(content: bytes) -> str:
    for encoding in ("utf-8", "cp1252"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="replace")


def _download(url: str, allow_insecure_tls: bool) -> tuple[bytes, bool]:
    """Fetch ``url``; returns (content, tls_verified)."""
    headers = {"User-Agent": USER_AGENT}
    try:
        resp = requests.get(url, headers=headers, timeout=40)
        verified = True
    except requests.exceptions.SSLError:
        if not allow_insecure_tls:
            raise
        # Public, read-only content; the TNAU server intermittently presents an
        # expired certificate. The fallback is recorded in the sidecar metadata.
        logger.warning("TLS verification failed for %s; retrying without verification", url)
        import urllib3

        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        resp = requests.get(url, headers=headers, timeout=40, verify=False)
        verified = False
    resp.raise_for_status()
    return resp.content, verified


def fetch_all(raw_dir: Path, force: bool = False, allow_insecure_tls: bool = False) -> dict[str, str]:
    """Fetch every manifest entry. Returns {document_id: status}."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    raw_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, str] = {}
    for entry in manifest["documents"]:
        doc_id = entry["document_id"]
        suffix = ".pdf" if entry.get("format") == "pdf" else ".txt"
        out_path = raw_dir / f"{doc_id}{suffix}"
        meta_path = out_path.with_name(out_path.name + ".meta.json")
        if out_path.exists() and meta_path.exists() and not force:
            results[doc_id] = "exists"
            continue
        try:
            content, verified = _download(entry["source_url"], allow_insecure_tls)
        except requests.RequestException as exc:
            logger.error("Failed to fetch %s: %s", doc_id, exc)
            results[doc_id] = f"error: {exc.__class__.__name__}"
            continue

        meta = {k: entry.get(k) for k in ("document_id", "source_title", "source_url", "publisher", "language", "topic", "published_date")}
        meta["retrieved_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        meta["tls_verified"] = verified

        if suffix == ".pdf":
            if not content.startswith(b"%PDF"):
                results[doc_id] = "error: response is not a PDF"
                continue
            out_path.write_bytes(content)
        else:
            page = _decode(content)
            meta["html_title"] = html_title(page)
            text = clean_text(html_to_text(page))
            if len(text) < MIN_TEXT_CHARS:
                results[doc_id] = f"error: only {len(text)} characters of text"
                continue
            out_path.write_text(text + "\n", encoding="utf-8")
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        results[doc_id] = "fetched" if verified else "fetched (TLS unverified)"
        time.sleep(DELAY_SECONDS)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    parser.add_argument(
        "--allow-insecure-tls",
        action="store_true",
        help="retry without certificate verification if the server's TLS certificate is invalid",
    )
    args = parser.parse_args(argv)
    configure_logging()
    from config.settings import Settings

    results = fetch_all(Settings.from_env().raw_data_dir, args.force, args.allow_insecure_tls)
    for doc_id, status in results.items():
        print(f"{status:28} {doc_id}")
    failed = [d for d, s in results.items() if s.startswith("error")]
    print(f"\n{len(results) - len(failed)} ok, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
