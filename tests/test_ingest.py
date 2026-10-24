import json

import pymupdf
import pytest

from knowledge_base.ingest import (
    DocumentLoadError,
    IngestReport,
    chunk_documents,
    extract_pdf_pages,
    load_document,
    load_documents,
)


def _make_pdf(path, pages):
    doc = pymupdf.open()
    for body in pages:
        page = doc.new_page()
        page.insert_text((72, 60), "Natural Pest Control Guide")  # running header
        page.insert_text((72, 100), body)
        page.insert_text((72, 800), f"Page {doc.page_count}")
    doc.save(path)
    doc.close()


def test_pdf_extraction_keeps_page_numbers_and_strips_running_headers(tmp_path):
    pdf = tmp_path / "guide.pdf"
    _make_pdf(pdf, ["Neem oil is used in pest control.", "Crop rotation breaks pest cycles.", "Use healthy seed."])
    pages = extract_pdf_pages(pdf)
    assert [p.page_number for p in pages] == [1, 2, 3]
    assert "Crop rotation" in pages[1].text
    assert all("Natural Pest Control Guide" not in p.text for p in pages)
    assert all("Page " not in p.text for p in pages)


def test_corrupt_pdf_raises_load_error(tmp_path):
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"%PDF-1.4 this is not really a pdf")
    with pytest.raises(DocumentLoadError):
        load_document(bad)


def test_pdf_without_text_raises_load_error(tmp_path):
    pdf = tmp_path / "scanned.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.save(pdf)
    doc.close()
    with pytest.raises(DocumentLoadError, match="no extractable text"):
        load_document(pdf)


def test_empty_and_unsupported_files(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("")
    with pytest.raises(DocumentLoadError, match="empty"):
        load_document(empty)
    other = tmp_path / "image.png"
    other.write_bytes(b"\x89PNG")
    with pytest.raises(DocumentLoadError, match="unsupported"):
        load_document(other)


def test_sidecar_metadata_is_preserved_into_chunks(tmp_path):
    src = tmp_path / "drip.txt"
    src.write_text("Drip irrigation saves water. " * 30, encoding="utf-8")
    (tmp_path / "drip.txt.meta.json").write_text(
        json.dumps(
            {
                "document_id": "tnau-drip",
                "source_title": "Irrigation Management: Drip Irrigation",
                "source_url": "https://agritech.tnau.ac.in/x.html",
                "language": "en",
                "published_date": None,
                "publisher": "TNAU",
            }
        ),
        encoding="utf-8",
    )
    doc = load_document(src)
    chunks = chunk_documents([doc], chunk_size=200, overlap=40)
    assert len(chunks) > 1
    for c in chunks:
        assert c.document_id == "tnau-drip"
        assert c.source_title == "Irrigation Management: Drip Irrigation"
        assert c.source_url == "https://agritech.tnau.ac.in/x.html"
        assert c.publisher == "TNAU"
        assert c.published_date is None
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_missing_metadata_is_not_invented(tmp_path):
    src = tmp_path / "my_notes.md"
    src.write_text("# Notes\nGreen manure improves soil.", encoding="utf-8")
    doc = load_document(src)
    assert doc.source_url is None
    assert doc.published_date is None
    assert doc.source_title == "my notes"
    assert doc.language == "en"


def test_tamil_language_detected_without_sidecar(tmp_path):
    src = tmp_path / "ta.txt"
    src.write_text("நெல் பயிருக்கு நீர் மேலாண்மை மிகவும் முக்கியம்.", encoding="utf-8")
    assert load_document(src).language == "ta"


def test_json_passages_with_page_numbers(tmp_path):
    src = tmp_path / "faq.json"
    src.write_text(
        json.dumps(
            {
                "source_title": "FAQ",
                "language": "en",
                "passages": [{"text": "Answer one.", "page_number": 2}, {"text": "Answer two."}, {"bad": 1}],
            }
        ),
        encoding="utf-8",
    )
    doc = load_document(src)
    assert doc.source_title == "FAQ"
    assert [p.page_number for p in doc.pages] == [2, None]


def test_invalid_json_raises(tmp_path):
    src = tmp_path / "bad.json"
    src.write_text("{not json", encoding="utf-8")
    with pytest.raises(DocumentLoadError, match="invalid JSON"):
        load_document(src)


def test_load_documents_skips_bad_files_and_reports(tmp_path):
    (tmp_path / "good.txt").write_text("Mulching conserves soil moisture.", encoding="utf-8")
    (tmp_path / "bad.pdf").write_bytes(b"garbage")
    (tmp_path / "empty.md").write_text("")
    (tmp_path / "good.txt.meta.json").write_text("{broken", encoding="utf-8")  # unreadable sidecar ignored
    report = IngestReport()
    docs = load_documents(tmp_path, report)
    assert [d.document_id for d in docs] == ["good"]
    assert set(report.skipped) == {"bad.pdf", "empty.md"}


def test_load_documents_missing_directory_returns_empty(tmp_path):
    assert load_documents(tmp_path / "does-not-exist") == []


def test_tamil_documents_use_smaller_chunks(tmp_path):
    src = tmp_path / "ta.txt"
    src.write_text("நெல் பயிருக்கு நீர் மேலாண்மை மிகவும் முக்கியம். " * 30, encoding="utf-8")
    doc = load_document(src)
    chunks = chunk_documents([doc], chunk_size=400, overlap=70, chunk_size_ta=200)
    assert chunks and all(len(c.text) <= 200 for c in chunks)


def test_table_debris_chunks_are_skipped(tmp_path):
    src = tmp_path / "drip.txt"
    src.write_text("Drip irrigation saves water.\n3-6 44 62 1.20 75,000 Pumpkin 1-2 3-6 15 54 1.20 75,000\n", encoding="utf-8")
    chunks = chunk_documents([load_document(src)], chunk_size=40, overlap=0)
    assert [c.text for c in chunks] == ["Drip irrigation saves water."]
