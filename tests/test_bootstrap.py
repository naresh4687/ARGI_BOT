from dataclasses import replace

import pytest

import chatbot.bootstrap as bootstrap
from config.settings import Settings
from knowledge_base.ingest import DocumentLoadError
from tests.conftest import HashingEmbedder


def _settings(tmp_path, **kw):
    return replace(
        Settings(),
        raw_data_dir=tmp_path / "raw",
        index_dir=tmp_path / "processed",
        embedding_model=HashingEmbedder.model_name,
        **kw,
    )


def test_empty_corpus_raises_without_auto_fetch(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap, "fetch_all", lambda *a, **k: pytest.fail("must not download"))
    with pytest.raises(DocumentLoadError):
        bootstrap.load_or_build_store(_settings(tmp_path), HashingEmbedder())


def test_auto_fetch_downloads_then_builds_index(tmp_path, monkeypatch):
    calls = []

    def fake_fetch(raw_dir, allow_insecure_tls=False):
        calls.append(allow_insecure_tls)
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / "drip.txt").write_text("Drip irrigation saves water and reduces weeds.", encoding="utf-8")
        return {"drip": "fetched"}

    monkeypatch.setattr(bootstrap, "fetch_all", fake_fetch)
    store = bootstrap.load_or_build_store(_settings(tmp_path, auto_fetch_sources=True), HashingEmbedder())
    assert calls == [True]
    assert store.size >= 1 and store.chunks[0].document_id == "drip"
    assert store.info["tls_unverified_documents"] == 0


def test_tls_unverified_downloads_are_counted_in_index_info(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for name, verified in [("drip", True), ("blast", False), ("ipm", False)]:
        (raw / f"{name}.txt").write_text(f"{name} passage about crops and water.", encoding="utf-8")
        (raw / f"{name}.txt.meta.json").write_text(f'{{"tls_verified": {str(verified).lower()}}}', encoding="utf-8")
    (raw / "local.txt").write_text("A manually added document without a sidecar.", encoding="utf-8")
    store = bootstrap.load_or_build_store(_settings(tmp_path), HashingEmbedder())
    assert store.info["documents"] == 4
    assert store.info["tls_unverified_documents"] == 2
