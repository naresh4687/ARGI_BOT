import numpy as np
import pytest

from knowledge_base.vector_store import (
    CHUNKS_FILE,
    FaissVectorStore,
    IndexCorruptError,
    IndexNotFoundError,
)
from tests.conftest import SAMPLE_CHUNKS


def test_search_returns_metadata_sorted_by_score(store, embedder):
    q = store.transform_query(embedder.embed(["drip irrigation water saving"])[0], "en")
    hits = store.search(q, k=3)
    assert len(hits) == 3
    assert hits[0].chunk.document_id == "drip"
    assert hits[0].chunk.source_url == "https://example.org/drip"
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_all_scores_matches_search(store, embedder):
    q = store.transform_query(embedder.embed(["rice blast spots"])[0], "en")
    scores = store.all_scores(q)
    best = store.search(q, k=1)[0]
    assert scores.shape == (len(SAMPLE_CHUNKS),)
    assert np.isclose(scores.max(), best.score, atol=1e-5)


def test_save_and_load_roundtrip(tmp_path, store, embedder):
    store.save(tmp_path)
    loaded = FaissVectorStore.load(tmp_path, expected_model=embedder.model_name)
    assert loaded.size == store.size
    assert loaded.chunks[5].page_number == 3
    assert set(loaded.centroids) == {"en", "ta"}
    q = embedder.embed(["குலைநோய் அறிகுறிகள்"])[0]
    a = store.search(store.transform_query(q, "ta"), k=2)
    b = loaded.search(loaded.transform_query(q, "ta"), k=2)
    assert [h.chunk.chunk_id for h in a] == [h.chunk.chunk_id for h in b]


def test_load_missing_index(tmp_path):
    with pytest.raises(IndexNotFoundError):
        FaissVectorStore.load(tmp_path)


def test_load_corrupt_index(tmp_path, store):
    store.save(tmp_path)
    (tmp_path / "index.faiss").write_bytes(b"corrupted")
    with pytest.raises(IndexCorruptError):
        FaissVectorStore.load(tmp_path)


def test_load_inconsistent_chunks(tmp_path, store):
    store.save(tmp_path)
    lines = (tmp_path / CHUNKS_FILE).read_text(encoding="utf-8").splitlines()
    (tmp_path / CHUNKS_FILE).write_text("\n".join(lines[:-1]), encoding="utf-8")
    with pytest.raises(IndexCorruptError):
        FaissVectorStore.load(tmp_path)


def test_model_mismatch_is_detected(tmp_path, store):
    store.save(tmp_path)
    with pytest.raises(IndexCorruptError, match="Rebuild"):
        FaissVectorStore.load(tmp_path, expected_model="some-other-model")


def test_dimension_mismatch_raises(store):
    with pytest.raises(ValueError):
        store.search(np.ones(7, dtype="float32"))


def test_build_rejects_misaligned_input():
    with pytest.raises(ValueError):
        FaissVectorStore.build(np.ones((2, 4), dtype="float32"), SAMPLE_CHUNKS[:1])
