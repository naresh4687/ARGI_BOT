import pytest

from chatbot.retriever import LanguageProfile, Retriever
from knowledge_base.lexical import LexicalIndex, char_ngrams


def test_retrieves_relevant_passage_with_metadata(retriever):
    passages = retriever.retrieve("What are the symptoms of rice blast?", "en")
    assert passages, "expected at least one passage"
    top = passages[0]
    assert top.chunk.document_id == "rice-blast"
    assert top.label == "S1"
    assert top.chunk.source_url == "https://example.org/rice-blast"
    assert [p.label for p in passages] == [f"S{i + 1}" for i in range(len(passages))]


def test_tamil_query_retrieves_tamil_passage(retriever):
    passages = retriever.retrieve("குலைநோயின் அறிகுறிகள் என்ன?", "ta")
    assert passages and passages[0].chunk.document_id == "rice-blast-ta"


def test_threshold_filters_irrelevant_queries(retriever):
    assert retriever.retrieve("Who won the football match yesterday?", "en") == []


def test_threshold_enforced_in_code(embedder, store):
    strict = Retriever(embedder, store, top_k=3, profiles={"en": LanguageProfile(0.5, 0.99), "ta": LanguageProfile(0.2, 0.99)})
    assert strict.retrieve("drip irrigation", "en") == []


def test_max_passages_per_document(embedder, store):
    r = Retriever(embedder, store, top_k=5, profiles={"en": LanguageProfile(0.5, -1.0), "ta": LanguageProfile(0.2, -1.0)}, max_per_document=2)
    passages = r.retrieve("drip irrigation water emitters weeds", "en")
    assert sum(p.chunk.document_id == "drip" for p in passages) <= 2


def test_relative_cutoff_drops_weak_passages_from_other_documents(embedder, store):
    open_profiles = {"en": LanguageProfile(0.5, -1.0), "ta": LanguageProfile(0.2, -1.0)}
    loose = Retriever(embedder, store, top_k=5, profiles=open_profiles, relative_cutoff=0.0)
    strict = Retriever(embedder, store, top_k=5, profiles=open_profiles, relative_cutoff=0.99)
    query = "drip irrigation water emitters weeds"
    assert {p.chunk.document_id for p in loose.retrieve(query, "en")} - {"drip"}
    kept = strict.retrieve(query, "en")
    # Passages from the best document are always kept, others are not.
    assert len(kept) == 2 and {p.chunk.document_id for p in kept} == {"drip"}


def test_page_number_carried_through(retriever):
    passages = retriever.retrieve("companion planting onions carrots pests", "en")
    assert passages[0].chunk.page_number == 3


def test_empty_query(retriever):
    assert retriever.retrieve("   ") == []


def test_char_ngrams_share_tamil_stems():
    a = set(char_ngrams("பாசனத்தின்"))
    b = set(char_ngrams("பாசனம்"))
    assert a & b


def test_lexical_index_scores():
    index = LexicalIndex(["drip irrigation saves water", "rice blast disease", "சொட்டு நீர்ப் பாசனம்"])
    scores = index.scores("சொட்டு நீர்ப் பாசனத்தின் நன்மைகள்")
    assert max(scores, key=scores.get) == 2
    assert index.scores("zzzz qqqq") == {}


# --------------------------------------------------------------------------- #
# Real model + real index (skipped automatically if not available)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def real_retriever():
    from chatbot.bootstrap import build_service
    from config.settings import Settings

    settings = Settings()  # defaults, independent of local .env
    return build_service(settings).retriever


@pytest.mark.model
@pytest.mark.parametrize(
    "question, expected_prefix",
    [
        ("What are the symptoms of rice blast disease?", "tnau-rice-disease-blast"),
        ("What are the advantages of drip irrigation?", "tnau-irrigation-drip"),
        ("What is integrated nutrient management?", "tnau-nutrient-inm"),
        ("நெல்லில் குலைநோயின் அறிகுறிகள் என்ன?", "tnau-rice-disease-blast"),
        ("ஒருங்கிணைந்த பூச்சி மேலாண்மையின் கூறுகள் யாவை?", "tnau-ipm"),
        ("திருந்திய நெல் சாகுபடியில் நாற்றங்கால் எப்படி தயார் செய்வது?", "tnau-rice-sri-nursery"),
    ],
)
def test_real_retrieval_english_and_tamil(real_retriever, question, expected_prefix):
    passages = real_retriever.retrieve(question)
    assert passages, f"no passage above threshold for {question!r}"
    assert any(p.chunk.document_id.startswith(expected_prefix) for p in passages)
    for p in passages:
        assert p.chunk.source_url and p.chunk.source_url.startswith("https://agritech.tnau.ac.in/")


@pytest.mark.model
@pytest.mark.parametrize("question", ["Who won the cricket world cup in 2011?", "Write a short poem about the moon."])
def test_real_retrieval_rejects_off_topic(real_retriever, question):
    assert real_retriever.retrieve(question) == []
