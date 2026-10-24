"""Streamlit UI smoke tests using the official AppTest harness (offline mode,
real embedding model and index)."""

from pathlib import Path

import pytest

APP = str(Path(__file__).resolve().parent.parent / "app.py")
pytestmark = pytest.mark.model


@pytest.fixture
def app(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("LLM_PROVIDER", "offline")
    return AppTest.from_file(APP, default_timeout=180).run()


def _markdown(msg):
    return " ".join(m.value for m in msg.markdown)


def test_app_starts_with_welcome_and_examples(app):
    assert not app.exception
    labels = [b.label for b in app.button]
    assert "What are the symptoms of rice blast disease?" in labels


def test_example_question_answers_with_sources(app):
    next(b for b in app.button if b.label.startswith("What are the symptoms of rice blast")).click().run()
    assert not app.exception
    assert len(app.chat_message) == 2
    reply = _markdown(app.chat_message[1])
    assert "Offline demo" in reply
    assert "agritech.tnau.ac.in" in reply  # source link from metadata


def test_off_topic_question_gets_uncertainty_response(app):
    app.chat_input[0].set_value("Who won the cricket world cup in 2011?").run()
    assert "could not find verified information" in _markdown(app.chat_message[-1])


def test_tamil_selection_and_clear_chat(app):
    app.sidebar.radio[1].set_value("ta").run()
    app.chat_input[0].set_value("சொட்டு நீர்ப் பாசனத்தின் நன்மைகள் என்ன?").run()
    assert not app.exception
    assert "ஆஃப்லைன்" in _markdown(app.chat_message[-1])
    clear = next(b for b in app.sidebar.button if "உரையாடலை அழி" in b.label)
    clear.click().run()
    assert len(app.chat_message) == 0


def test_duplicate_submission_processed_once(app):
    app.chat_input[0].set_value("What are the advantages of drip irrigation?").run()
    app.chat_input[0].set_value("What are the advantages of drip irrigation?").run()
    assert len(app.chat_message) == 2


def test_info_pages_render(app):
    for page in ("help", "about", "project"):
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception


def test_missing_credentials_falls_back_to_offline(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("LLM_MODEL", "some-model")
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception
    assert any("not configured" in w.value and "offline demo" in w.value for w in at.warning)
    at.chat_input[0].set_value("What are the symptoms of rice blast disease?").run()
    assert "Offline demo" in _markdown(at.chat_message[-1])


def test_malformed_key_falls_back_to_offline_without_crashing(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM_API_KEY", "gsk_P8d2…")  # key pasted with an ellipsis
    monkeypatch.setenv("LLM_MODEL", "some-model")
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception
    assert any("malformed" in w.value and "offline demo" in w.value for w in at.warning)
    assert all("gsk_P8d2" not in w.value for w in at.warning)
    at.chat_input[0].set_value("What are the symptoms of rice blast disease?").run()
    assert not at.exception


def test_provider_outage_shows_friendly_error_and_sources(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM_API_KEY", "test-key-not-real")
    monkeypatch.setenv("LLM_MODEL", "some-model")
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:9/v1")  # nothing listens here
    monkeypatch.setenv("LLM_MAX_RETRIES", "0")
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.chat_input[0].set_value("What are the symptoms of rice blast disease?").run()
    assert not at.exception
    reply = _markdown(at.chat_message[-1])
    assert "Service error" in reply and "unreachable" in reply
    assert "agritech.tnau.ac.in" in reply  # verified sources still shown
    assert "test-key-not-real" not in reply
