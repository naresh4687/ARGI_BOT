"""Integration test against a REAL LLM provider.

Skipped unless AGRIBOT_RUN_INTEGRATION=1 and a provider is configured
(LLM_PROVIDER=openai_compatible with LLM_API_KEY/LLM_MODEL, or
LLM_PROVIDER=ollama with a running server). This makes real network calls
and may consume API quota.
"""

import os

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.model]

if os.environ.get("AGRIBOT_RUN_INTEGRATION") != "1":
    pytest.skip("set AGRIBOT_RUN_INTEGRATION=1 to run real-provider tests", allow_module_level=True)


@pytest.fixture(scope="module")
def live_service():
    from chatbot.bootstrap import build_service
    from config.settings import Settings

    settings = Settings.from_env()
    if settings.llm_provider == "offline":
        pytest.skip("LLM_PROVIDER is offline")
    return build_service(settings)


@pytest.mark.parametrize(
    "question, language",
    [("What are the symptoms of rice blast disease?", "en"), ("நெல்லில் குலைநோயின் அறிகுறிகள் என்ன?", "ta")],
)
def test_live_grounded_answer(live_service, question, language):
    resp = live_service.answer(question, language)
    assert resp.status in ("answered", "unverified"), resp.answer
    assert resp.sources
    if language == "ta":
        from chatbot.language import detect_language

        assert detect_language(resp.answer) == "ta"
    print(f"\n[{language}] {resp.status} {resp.latency_ms}ms\n{resp.answer}\nwarnings={resp.warnings}")


def test_live_off_topic(live_service):
    resp = live_service.answer("Who won the cricket world cup in 2011?", "en")
    assert resp.status == "insufficient"
