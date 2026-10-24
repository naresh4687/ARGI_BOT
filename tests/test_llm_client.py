import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
import requests

from chatbot.llm_client import (
    AuthenticationError,
    InvalidResponseError,
    MissingCredentialsError,
    OllamaProvider,
    OpenAICompatibleProvider,
    ProviderTimeoutError,
    ProviderUnavailableError,
    QuotaExceededError,
    create_provider,
)
from config.settings import Settings

SECRET = "sk-test-SECRET-123456"
MESSAGES = [{"role": "user", "content": "hi"}]


class FakeResponse:
    def __init__(self, status=200, payload=None, text=None, headers=None):
        self.status_code = status
        self._payload = payload
        self._text = text
        self.headers = headers or {}

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeSession:
    """Returns queued responses / raises queued exceptions."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def ok(content="Answer [S1]"):
    return FakeResponse(200, {"choices": [{"message": {"content": content}}]})


def make(session, retries=2):
    return OpenAICompatibleProvider(
        api_key=SECRET, model="test-model", base_url="https://api.example/v1/", max_retries=retries,
        session=session, sleep=lambda s: None,
    )


def test_missing_credentials():
    with pytest.raises(MissingCredentialsError):
        OpenAICompatibleProvider(api_key="", model="m", base_url="https://x")
    with pytest.raises(MissingCredentialsError):
        create_provider(Settings(llm_provider="openai_compatible", llm_api_key="", llm_model="m"))


def test_offline_mode_creates_no_provider():
    assert create_provider(Settings(llm_provider="offline")) is None


def test_successful_request_format():
    session = FakeSession(ok("Hello"))
    assert make(session).generate(MESSAGES) == "Hello"
    call = session.calls[0]
    assert call["url"] == "https://api.example/v1/chat/completions"
    assert call["json"]["model"] == "test-model" and call["json"]["messages"] == MESSAGES
    assert call["headers"]["Authorization"] == f"Bearer {SECRET}"
    assert call["timeout"] == 45.0


def test_quota_error_retried_then_raised():
    session = FakeSession(FakeResponse(429), FakeResponse(429), FakeResponse(429))
    with pytest.raises(QuotaExceededError) as exc:
        make(session, retries=2).generate(MESSAGES)
    assert len(session.calls) == 3
    assert "quota" in exc.value.user_message.lower()


def test_quota_retry_waits_for_short_retry_after():
    waits = []
    session = FakeSession(FakeResponse(429, headers={"retry-after": "12"}), ok("Hello"))
    provider = OpenAICompatibleProvider(
        api_key=SECRET, model="m", base_url="https://api.example/v1", max_retries=2,
        session=session, sleep=waits.append,
    )
    assert provider.generate(MESSAGES) == "Hello"
    assert waits == [12.0]


def test_quota_with_long_retry_after_is_not_retried():
    session = FakeSession(FakeResponse(429, headers={"retry-after": "70000"}), ok())
    with pytest.raises(QuotaExceededError):
        make(session, retries=2).generate(MESSAGES)
    assert len(session.calls) == 1


def test_transient_server_error_recovers():
    session = FakeSession(FakeResponse(503), ok("Recovered"))
    assert make(session).generate(MESSAGES) == "Recovered"


def test_timeout_maps_to_timeout_error():
    session = FakeSession(requests.Timeout("slow"), requests.Timeout("slow"))
    with pytest.raises(ProviderTimeoutError):
        make(session, retries=1).generate(MESSAGES)


def test_connection_error_maps_to_unavailable():
    session = FakeSession(requests.ConnectionError("down"))
    with pytest.raises(ProviderUnavailableError):
        make(session, retries=0).generate(MESSAGES)


def test_auth_error_not_retried():
    session = FakeSession(FakeResponse(401))
    with pytest.raises(AuthenticationError):
        make(session).generate(MESSAGES)
    assert len(session.calls) == 1


@pytest.mark.parametrize(
    "response",
    [FakeResponse(200, None), FakeResponse(200, {"choices": []}), FakeResponse(200, {"choices": [{"message": {"content": "  "}}]})],
)
def test_malformed_responses(response):
    with pytest.raises(InvalidResponseError):
        make(FakeSession(response)).generate(MESSAGES)


def test_api_key_never_logged_or_exposed(caplog):
    caplog.set_level(logging.DEBUG)
    provider = make(FakeSession(FakeResponse(500), FakeResponse(500), FakeResponse(500)))
    with pytest.raises(ProviderUnavailableError) as exc:
        provider.generate(MESSAGES)
    assert SECRET not in caplog.text
    assert SECRET not in repr(provider)
    assert SECRET not in str(exc.value) and SECRET not in exc.value.user_message
    assert SECRET not in str(Settings(llm_api_key=SECRET).redacted())


def test_ollama_unreachable_message():
    provider = OllamaProvider(model="llama3.1:8b", base_url="http://localhost:9", max_retries=0,
                              session=FakeSession(requests.ConnectionError("refused")), sleep=lambda s: None)
    with pytest.raises(ProviderUnavailableError) as exc:
        provider.generate(MESSAGES)
    assert "ollama serve" in exc.value.user_message


def test_ollama_success():
    session = FakeSession(FakeResponse(200, {"message": {"content": "Vanakkam"}}))
    provider = OllamaProvider(model="m", session=session, sleep=lambda s: None)
    assert provider.generate(MESSAGES) == "Vanakkam"
    assert session.calls[0]["url"].endswith("/api/chat") and session.calls[0]["json"]["stream"] is False
    options = session.calls[0]["json"]["options"]
    assert options["num_ctx"] == 8192 and options["num_predict"] == 700


def test_ollama_warns_when_context_nearly_full(caplog):
    payload = {"message": {"content": "ok"}, "prompt_eval_count": 4000}
    provider = OllamaProvider(model="m", num_ctx=4096, max_tokens=700,
                              session=FakeSession(FakeResponse(200, payload)), sleep=lambda s: None)
    provider.generate(MESSAGES)
    assert "may have been truncated" in caplog.text


# ---- real HTTP round-trip against a local fake OpenAI-compatible server ----
class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.headers.get("Authorization") != f"Bearer {SECRET}":
            self.send_response(401)
            self.end_headers()
            return
        reply = {"choices": [{"message": {"content": f"echo:{body['messages'][-1]['content']}"}}]}
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def local_server():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/v1"
    server.shutdown()


def test_http_round_trip(local_server):
    provider = OpenAICompatibleProvider(api_key=SECRET, model="m", base_url=local_server, timeout=5)
    assert provider.generate([{"role": "user", "content": "நெல்"}]) == "echo:நெல்"


def test_http_wrong_key(local_server):
    provider = OpenAICompatibleProvider(api_key="wrong", model="m", base_url=local_server, timeout=5)
    with pytest.raises(AuthenticationError):
        provider.generate(MESSAGES)
