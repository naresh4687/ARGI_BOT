import pytest

from chatbot.language import EXAMPLE_QUESTIONS, UI_TEXT, detect_language, t


@pytest.mark.parametrize(
    "text, expected",
    [
        ("How do I control stem borer in paddy?", "en"),
        ("நெல்லில் தண்டுத் துளைப்பான் கட்டுப்பாடு", "ta"),
        ("நெல்லில் urea எப்போது போட வேண்டும்?", "ta"),  # mixed Tamil + English word
        ("धान में कीट नियंत्रण कैसे करें?", "other"),  # Hindi
        ("వరి పంటలో తెగులు", "other"),  # Telugu
        ("12345 ???", "unknown"),
        ("", "unknown"),
    ],
)
def test_detect_language(text, expected):
    assert detect_language(text) == expected


def test_ui_strings_complete_in_both_languages():
    assert set(UI_TEXT["en"]) == set(UI_TEXT["ta"])
    for key in UI_TEXT["ta"]:
        assert UI_TEXT["ta"][key].strip()


def test_tamil_ui_strings_are_tamil_script():
    for key in ("welcome_title", "insufficient", "safety_note", "thinking"):
        assert detect_language(UI_TEXT["ta"][key]) == "ta"


def test_examples_match_their_language():
    assert all(detect_language(q) == "en" for q in EXAMPLE_QUESTIONS["en"])
    assert all(detect_language(q) == "ta" for q in EXAMPLE_QUESTIONS["ta"])


def test_translation_helper_formats_and_falls_back():
    assert "1500" in t("en", "too_long", length=1500, limit=1000)
    assert "1500" in t("ta", "too_long", length=1500, limit=1000)
    assert t("fr", "empty") == UI_TEXT["en"]["empty"]
