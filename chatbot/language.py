"""Language identification and bilingual UI strings.

Tamil uses its own Unicode block (U+0B80–U+0BFF), so script ratios identify
Tamil text reliably without a statistical language-ID model. Text in other
non-Latin scripts (e.g. Hindi, Telugu) is reported as ``"other"`` so the app
can say clearly that only English and Tamil are supported.
"""

from __future__ import annotations

import re
import unicodedata

SUPPORTED_LANGUAGES = {"en": "English", "ta": "தமிழ் (Tamil)"}

_TAMIL = re.compile(r"[஀-௿]")
_LATIN = re.compile(r"[A-Za-z]")
# Letters from other Indic and common non-Latin scripts.
_OTHER = re.compile(
    r"[ऀ-୿ఀ-෿؀-ۿЀ-ӿ一-鿿぀-ヿ가-힯]"
)


def script_counts(text: str) -> dict[str, int]:
    """Count Tamil, Latin and other-script letters in ``text``."""
    return {
        "ta": len(_TAMIL.findall(text)),
        "en": len(_LATIN.findall(text)),
        "other": len(_OTHER.findall(text)),
    }


def detect_language(text: str) -> str:
    """Return ``"ta"``, ``"en"``, ``"other"`` or ``"unknown"`` for ``text``.

    Mixed Tamil/English text (common in farmer queries, e.g. Tamil with an
    English product name) is classified as Tamil if Tamil letters make up at
    least 30% of the letters.
    """
    counts = script_counts(unicodedata.normalize("NFC", text))
    total = sum(counts.values())
    if total == 0:
        return "unknown"
    if counts["other"] / total > 0.5:
        return "other"
    if counts["ta"] / total >= 0.3:
        return "ta"
    if counts["en"] > 0:
        return "en"
    return "other"


UI_TEXT: dict[str, dict[str, str]] = {
    "en": {
        "welcome_title": "Welcome to AgriBot AI",
        "welcome_body": (
            "Ask farming questions about crops, soil, irrigation, fertilizers, pests "
            "and diseases. Answers are based on a small set of verified Tamil Nadu "
            "Agricultural University (TNAU) documents, with sources shown below each answer."
        ),
        "examples": "Try an example question",
        "input_placeholder": "Type your farming question…",
        "thinking": "Searching verified sources and preparing an answer…",
        "sources": "Sources",
        "relevance": "relevance",
        "page": "page",
        "no_sources": "No verified source passage was relevant enough to answer this question.",
        "clear": "Clear chat",
        "language": "Answer language",
        "mismatch": "You wrote in {detected} but the selected answer language is {selected}. "
        "The answer will be in {selected}; change the language in the sidebar if needed.",
        "unsupported_language": "AgriBot AI currently supports only English and Tamil. "
        "Please ask your question in English or Tamil.",
        "empty": "Please type a question.",
        "too_short": "Your question is too short. Please add more detail.",
        "too_long": "Your question is too long ({length} characters). Please keep it under {limit} characters.",
        "busy": "Still working on your previous question. Please wait.",
        "insufficient": (
            "I could not find verified information about this in my knowledge base. "
            "Please rephrase with the crop name and the problem you see, or contact your "
            "nearest Krishi Vigyan Kendra (KVK) or agricultural extension officer."
        ),
        "safety_note": (
            "Safety: always follow the product label and local agricultural department or "
            "TNAU guidance before using any pesticide or fertilizer. Diagnoses from text "
            "descriptions are not certain; consult a local agricultural officer if unsure."
        ),
        "offline_label": "Offline demo mode — no AI generation. Showing the most relevant verified passages.",
        "offline_lang_note": "Offline mode cannot translate. Some passages below are in English.",
        "error_prefix": "Sorry, the answer could not be generated.",
    },
    "ta": {
        "welcome_title": "அக்ரிபாட் AI-க்கு வரவேற்கிறோம்",
        "welcome_body": (
            "பயிர், மண், நீர்ப் பாசனம், உரம், பூச்சி மற்றும் நோய்கள் பற்றிய விவசாயக் கேள்விகளைக் "
            "கேளுங்கள். பதில்கள் தமிழ்நாடு வேளாண்மைப் பல்கலைக்கழகத்தின் (TNAU) சரிபார்க்கப்பட்ட "
            "ஆவணங்களை அடிப்படையாகக் கொண்டவை; ஒவ்வொரு பதிலின் கீழும் ஆதாரங்கள் காட்டப்படும்."
        ),
        "examples": "எடுத்துக்காட்டுக் கேள்விகள்",
        "input_placeholder": "உங்கள் கேள்வியைக் கேளுங்கள்…",
        "thinking": "சரிபார்க்கப்பட்ட ஆதாரங்களைத் தேடி பதில் தயாரிக்கப்படுகிறது…",
        "sources": "ஆதாரங்கள்",
        "relevance": "பொருத்தம்",
        "page": "பக்கம்",
        "no_sources": "இந்தக் கேள்விக்குப் பதிலளிக்கப் போதுமான பொருத்தமான ஆதாரம் கிடைக்கவில்லை.",
        "clear": "உரையாடலை அழி",
        "language": "பதில் மொழி",
        "mismatch": "நீங்கள் {detected} மொழியில் எழுதியுள்ளீர்கள்; தேர்ந்தெடுத்த பதில் மொழி {selected}. "
        "பதில் {selected} மொழியில் வழங்கப்படும்; தேவைப்பட்டால் பக்கப்பட்டியில் மொழியை மாற்றவும்.",
        "unsupported_language": "அக்ரிபாட் AI தற்போது ஆங்கிலம் மற்றும் தமிழ் மொழிகளை மட்டுமே ஆதரிக்கிறது. "
        "தயவுசெய்து ஆங்கிலம் அல்லது தமிழில் கேட்கவும்.",
        "empty": "தயவுசெய்து ஒரு கேள்வியைத் தட்டச்சு செய்யவும்.",
        "too_short": "உங்கள் கேள்வி மிகவும் சிறியது. இன்னும் விவரம் சேர்க்கவும்.",
        "too_long": "உங்கள் கேள்வி மிக நீளமானது ({length} எழுத்துகள்). {limit} எழுத்துகளுக்குள் கேட்கவும்.",
        "busy": "முந்தைய கேள்விக்கான பதில் தயாராகிறது. சற்று காத்திருக்கவும்.",
        "insufficient": (
            "இதைப் பற்றிய சரிபார்க்கப்பட்ட தகவல் என் அறிவுத் தளத்தில் கிடைக்கவில்லை. பயிரின் பெயர் "
            "மற்றும் நீங்கள் காணும் பிரச்சனையுடன் மீண்டும் கேட்கவும், அல்லது அருகிலுள்ள வேளாண் அறிவியல் "
            "மையம் (KVK) / வேளாண் விரிவாக்க அலுவலரைத் தொடர்பு கொள்ளவும்."
        ),
        "safety_note": (
            "பாதுகாப்பு: எந்தப் பூச்சிக்கொல்லி அல்லது உரத்தையும் பயன்படுத்தும் முன் அதன் லேபிள் "
            "வழிமுறைகளையும் உள்ளூர் வேளாண் துறை / TNAU பரிந்துரைகளையும் பின்பற்றவும். விவரிப்பின் "
            "அடிப்படையிலான நோய் கண்டறிதல் உறுதியானதல்ல; சந்தேகம் இருந்தால் வேளாண் அலுவலரை அணுகவும்."
        ),
        "offline_label": "ஆஃப்லைன் டெமோ முறை — AI உருவாக்கம் இல்லை. மிகவும் பொருத்தமான சரிபார்க்கப்பட்ட பகுதிகள் காட்டப்படுகின்றன.",
        "offline_lang_note": "ஆஃப்லைன் முறையில் மொழிபெயர்ப்பு இல்லை. கீழே உள்ள சில பகுதிகள் ஆங்கிலத்தில் உள்ளன.",
        "error_prefix": "மன்னிக்கவும், பதிலை உருவாக்க முடியவில்லை.",
    },
}

EXAMPLE_QUESTIONS: dict[str, list[str]] = {
    "en": [
        "What are the symptoms of rice blast disease?",
        "What are the advantages of drip irrigation?",
        "How can I improve fertilizer use efficiency?",
        "What are the main components of integrated pest management?",
    ],
    "ta": [
        "நெல்லில் குலைநோயின் அறிகுறிகள் என்ன?",
        "சொட்டு நீர்ப் பாசனத்தின் நன்மைகள் என்ன?",
        "திருந்திய நெல் சாகுபடியில் நாற்றங்கால் எப்படி தயார் செய்வது?",
        "ஒருங்கிணைந்த ஊட்டச்சத்து மேலாண்மை என்றால் என்ன?",
    ],
}


def t(language: str, key: str, **kwargs: object) -> str:
    """Return the UI string ``key`` in ``language`` (falls back to English)."""
    table = UI_TEXT.get(language, UI_TEXT["en"])
    text = table.get(key) or UI_TEXT["en"][key]
    return text.format(**kwargs) if kwargs else text


def language_name(code: str) -> str:
    """Human-readable language name for a code."""
    return {"en": "English", "ta": "Tamil", "other": "another language"}.get(code, code)
