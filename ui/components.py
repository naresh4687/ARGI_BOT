"""Reusable rendering helpers and styles for the Streamlit UI."""

from __future__ import annotations

import re
from typing import Any

import streamlit as st

from chatbot.language import t

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Tamil:wght@400;600&family=Noto+Sans:wght@400;600&display=swap');

:root {
  --agri-green: #2e7d32;
  --agri-green-dark: #1b5e20;
  --agri-green-soft: #eef5ea;
  --agri-border: #d6e4d0;
  --agri-text-muted: #55655a;
}
html, body, [class*="css"], .stMarkdown, .stChatMessage, button, input, textarea {
  font-family: 'Noto Sans', 'Noto Sans Tamil', 'Latha', 'Nirmala UI', system-ui, sans-serif;
}
/* Tamil glyphs have tall ascenders/descenders - give lines more room. */
.stMarkdown p, .stMarkdown li { line-height: 1.7; }
.block-container { padding-top: 1.6rem; max-width: 880px; }

.agri-header { display: flex; align-items: center; gap: .75rem; margin-bottom: .25rem; }
.agri-header h1 { color: var(--agri-green-dark); font-size: 1.9rem; margin: 0; padding: 0; }
.agri-header .agri-logo { font-size: 2rem; line-height: 1; }
.agri-tagline { color: var(--agri-text-muted); margin: 0 0 1rem 0; }

.agri-card {
  border: 1px solid var(--agri-border); background: var(--agri-green-soft);
  border-radius: 12px; padding: 1rem 1.2rem; margin: .5rem 0 1rem 0;
}
.agri-card h3 { margin-top: 0; color: var(--agri-green-dark); font-size: 1.15rem; }

.agri-badge {
  display: inline-block; font-size: .75rem; font-weight: 600; border-radius: 999px;
  padding: .1rem .6rem; margin: 0 .35rem .35rem 0; border: 1px solid transparent;
}
.agri-badge.ai { background: #e3f2e1; color: #1b5e20; border-color: #b9d8b4; }
.agri-badge.offline { background: #fff4e0; color: #7a4a00; border-color: #f0d29a; }
.agri-badge.none { background: #f1f1f1; color: #444; border-color: #ddd; }
.agri-badge.error { background: #fdecea; color: #8a1c13; border-color: #f3c2bd; }

.agri-source { border-left: 3px solid var(--agri-green); padding: .2rem 0 .2rem .8rem; margin: .6rem 0; }
.agri-source .meta { color: var(--agri-text-muted); font-size: .82rem; }

/* Larger tap targets for example buttons on phones. */
div[data-testid="stButton"] button { min-height: 2.75rem; height: auto; white-space: normal; }
/* Streamlit truncates button labels with an ellipsis; let long questions wrap. */
div[data-testid="stButton"] button p,
div[data-testid="stButton"] button div {
  white-space: normal; overflow: visible; text-overflow: clip; line-height: 1.4;
}
@media (max-width: 640px) {
  .agri-header h1 { font-size: 1.5rem; }
  /* Clear Streamlit's fixed top bar (sidebar toggle) on small screens. */
  .block-container { padding-top: 3.75rem; padding-left: 1rem; padding-right: 1rem; }
  /* Streamlit's chat container auto-scrolls to the bottom, so a welcome screen
     taller than the phone pushes the title under the top bar. Keep it compact. */
  .agri-tagline { font-size: .85rem; margin-bottom: .5rem; }
  .agri-card { padding: .7rem .9rem; margin: .25rem 0 .6rem 0; }
  .agri-card h3 { font-size: 1.05rem; margin-bottom: .3rem; }
  .agri-card p { font-size: .9rem; line-height: 1.5; margin-bottom: 0; }
  div[data-testid="stButton"] button { min-height: 2.4rem; }
}
</style>
"""

_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]<>()#+\-!|~])")


def inject_styles() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def escape_md(text: str) -> str:
    """Escape Markdown so untrusted source text renders literally."""
    return _MD_SPECIAL.sub(r"\\\1", text)


def header() -> None:
    st.markdown(
        '<div class="agri-header"><span class="agri-logo" aria-hidden="true">🌾</span>'
        "<h1>AgriBot AI</h1></div>"
        '<p class="agri-tagline">Intelligent Bilingual Agriculture Assistant · English &amp; தமிழ்</p>',
        unsafe_allow_html=True,
    )


_BADGES = {
    "answered": ("ai", "AI answer · sources cited"),
    "unverified": ("error", "AI answer · no valid citation"),
    "offline": ("offline", "Offline demo · not AI-generated"),
    "insufficient": ("none", "No verified information"),
    "error": ("error", "Service error"),
    "invalid_input": ("none", "Input problem"),
}


def status_badge(status: str) -> None:
    css, label = _BADGES.get(status, ("none", status))
    st.markdown(f'<span class="agri-badge {css}">{label}</span>', unsafe_allow_html=True)


def render_response(resp: dict[str, Any]) -> None:
    """Render an assistant response dict (see ChatResponse)."""
    lang = resp.get("language", "en")
    status_badge(resp["status"])
    st.markdown(resp["answer"])
    for notice in resp.get("notices", []):
        st.info(notice, icon="ℹ️")
    for warning in resp.get("warnings", []):
        st.warning(warning, icon="⚠️")
    if resp.get("safety_note"):
        st.caption("🛡️ " + t(lang, "safety_note"))
    sources = resp.get("sources") or []
    if sources:
        render_sources(sources, lang)
    if resp.get("latency_ms"):
        st.caption(f"⏱ {resp['latency_ms'] / 1000:.1f}s · mode: {resp.get('mode', '')}")


def render_sources(sources: list[dict[str, Any]], lang: str) -> None:
    cited = [s for s in sources if s.get("cited")]
    with st.expander(f"📚 {t(lang, 'sources')} ({len(cited) or len(sources)})", expanded=False):
        for src in sources:
            title = escape_md(src["title"])
            heading = f"**[{src['label']}]** [{title}]({src['url']})" if src.get("url") else f"**[{src['label']}]** {title}"
            if not src.get("cited"):
                heading += " — _retrieved, not cited_"
            st.markdown(heading)
            meta = []
            if src.get("publisher"):
                meta.append(escape_md(src["publisher"]))
            if src.get("page_number"):
                meta.append(f"{t(lang, 'page')} {src['page_number']}")
            if src.get("language"):
                meta.append({"en": "English", "ta": "தமிழ்"}.get(src["language"], src["language"]))
            meta.append(f"{t(lang, 'relevance')} {src['score']:.2f}")
            st.caption(" · ".join(meta))
            excerpt = src["excerpt"]
            st.markdown("> " + escape_md(excerpt[:600] + ("…" if len(excerpt) > 600 else "")))
