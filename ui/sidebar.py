"""Sidebar: navigation, language selection, session controls and status."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from chatbot.language import SUPPORTED_LANGUAGES, t

PAGES = {
    "chat": "💬 Chat",
    "help": "❓ Help",
    "about": "🌱 About",
    "project": "🛠️ Project information",
}


@dataclass
class SidebarState:
    page: str
    language: str


def render_sidebar(mode_label: str, kb_summary: str) -> SidebarState:
    with st.sidebar:
        st.markdown("### 🌾 AgriBot AI")
        page = st.radio(
            "Navigation",
            list(PAGES),
            format_func=PAGES.get,
            key="page",
            label_visibility="collapsed",
        )
        st.divider()
        current = st.session_state.get("language", "en")
        language = st.radio(
            t(current, "language"),
            list(SUPPORTED_LANGUAGES),
            format_func=SUPPORTED_LANGUAGES.get,
            key="language",
            help="Answers are given in this language for the rest of the conversation.",
        )
        disabled = bool(st.session_state.get("processing")) or not st.session_state.get("messages")
        if st.button("🗑️ " + t(language, "clear"), use_container_width=True, disabled=disabled):
            st.session_state.messages = []
            st.session_state.pending = None
            st.rerun()
        st.divider()
        st.caption(f"**Mode:** {mode_label}")
        st.caption(f"**Knowledge base:** {kb_summary}")
        st.caption("Chat history is kept only for this browser session and is not saved.")
    return SidebarState(page=page, language=language)
