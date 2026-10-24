"""Chat page: history, example questions, input handling and processing."""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict
from typing import Callable

import streamlit as st

from chatbot.language import EXAMPLE_QUESTIONS, t
from chatbot.response_service import ChatResponse
from ui.components import escape_md, render_response

DUPLICATE_WINDOW_SECONDS = 5.0

AnswerFn = Callable[[str, str], ChatResponse]


def init_state() -> None:
    defaults = {"messages": [], "pending": None, "processing": False, "last_request": None}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _queue(question: str) -> None:
    st.session_state.pending = question


def _is_duplicate(question: str, language: str) -> bool:
    """True if the same question was just processed (double click / resubmit)."""
    digest = hashlib.sha256(f"{language}:{question.strip()}".encode()).hexdigest()
    last = st.session_state.last_request
    now = time.monotonic()
    if last and last[0] == digest and now - last[1] < DUPLICATE_WINDOW_SECONDS:
        return True
    st.session_state.last_request = (digest, now)
    return False


def render_welcome(language: str) -> None:
    st.markdown(
        f'<div class="agri-card"><h3>{t(language, "welcome_title")}</h3>'
        f"<p>{t(language, 'welcome_body')}</p></div>",
        unsafe_allow_html=True,
    )
    st.markdown(f"**{t(language, 'examples')}**")
    cols = st.columns(2)
    for i, question in enumerate(EXAMPLE_QUESTIONS[language]):
        cols[i % 2].button(
            question,
            key=f"example_{language}_{i}",
            on_click=_queue,
            args=(question,),
            use_container_width=True,
            disabled=st.session_state.processing,
        )


def render_chat(answer_fn: AnswerFn, language: str) -> None:
    """Render the conversation and process at most one pending question."""
    if not st.session_state.messages:
        render_welcome(language)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑‍🌾" if msg["role"] == "user" else "🌾"):
            if msg["role"] == "user":
                st.markdown(escape_md(msg["content"]))
            else:
                render_response(msg["response"])

    submitted = st.chat_input(t(language, "input_placeholder"), disabled=st.session_state.processing)
    if submitted is not None:
        _queue(submitted)

    question = st.session_state.pending
    if question is None:
        return
    st.session_state.pending = None
    if st.session_state.processing:
        st.toast(t(language, "busy"))
        return
    if not question.strip():
        st.toast(t(language, "empty"))
        return
    if _is_duplicate(question, language):
        return

    with st.chat_message("user", avatar="🧑‍🌾"):
        st.markdown(escape_md(question))
    st.session_state.processing = True
    try:
        with st.chat_message("assistant", avatar="🌾"):
            with st.spinner(t(language, "thinking")):
                response = answer_fn(question, language)
            render_response(asdict(response))
    finally:
        st.session_state.processing = False
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append({"role": "assistant", "response": asdict(response)})
    st.rerun()
