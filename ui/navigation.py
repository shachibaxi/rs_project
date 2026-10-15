"""Lets pages link to each other: app.py creates its st.Page objects through ``page``."""

from __future__ import annotations

import streamlit as st

_PAGES: dict[str, st.Page] = {}


def page(render, title: str, icon: str | None, url_path: str, default: bool = False) -> st.Page:
    """Create an st.Page and remember it under ``url_path`` for links."""
    created = st.Page(render, title=title, icon=icon, url_path=url_path, default=default)
    _PAGES[url_path] = created
    return created


def link(url_path: str, label: str, icon: str | None = None) -> None:
    if url_path in _PAGES:
        st.page_link(_PAGES[url_path], label=label, icon=icon)


def go(url_path: str) -> None:
    """Switch to a registered page (no-op when the page runs on its own, e.g. in tests)."""
    if url_path in _PAGES:
        st.switch_page(_PAGES[url_path])


def next_step(url_path: str, label: str) -> None:
    """A right-aligned button that moves the learner to the next step of the journey."""
    st.markdown('<div style="height:0.6rem"></div>', unsafe_allow_html=True)
    _, right = st.columns([3, 1.2])
    if right.button(label, type="primary", icon=":material/arrow_forward:", icon_position="right",
                    width="stretch", key=f"next_{url_path}"):
        go(url_path)
