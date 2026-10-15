"""Explore the catalogue: search, simple filters and a card grid."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.knowledge_base import ADDRESSES_THRESHOLD
from ui.components import course_grid, page_header, snippet
from ui.state import get_engine

PAGE_SIZE = 12
ANY = "Any"
RATING_OPTIONS = {ANY: None, "4.0+": 4.0, "4.5+": 4.5, "4.7+": 4.7}
SORT_OPTIONS = ["Highest rated", "Title A–Z"]
PAGE_KEY = "explore_shown"
FILTER_STATE_KEY = "explore_filters"


def filter_courses(courses: pd.DataFrame, skill_matrix: pd.DataFrame, query: str, difficulties: list[str],
                   providers: list[str], domains: list[str], skills: list[str], min_rating: float,
                   include_unrated: bool) -> pd.DataFrame:
    """Apply the Explore filters. Every word of ``query`` must appear in the title, tags or description."""
    mask = pd.Series(True, index=courses.index)
    if query.strip():
        haystack = (courses["title"] + " " + courses["skills"].map(" ".join) + " " + courses["description"]).str.lower()
        for word in query.lower().split():
            mask &= haystack.str.contains(word, regex=False)
    if difficulties:
        mask &= courses["difficulty"].isin(difficulties)
    if providers:
        mask &= courses["provider"].isin(providers)
    if domains:
        mask &= courses["domain"].isin(domains)
    if skills:
        teaches = (skill_matrix.loc[courses["course_id"], skills] >= ADDRESSES_THRESHOLD).any(axis=1).to_numpy()
        mask &= teaches
    rating_ok = courses["rating"] >= min_rating
    mask &= rating_ok | (courses["rating"].isna() & include_unrated)
    return courses[mask]


def render() -> None:
    engine = get_engine()
    courses = engine.courses
    page_header("Explore", "Browse the catalogue", "Search every course and narrow it down by level, subject or skill.")

    query = st.text_input("Search", placeholder="Search courses, e.g. deep learning tensorflow",
                          label_visibility="collapsed", icon=":material/search:")
    level_col, subject_col, skill_col, rating_col, sort_col = st.columns(5)
    level = level_col.selectbox("Level", [ANY, "Beginner", "Intermediate", "Advanced", "Conversant"])
    domains = sorted(d for d in courses["domain"].unique() if d)
    subject = subject_col.selectbox("Subject", [ANY] + domains,
                                    format_func=lambda d: d if d == ANY else d.replace("-", " ").capitalize())
    skill = skill_col.selectbox("Teaches", [ANY] + engine.knowledge.skill_names)
    rating = rating_col.selectbox("Rating", list(RATING_OPTIONS))
    sort = sort_col.selectbox("Sort by", SORT_OPTIONS)

    min_rating = RATING_OPTIONS[rating]
    results = filter_courses(
        courses, engine.skill_matrix, query,
        difficulties=[] if level == ANY else [level], providers=[],
        domains=[] if subject == ANY else [subject], skills=[] if skill == ANY else [skill],
        min_rating=min_rating or 1.0, include_unrated=min_rating is None,
    )
    if sort == "Highest rated":
        results = results.sort_values(["rating", "title"], ascending=[False, True], na_position="last")
    else:
        results = results.sort_values("title")

    # Start from the first page whenever the filters change.
    filters = (query, level, subject, skill, rating, sort)
    if st.session_state.get(FILTER_STATE_KEY) != filters:
        st.session_state[FILTER_STATE_KEY] = filters
        st.session_state[PAGE_KEY] = PAGE_SIZE
    shown = st.session_state.get(PAGE_KEY, PAGE_SIZE)

    if results.empty:
        st.info("No courses match. Try fewer words or a broader filter.")
        return
    st.markdown(f'<div class="sb-results-meta">{len(results):,} courses</div>', unsafe_allow_html=True)
    page = results.head(shown)
    course_grid(page, descriptions={cid: snippet(text, 150) for cid, text in zip(page["course_id"], page["description"])})
    if shown < len(results):
        _, middle, _ = st.columns([2, 1, 2])
        if middle.button("Show more", width="stretch"):
            st.session_state[PAGE_KEY] = shown + PAGE_SIZE
            st.rerun()
