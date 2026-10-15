"""Home page: what SkillBridge does, with a one-click start from a career goal."""

from __future__ import annotations

import streamlit as st

from ui import navigation
from ui.components import section, stat_tiles
from ui.state import get_engine, get_profile, profile_is_empty, save_profile

STEPS = [
    ("01", "Your profile", "Pick a career goal and the skills you already have. It takes under a minute."),
    ("02", "Skill gaps", "See which skills the role needs, which you have, and what's missing."),
    ("03", "Courses for you", "Real Coursera courses ranked for your gaps, your interests and your level."),
    ("04", "Learning path", "Your gaps in the right order, with one course for every step."),
]
GO_KEY = "home_go_to_skill_gaps"


def _start_with_goal() -> None:
    goal = st.session_state.get("home_goal")
    if goal:
        profile = get_profile()
        profile.career_goal = goal
        save_profile(profile)
        st.session_state[GO_KEY] = True


def render() -> None:
    if st.session_state.pop(GO_KEY, False):
        navigation.go("skill-gap")

    engine = get_engine()
    courses = engine.courses
    profile = get_profile()

    st.markdown(
        """<div class="sb-hero"><h1>Know what to learn next.</h1>
        <p>SkillBridge turns your career goal into a clear plan: the skills you're missing, the courses that
        teach them, and the order to take them in.</p></div>""",
        unsafe_allow_html=True,
    )
    first, second, _ = st.columns([1.1, 1, 3])
    if first.button("Build my profile" if profile_is_empty(profile) else "Edit my profile", type="primary",
                    width="stretch"):
        navigation.go("profile")
    if second.button("Browse courses", width="stretch"):
        navigation.go("explore")

    st.pills("Or jump straight in — pick a career goal", list(engine.knowledge.careers), key="home_goal",
             on_change=_start_with_goal)

    stat_tiles([
        ("Coursera courses", f"{len(courses):,}"),
        ("Universities & companies", f"{courses['provider'].nunique():,}"),
        ("Career goals", str(len(engine.knowledge.careers))),
        ("Skills mapped", str(len(engine.knowledge.skills))),
    ])

    section("How it works")
    cards = "".join(f'<div class="sb-stepcard"><em>{number}</em><b>{title}</b><span>{text}</span></div>'
                    for number, title, text in STEPS)
    st.markdown(f'<div class="sb-steps">{cards}</div>', unsafe_allow_html=True)
