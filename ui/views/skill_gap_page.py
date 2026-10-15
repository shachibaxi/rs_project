"""Skill-gap analysis page."""

from __future__ import annotations

from html import escape

import streamlit as st

from src.skill_gap import analyse_skill_gap
from ui import navigation
from ui.components import chips, course_rows_html, journey, page_header, readiness_ring, section, technique
from ui.state import get_engine, get_profile

COURSES_PER_GAP = 2


def _gap_card(gap) -> str:
    if gap.missing_prerequisites:
        first = " or ".join(gap.missing_prerequisites[0])
        status = f'<span class="sb-pill sb-status-wait">Learn {escape(first)} first</span>'
    else:
        status = '<span class="sb-pill sb-status-ready">Ready to start</span>'
    if gap.recommendations.empty:
        courses = '<div class="sb-meta" style="padding:0.5rem 0">No course matches your filters for this skill.</div>'
    else:
        courses = course_rows_html(gap.recommendations)
    return (f'<div class="sb-gap"><div class="sb-gap-head"><span class="sb-gap-name">{escape(gap.skill)}</span>'
            f'{status}</div>{courses}</div>')


def render() -> None:
    engine = get_engine()
    profile = get_profile()
    goal = profile.career_goal or "your goal"
    page_header("Step 2", f"What {goal} needs",
                "Your skills compared with the skills this goal requires, with the best courses for each gap.")
    journey("skill-gap")

    if not (profile.career_goal or profile.desired_skills):
        st.info("Choose a career goal or skills you want to learn to see your gaps.")
        navigation.link("profile", "Set up your profile", ":material/person:")
        return

    report = analyse_skill_gap(engine, profile, courses_per_gap=COURSES_PER_GAP)
    technique("Knowledge-based", "Your skills are compared with what this career needs; prerequisites decide "
                                 "which gaps you can start now.")
    left, right = st.columns([1, 1.6], gap="medium")
    with left:
        st.markdown(readiness_ring(report.readiness_percent, len(report.possessed), len(report.target_skills)),
                    unsafe_allow_html=True)
    with right:
        have = chips(report.possessed, "accent") or '<span class="sb-meta">None yet</span>'
        missing = chips([g.skill for g in report.gaps], "warm") or '<span class="sb-meta">Nothing — well done</span>'
        st.markdown(f'<div class="sb-label">You have</div><div class="sb-chips">{have}</div>'
                    f'<div class="sb-label" style="margin-top:1rem">To learn</div><div class="sb-chips">{missing}</div>',
                    unsafe_allow_html=True)

    if not report.gaps:
        st.success("You already have every skill this goal needs. Add skills you want to learn to go further.")
        return

    section("Skills to learn")
    columns = st.columns(2, gap="medium")
    for index, gap in enumerate(report.gaps):
        with columns[index % 2]:
            st.markdown(_gap_card(gap), unsafe_allow_html=True)
    navigation.next_step("recommendations", "Next: courses for you")
