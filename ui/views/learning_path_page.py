"""Learning-path page: an ordered sequence of courses."""

from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st

from src.learning_path import generate_learning_path, in_path_dependencies
from ui import navigation
from ui.components import course_meta, journey, page_header, technique
from ui.state import get_engine, get_profile

STAGE_NAMES = {1: "Foundations", 2: "Build on it", 3: "Go deeper", 4: "Specialise"}


def _step_html(step, builds_on: list[str]) -> str:
    course = step.course
    if course:
        course_block = (
            f'<a class="sb-row-title" href="{escape(course["url"])}" target="_blank" rel="noopener">'
            f'{escape(course["title"])}</a><div class="sb-meta">{course_meta(pd.Series(course))}</div>'
        )
    else:
        course_block = '<div class="sb-meta">No course in the catalogue teaches this under your filters.</div>'
    builds = f'<div class="sb-step-why">Builds on {escape(", ".join(builds_on))}</div>' if builds_on else ""
    return (
        f'<div class="sb-step"><div class="sb-step-num">{step.position}</div><div class="sb-step-body">'
        f'<div class="sb-step-skill">{escape(step.skill)}'
        f'<span class="sb-pill sb-pill-level">{escape(step.target_difficulty)}</span></div>'
        f'<div style="margin-top:0.5rem">{course_block}</div>{builds}</div></div>'
    )


def render() -> None:
    engine = get_engine()
    profile = get_profile()
    goal = profile.career_goal or "your goal"
    page_header("Step 4", f"Your path to {goal}",
                "Missing skills in the order you should learn them, with one course for each step.")
    journey("learning-path")
    if not (profile.career_goal or profile.desired_skills):
        st.info("Choose a career goal or skills you want to learn to build a path.")
        navigation.link("profile", "Set up your profile", ":material/person:")
        return

    path = generate_learning_path(engine, profile)
    if not path.steps:
        st.success("You already have every skill this goal needs.")
        return

    technique("Prerequisite graph + hybrid", "Skills are ordered so each comes after what it builds on "
                                              "(topological sort); the hybrid picks one course per step.")
    stages = sorted({s.stage for s in path.steps})
    path_skills = {s.skill for s in path.steps}
    summary, download = st.columns([3, 1], vertical_alignment="center")
    summary.markdown(f"**{len(path.steps)} steps** in **{len(stages)} stages**")
    export = pd.DataFrame([{
        "step": s.position, "stage": s.stage, "skill": s.skill, "target_level": s.target_difficulty,
        "course": s.course["title"] if s.course else "", "provider": s.course["provider"] if s.course else "",
        "difficulty": s.course["difficulty"] if s.course else "", "url": s.course["url"] if s.course else "",
        "why_in_path": s.reason_in_path, "why_here": s.reason_for_position,
    } for s in path.steps])
    download.download_button("Download CSV", export.to_csv(index=False), file_name="skillbridge_learning_path.csv",
                             mime="text/csv", icon=":material/download:", width="stretch")

    for stage in stages:
        st.markdown(f'<div class="sb-stage"><b>Stage {stage} · {STAGE_NAMES.get(stage, "Advance")}</b><span></span></div>',
                    unsafe_allow_html=True)
        steps = [s for s in path.steps if s.stage == stage]
        html = "".join(_step_html(s, in_path_dependencies(s.skill, path_skills, engine.knowledge)) for s in steps)
        st.markdown(f'<div class="sb-timeline">{html}</div>', unsafe_allow_html=True)
