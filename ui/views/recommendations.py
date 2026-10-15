"""Personalised recommendations and similar-course search."""

from __future__ import annotations

import streamlit as st

from src.knowledge_based import compute_targets
from src.recommender import CONTENT, HYBRID, KNOWLEDGE
from ui import navigation
from ui.components import course_grid, featured_card, journey, page_header, snippet, summary_strip, technique
from ui.state import get_engine, get_profile, profile_is_empty, save_profile

# The website offers the three personalised rankings built on real catalogue data.
# Collaborative filtering (synthetic learners) and the top-rated baseline are
# compared in the evaluation notebook instead.
VIEWS = {
    "Best match": HYBRID,
    "Fill my skill gaps": KNOWLEDGE,
    "Match my interests": CONTENT,
}

# Which profile field to change when the learner loosens a hard filter, and the button text.
RELAX_ACTIONS = {
    "Difficulty": ("strict_difficulty", False, "Allow other difficulty levels"),
    "Provider": ("strict_provider", False, "Allow other providers"),
    "Minimum rating": ("strict_min_rating", False, "Allow lower-rated courses"),
    "Must-cover skills": ("strict_must_cover", False, "Don't require specific skills"),
    "Excluded topics": ("excluded_topics", [], "Stop hiding topics"),
}
RESULTS = 13          # one featured course + a grid of twelve

# The technique behind each view, in plain language (shown as a small tag).
TECHNIQUES = {
    "Best match": ("Hybrid", "Mostly your skill gaps and level, plus your interests and course ratings; "
                             "near-duplicate courses are removed."),
    "Fill my skill gaps": ("Knowledge-based", "Rules: teaches a skill you're missing, fits your level, and you have "
                                              "its prerequisites."),
    "Match my interests": ("Content-based", "TF-IDF text similarity between your goal and interests and each course."),
}


def _relax(constraint: str) -> None:
    profile = get_profile()
    field, value, _ = RELAX_ACTIONS[constraint]
    setattr(profile, field, value)
    save_profile(profile)


def _summary(engine, profile) -> None:
    targets = compute_targets(profile, engine.knowledge)
    items = [("Goal", profile.career_goal or "Your interests"), ("Level", profile.proficiency)]
    if targets.target_skills:
        items += [("Skills you have", f"{len(targets.possessed)} of {len(targets.target_skills)}"),
                  ("Skills to learn", str(len(targets.gaps)))]
    left, right = st.columns([5, 1], vertical_alignment="center")
    with left:
        summary_strip(items)
    with right:
        navigation.link("profile", "Edit profile", ":material/edit:")


def _for_you(engine, profile) -> None:
    view = st.segmented_control("View", list(VIEWS), default="Best match", required=True, key="rec_view",
                                label_visibility="collapsed")
    technique(*TECHNIQUES[view])
    result = engine.recommend(profile, VIEWS[view], RESULTS, use_collaborative=False)
    if result.table.empty:
        st.info("No courses match yet. Add a career goal, skills you want, or a few interests.")
        return

    report = result.constraint_report
    culprit = report.most_restrictive() if report else None
    hard_filters = [name for name in report.active if name != "Already completed"] if report else []
    if hard_filters:
        technique("Constraint-based", f"Your filters ({', '.join(hard_filters).lower()}) leave "
                                      f"{int(report.eligible.sum()):,} eligible courses.")
    if len(result.table) < RESULTS and culprit in RELAX_ACTIONS:
        message, action = st.columns([3, 1.3], vertical_alignment="center")
        message.warning(f"Only {len(result.table)} courses match all of your filters.")
        action.button(RELAX_ACTIONS[culprit][2], on_click=_relax, args=(culprit,), width="stretch")

    top = result.table.iloc[0]
    featured_card(top, snippet(engine.course(top["course_id"])["description"], 320))
    if len(result.table) > 1:
        course_grid(result.table.iloc[1:])


def _similar(engine) -> None:
    labels = dict(zip(engine.courses["course_id"], engine.courses["title"] + " — " + engine.courses["provider"]))
    course_id = st.selectbox("Find courses like", list(labels), format_func=labels.get, index=None,
                             placeholder="Search for a course you liked…", label_visibility="collapsed")
    technique("Content-based · nearest neighbours", "Courses whose text is most similar to the one you pick.")
    if not course_id:
        return
    table = engine.similar_courses(course_id, 9)
    descriptions = {cid: snippet(engine.course(cid)["description"], 150) for cid in table["course_id"]}
    course_grid(table, score_label="similar", descriptions=descriptions)


def render() -> None:
    engine = get_engine()
    profile = get_profile()
    page_header("Step 3", "Courses for you",
                "Courses ranked for your goal, your current skills and your level.")
    journey("recommendations")

    tab_profile, tab_similar = st.tabs(["For you", "Similar courses"])
    with tab_profile:
        if profile_is_empty(profile):
            st.info("Tell us your goal and interests to get personalised courses.")
            navigation.link("profile", "Set up your profile", ":material/person:")
        else:
            _summary(engine, profile)
            _for_you(engine, profile)
            navigation.next_step("learning-path", "Next: your learning path")
    with tab_similar:
        _similar(engine)
