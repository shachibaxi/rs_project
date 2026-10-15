"""Learner profile page: every input that personalises the recommendations."""

from __future__ import annotations

import streamlit as st

from src.profile import PROFICIENCY_LEVELS, LearnerProfile
from ui import navigation
from ui.components import journey, page_header
from ui.state import EXAMPLE_PROFILES, get_engine, get_profile, profile_is_empty, save_profile

VERSION_KEY = "profile_form_version"   # bumped to reset widgets after loading an example
NO_GOAL = "Choose a goal…"


def _load(profile: LearnerProfile) -> None:
    save_profile(profile)
    st.session_state[VERSION_KEY] = st.session_state.get(VERSION_KEY, 0) + 1


def _on_example() -> None:
    choice = st.session_state.get("example_choice")
    if choice:
        _load(EXAMPLE_PROFILES[choice])
        st.session_state["example_choice"] = None


def render() -> None:
    engine = get_engine()
    knowledge = engine.knowledge
    profile = get_profile()
    version = st.session_state.get(VERSION_KEY, 0)

    page_header("Step 1", "Tell us about yourself",
                "Only a goal or a few interests are needed. Everything else is optional.")
    journey("profile")
    st.pills("Or start from an example", list(EXAMPLE_PROFILES), key="example_choice", on_change=_on_example)

    skill_names = knowledge.skill_names
    careers = [NO_GOAL] + list(knowledge.careers)
    providers = sorted(engine.courses["provider"].unique())
    course_labels = dict(zip(engine.courses["course_id"],
                             engine.courses["title"] + " — " + engine.courses["provider"]))

    with st.form(f"profile_form_{version}", border=False):
        with st.container(border=True):
            left, right = st.columns([3, 2])
            career = left.selectbox("Career goal", careers,
                                    index=careers.index(profile.career_goal) if profile.career_goal in careers else 0)
            proficiency = right.segmented_control("Your level", PROFICIENCY_LEVELS, default=profile.proficiency,
                                                  required=True, width="stretch")
            current = st.multiselect("Skills you already have", skill_names, default=profile.current_skills,
                                     placeholder="e.g. Python Programming")
            desired = st.multiselect("Skills you want to learn", skill_names, default=profile.desired_skills,
                                     placeholder="Optional — your goal already implies some")
            interests = st.text_input("Interests", value=profile.interests,
                                      placeholder="e.g. computer vision for healthcare, startup finance")

        with st.expander("Filters", icon=":material/filter_list:"):
            st.caption("**Constraint-based filtering.** Turn on *Only these* to hide everything else; "
                       "otherwise it is a preference that only changes the order.")
            c1, c2 = st.columns([4, 1], vertical_alignment="bottom")
            difficulties = c1.multiselect("Difficulty", PROFICIENCY_LEVELS, default=profile.preferred_difficulties)
            strict_difficulty = c2.toggle("Only these", value=profile.strict_difficulty, key=f"sd_{version}")

            c1, c2 = st.columns([4, 1], vertical_alignment="bottom")
            chosen_providers = c1.multiselect("Universities / companies", providers, default=profile.preferred_providers)
            strict_provider = c2.toggle("Only these", value=profile.strict_provider, key=f"sp_{version}")

            c1, c2 = st.columns([4, 1], vertical_alignment="bottom")
            rating_options = ["Any", 4.0, 4.3, 4.5, 4.7]
            min_rating = c1.selectbox("Minimum rating", rating_options,
                                      index=rating_options.index(profile.min_rating) if profile.min_rating in rating_options else 0,
                                      format_func=lambda v: "Any rating" if v == "Any" else f"★ {v} or higher")
            strict_rating = c2.toggle("Only these", value=profile.strict_min_rating, key=f"sr_{version}")

            c1, c2 = st.columns([4, 1], vertical_alignment="bottom")
            must_cover = c1.multiselect("Must teach one of", skill_names, default=profile.must_cover_skills)
            strict_cover = c2.toggle("Only these", value=profile.strict_must_cover, key=f"sm_{version}")

            excluded = st.text_input("Hide topics", value=", ".join(profile.excluded_topics),
                                     placeholder="e.g. php, blockchain")

        with st.expander("Courses you've completed", icon=":material/task_alt:"):
            completed = st.multiselect(
                "Completed courses", list(course_labels), default=profile.completed_course_ids,
                format_func=lambda cid: course_labels.get(cid, cid), label_visibility="collapsed",
                placeholder="Search courses you've already taken",
            )
            st.caption("Courses you've completed won't be recommended again.")

        submitted = st.form_submit_button("Save and continue", type="primary", width="stretch",
                                          icon=":material/arrow_forward:", icon_position="right")

    if submitted:
        new_profile = LearnerProfile(
            career_goal=None if career == NO_GOAL else career,
            current_skills=current, desired_skills=desired, proficiency=proficiency or "Beginner",
            interests=interests.strip(), preferred_difficulties=difficulties,
            preferred_providers=chosen_providers, min_rating=None if min_rating == "Any" else float(min_rating),
            excluded_topics=[t.strip() for t in excluded.split(",") if t.strip()],
            must_cover_skills=must_cover, completed_course_ids=completed,
            strict_difficulty=strict_difficulty, strict_provider=strict_provider,
            strict_min_rating=strict_rating, strict_must_cover=strict_cover,
        )
        warnings = new_profile.validate(knowledge)
        save_profile(new_profile)
        if profile_is_empty(new_profile):
            st.info("Saved. Add a career goal, a skill you want or an interest to get recommendations.")
        elif warnings:
            for warning in warnings:
                st.warning(warning)
        else:
            navigation.go("skill-gap")          # step 1 done -> step 2
