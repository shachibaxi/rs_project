"""Session state and cached resources for the Streamlit app.

The engine (data + fitted models) is identical for every user, so it is cached
once with ``st.cache_resource``. Its cache key is a fingerprint of the files it
is built from, so editing the dataset or the knowledge config invalidates it.

The learner profile is per browser session (``st.session_state``) and is never
put in a shared cache, so different users' profiles cannot mix.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, fields

import streamlit as st

from src.data_loader import (KNOWLEDGE_CONFIG_PATH, PROCESSED_CATALOGUE_PATH, RATED_CATALOGUE_PATH,
                             SYNTHETIC_INTERACTIONS_PATH, file_fingerprint, find_raw_csv)
from src.profile import LearnerProfile
from src.recommender import SkillBridgeEngine

PROFILE_KEY = "learner_profile"


def data_fingerprint() -> str:
    paths = [find_raw_csv(), KNOWLEDGE_CONFIG_PATH, PROCESSED_CATALOGUE_PATH,
             RATED_CATALOGUE_PATH, SYNTHETIC_INTERACTIONS_PATH]
    return "|".join(file_fingerprint(p) if p.exists() else "missing" for p in paths)


@st.cache_resource(show_spinner="Loading the catalogue and fitting the models…")
def _load_engine(fingerprint: str) -> SkillBridgeEngine:  # fingerprint is the cache key
    return SkillBridgeEngine.load()


def get_engine() -> SkillBridgeEngine:
    return _load_engine(data_fingerprint())


# ----------------------------------------------------------------- profile
def get_profile() -> LearnerProfile:
    """A fresh copy of the session's profile (callers may modify it freely)."""
    stored = st.session_state.get(PROFILE_KEY)
    if stored is None:
        return LearnerProfile()
    valid = {f.name for f in fields(LearnerProfile)}
    return LearnerProfile(**{k: deepcopy(v) for k, v in stored.items() if k in valid})


def save_profile(profile: LearnerProfile) -> None:
    st.session_state[PROFILE_KEY] = asdict(profile)


def profile_is_empty(profile: LearnerProfile) -> bool:
    return not (profile.career_goal or profile.desired_skills or profile.interests.strip()
                or profile.completed_course_ids)


EXAMPLE_PROFILES = {
    "Beginner → Data Scientist": LearnerProfile(
        career_goal="Data Scientist", current_skills=["Python Programming"], proficiency="Beginner",
        interests="data analysis with python, statistics, machine learning projects",
        preferred_difficulties=["Beginner"], min_rating=4.5,
    ),
    "Intermediate → ML Engineer": LearnerProfile(
        career_goal="Machine Learning Engineer",
        current_skills=["Python Programming", "Statistics", "Machine Learning"],
        proficiency="Intermediate", interests="deploying models, tensorflow, cloud",
        desired_skills=["MLOps & Model Deployment"],
    ),
    "Career switcher → Web Developer": LearnerProfile(
        career_goal="Web Developer", current_skills=[], proficiency="Beginner",
        interests="build websites, front-end", excluded_topics=["php"],
        completed_course_ids=["style-html-tables-css"],
    ),
}
