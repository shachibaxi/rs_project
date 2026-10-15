"""Streamlit application tests using Streamlit's headless AppTest runner."""

from __future__ import annotations

from dataclasses import asdict

import pytest
from streamlit.testing.v1 import AppTest

from src.profile import LearnerProfile
from tests.conftest import PROJECT_ROOT
from ui.views.explore import filter_courses

TIMEOUT = 120
PAGE_SCRIPT = """
import sys
sys.path.insert(0, {root!r})
from ui.views import {module}
{module}.render()
"""


def run_page(module: str, profile: LearnerProfile | None = None) -> AppTest:
    app = AppTest.from_string(PAGE_SCRIPT.format(root=str(PROJECT_ROOT), module=module), default_timeout=TIMEOUT)
    if profile is not None:
        app.session_state["learner_profile"] = asdict(profile)
    return app.run()


PROFILE = LearnerProfile(career_goal="Data Scientist", current_skills=["Python Programming"],
                         interests="machine learning")


def test_full_app_starts_and_shows_overview():
    app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=TIMEOUT).run()
    assert not app.exception
    assert any("Know what to learn next" in m.value for m in app.markdown)


@pytest.mark.parametrize("module", ["home", "profile_page", "recommendations", "explore", "skill_gap_page",
                                    "learning_path_page"])
def test_every_page_renders_without_errors(module):
    app = run_page(module, PROFILE)
    assert not app.exception, app.exception


def test_recommendations_page_shows_cards_and_links():
    app = run_page("recommendations", PROFILE)
    grid = next(m.value for m in app.markdown if "sb-grid" in m.value)
    assert grid.count('class="sb-card"') == 12
    assert grid.count("https://www.coursera.org/learn/") >= 10


def test_switching_method_reruns(engine):
    app = run_page("recommendations", PROFILE)
    before = [m.value for m in app.markdown if "sb-grid" in m.value]
    app.session_state["rec_view"] = "Fill my skill gaps"
    app.run()
    assert not app.exception
    after = [m.value for m in app.markdown if "sb-grid" in m.value]
    assert after and after != before                     # a different method gives a different list


def test_profile_form_saves_profile():
    app = run_page("profile_page")
    app.selectbox[0].select("Web Developer")
    app.button[-1].click()            # last button is the form submit
    app.run()
    assert not app.exception
    assert app.session_state["learner_profile"]["career_goal"] == "Web Developer"


def test_empty_profile_prompts_user():
    app = run_page("skill_gap_page", LearnerProfile())
    assert any("career goal" in i.value for i in app.info)


def test_explore_search_and_filters(engine):
    courses = engine.courses
    hits = filter_courses(courses, engine.skill_matrix, "tensorflow", [], [], [], [], 1.0, True)
    assert len(hits) > 10
    assert hits.apply(lambda r: "tensorflow" in (r["title"] + " ".join(r["skills"]) + r["description"]).lower(),
                      axis=1).all()
    narrowed = filter_courses(courses, engine.skill_matrix, "tensorflow", ["Beginner"], [], [], ["Deep Learning"],
                              4.5, False)
    assert (narrowed["difficulty"] == "Beginner").all() and (narrowed["rating"] >= 4.5).all()
    assert len(narrowed) < len(hits)
