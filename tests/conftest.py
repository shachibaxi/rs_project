"""Shared fixtures.

``engine`` loads the REAL catalogue once per test session. ``tiny_courses``
is a hand-made cleaned catalogue for edge cases (missing fields, few courses).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.knowledge_base import SkillKnowledge  # noqa: E402
from src.recommender import SkillBridgeEngine  # noqa: E402


@pytest.fixture(scope="session")
def engine() -> SkillBridgeEngine:
    return SkillBridgeEngine.load()


@pytest.fixture
def tiny_knowledge() -> SkillKnowledge:
    return SkillKnowledge(
        skills={
            "Python Programming": {"keywords": ["python"], "prerequisites": []},
            "Statistics": {"keywords": ["statistics"], "prerequisites": []},
            "Machine Learning": {"keywords": ["machine learning"],
                                 "prerequisites": [["Python Programming"], ["Statistics"]]},
            "Quantum Basket Weaving": {"keywords": ["quantum basket weaving"], "prerequisites": []},
        },
        careers={"ML Person": {"description": "", "required_skills": ["Python Programming", "Statistics",
                                                                      "Machine Learning", "Quantum Basket Weaving"],
                               "related_subdomains": ["machine-learning"]}},
    )


@pytest.fixture
def tiny_courses() -> pd.DataFrame:
    """Cleaned-format catalogue with deliberately missing optional metadata."""
    rows = [
        ("py-intro", "Python for Everybody", "Uni A", "Beginner", 4.8, "Learn python basics.", ["python"], 1.0),
        ("stats-1", "Intro Statistics", "Uni B", "Intermediate", np.nan, "", [], 2.0),
        ("ml-adv", "Advanced Machine Learning", "Uni A", "Advanced", 4.2, "machine learning with python",
         ["machine learning"], 3.0),
        ("mystery", "Untitled Topic", "Unknown provider", "Not Calibrated", np.nan, "", [], np.nan),
    ]
    frame = pd.DataFrame(rows, columns=["course_id", "title", "provider", "difficulty", "rating", "description",
                                        "skills", "difficulty_rank"])
    frame["url"] = "https://www.coursera.org/learn/" + frame["course_id"]
    frame["domain"] = ["data-science", "data-science", "data-science", ""]
    frame["subdomain"] = ["machine-learning", "probability-and-statistics", "machine-learning", ""]
    frame["has_rating"] = frame["rating"].notna()
    return frame


@pytest.fixture
def tiny_engine(tiny_courses, tiny_knowledge) -> SkillBridgeEngine:
    rated = tiny_courses.loc[tiny_courses["rating"].notna(), ["course_id", "rating"]]
    return SkillBridgeEngine(tiny_courses, rated, tiny_knowledge, interactions=None)
