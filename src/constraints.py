"""Constraint-based filtering: hard constraints and soft preferences.

HARD constraints remove courses. Every removal is counted per constraint so
the interface can explain an empty or short result and let the learner relax
the constraint that removed the most courses.

SOFT preferences never remove courses; they produce a [0, 1] preference score
that the knowledge-based and hybrid rankers add to their scoring.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.knowledge_base import ADDRESSES_THRESHOLD
from src.profile import PROFICIENCY_RANK, LearnerProfile


@dataclass
class ConstraintReport:
    """Outcome of applying hard constraints."""

    eligible: pd.Series                                     # bool, indexed by course_id
    removed_by: dict[str, int] = field(default_factory=dict)  # constraint -> courses it rejects
    active: list[str] = field(default_factory=list)

    @property
    def eligible_ids(self) -> set[str]:
        return set(self.eligible.index[self.eligible])

    def most_restrictive(self) -> str | None:
        relaxable = {k: v for k, v in self.removed_by.items() if k != "Already completed"}
        return max(relaxable, key=relaxable.get) if relaxable else None


def _topic_pattern(topic: str) -> re.Pattern:
    return re.compile(rf"(?<![a-z0-9]){re.escape(topic.lower())}(?![a-z0-9])")


def apply_hard_constraints(
    courses: pd.DataFrame,
    profile: LearnerProfile,
    skill_matrix: pd.DataFrame,
    rated_ids: set[str],
) -> ConstraintReport:
    """Evaluate every hard constraint independently and combine them with AND.

    ``removed_by`` counts how many courses each constraint rejects on its own
    (a course can be rejected by several), which is what a learner needs to
    know to decide which constraint to relax.
    """
    index = courses["course_id"].values
    checks: dict[str, np.ndarray] = {}

    if profile.completed_course_ids:
        checks["Already completed"] = ~courses["course_id"].isin(profile.completed_course_ids).values

    if profile.excluded_topics:
        searchable = (
            courses["title"].str.lower() + " | "
            + courses["skills"].map(" | ".join) + " | "
            + courses["domain"].str.replace("-", " ") + " | " + courses["subdomain"].str.replace("-", " ")
        )
        excluded = np.zeros(len(courses), dtype=bool)
        for topic in profile.excluded_topics:
            excluded |= searchable.str.contains(_topic_pattern(topic)).values
        checks["Excluded topics"] = ~excluded

    if profile.strict_difficulty and profile.preferred_difficulties:
        # "Conversant" is ranked as Intermediate, so compare ranks, not labels.
        allowed_ranks = {PROFICIENCY_RANK[d] for d in profile.preferred_difficulties}
        checks["Difficulty"] = courses["difficulty_rank"].isin(allowed_ranks).values

    if profile.strict_provider and profile.preferred_providers:
        checks["Provider"] = courses["provider"].isin(profile.preferred_providers).values

    if profile.strict_min_rating and profile.min_rating is not None:
        # Only genuinely rated courses can be shown to satisfy a rating threshold.
        is_rated = courses["course_id"].isin(rated_ids).values
        checks["Minimum rating"] = is_rated & (courses["rating"].fillna(-1).values >= profile.min_rating)

    if profile.strict_must_cover and profile.must_cover_skills:
        covered = (skill_matrix.loc[index, profile.must_cover_skills] >= ADDRESSES_THRESHOLD).any(axis=1)
        checks["Must-cover skills"] = covered.values

    eligible = np.ones(len(courses), dtype=bool)
    for passed in checks.values():
        eligible &= passed
    return ConstraintReport(
        eligible=pd.Series(eligible, index=index),
        removed_by={name: int((~passed).sum()) for name, passed in checks.items()},
        active=list(checks),
    )


# Soft-preference weights inside the preference score. Difficulty matters most
# because a course at the wrong level is the most common reason it is unusable.
SOFT_WEIGHTS = {"difficulty": 0.5, "provider": 0.2, "rating": 0.3}


def soft_preference_score(courses: pd.DataFrame, profile: LearnerProfile) -> pd.Series:
    """Score in [0, 1] for how well each course fits the NON-strict preferences.

    Preferences the learner did not set (or set as strict, where they already
    filtered) contribute nothing; the score is re-normalised over the
    preferences that are active.
    """
    parts: dict[str, pd.Series] = {}
    index = courses["course_id"].values

    if profile.preferred_difficulties and not profile.strict_difficulty:
        ranks = [PROFICIENCY_RANK[d] for d in profile.preferred_difficulties]
        distance = courses["difficulty_rank"].map(
            lambda r: np.nan if pd.isna(r) else min(abs(r - x) for x in ranks)
        )
        # 0 steps away -> 1, 1 step -> 0.5, 2 steps -> 0; unknown level -> neutral 0.5
        parts["difficulty"] = (1 - distance / 2).fillna(0.5)

    if profile.preferred_providers and not profile.strict_provider:
        parts["provider"] = courses["provider"].isin(profile.preferred_providers).astype(float)

    if profile.min_rating is not None and not profile.strict_min_rating:
        # Unrated courses get a neutral 0.5: no evidence either way.
        meets = courses["rating"] >= profile.min_rating
        parts["rating"] = np.where(courses["rating"].isna(), 0.5, meets.astype(float))

    if not parts:
        return pd.Series(np.nan, index=index)
    total_weight = sum(SOFT_WEIGHTS[name] for name in parts)
    combined = sum(SOFT_WEIGHTS[name] * np.asarray(values, dtype=float) for name, values in parts.items())
    return pd.Series(combined / total_weight, index=index)
