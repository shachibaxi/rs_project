"""Skill-gap analysis.

Compares the learner's current skills with the (manually defined) skills of
their career goal plus any desired skills, then asks the shared hybrid
recommender for the best courses addressing each missing skill.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.knowledge_base import ADDRESSES_THRESHOLD
from src.knowledge_based import DESIRED_SKILL_PRIORITY, compute_targets
from src.learning_path import starting_rank
from src.profile import PROFICIENCY_LEVELS, LearnerProfile
from src.recommender import SkillBridgeEngine


@dataclass
class GapDetail:
    skill: str
    priority: float                         # 1.0 desired, 0.8 career requirement
    source: str                             # "Desired skill" or "Career requirement"
    prerequisite_depth: int                 # 0 = foundational
    missing_prerequisites: list[list[str]]  # groups the learner does not satisfy yet
    courses_in_catalogue: int               # courses directly addressing the skill
    recommendations: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class SkillGapReport:
    career_goal: str | None
    target_skills: list[str]
    possessed: list[str]
    gaps: list[GapDetail]
    extra_skills: list[str]                 # learner skills outside the target set
    notes: list[str] = field(default_factory=list)

    @property
    def readiness_percent(self) -> float:
        """Share of target skills the learner already has (0-100)."""
        if not self.target_skills:
            return 0.0
        return 100.0 * len(self.possessed) / len(self.target_skills)


def analyse_skill_gap(engine: SkillBridgeEngine, profile: LearnerProfile, courses_per_gap: int = 3) -> SkillGapReport:
    notes = profile.validate(engine.knowledge)
    knowledge = engine.knowledge
    targets = compute_targets(profile, knowledge)
    if not targets.target_skills:
        notes.append("Choose a career goal or desired skills to run the analysis.")

    supply = (engine.skill_matrix >= ADDRESSES_THRESHOLD).sum()
    gaps: list[GapDetail] = []
    for skill, priority in targets.gaps.items():
        # Same focused request as the learning path: the best courses for this one skill.
        level = PROFICIENCY_LEVELS[starting_rank(skill, profile, knowledge) - 1]
        result = engine.recommend_for_skill(profile, skill, courses_per_gap, level=level)
        gaps.append(GapDetail(
            skill=skill,
            priority=priority,
            source="Desired skill" if priority == DESIRED_SKILL_PRIORITY else "Career requirement",
            prerequisite_depth=knowledge.prerequisite_depth(skill),
            missing_prerequisites=knowledge.missing_prerequisites(skill, profile.known_skills),
            courses_in_catalogue=int(supply[skill]),
            recommendations=result.table,
        ))
    # Foundational, ready-to-start skills first; then by priority.
    gaps.sort(key=lambda g: (len(g.missing_prerequisites) > 0, g.prerequisite_depth, -g.priority, g.skill))
    return SkillGapReport(
        career_goal=profile.career_goal,
        target_skills=targets.target_skills,
        possessed=targets.possessed,
        gaps=gaps,
        extra_skills=[s for s in profile.current_skills if s not in targets.target_skills],
        notes=notes,
    )
