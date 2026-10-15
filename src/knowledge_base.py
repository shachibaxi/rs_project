"""Manually defined skill knowledge and its link to the course catalogue.

``config/skill_knowledge.json`` holds hand-written knowledge (skill keywords,
prerequisites, career requirements). This module loads and validates it and
computes how strongly each course is associated with each skill by searching
the course's dataset fields (skill tags, title, description) for the skill's
keywords. The *matching* uses dataset text; the *knowledge* is manual.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import KNOWLEDGE_CONFIG_PATH

# Evidence strength when a skill keyword is found in a course field.
# A course's own skill tags or title are direct evidence that it teaches the
# skill; a mention somewhere in a long description is weaker evidence.
# (Weighting titles above the noisy auto-generated tags was tried and rejected:
# it lowered held-out graded NDCG in the cross-validated evaluation.)
TAG_OR_TITLE_STRENGTH = 1.0
DESCRIPTION_STRENGTH = 0.4
# A course is said to "address" a skill only on direct (tag/title) evidence.
ADDRESSES_THRESHOLD = TAG_OR_TITLE_STRENGTH


class KnowledgeConfigError(Exception):
    """Raised when skill_knowledge.json is inconsistent."""


@dataclass(frozen=True)
class SkillKnowledge:
    """Parsed contents of ``skill_knowledge.json``."""

    skills: dict[str, dict]      # skill name -> {"keywords": [...], "prerequisites": [[...], ...]}
    careers: dict[str, dict]     # career name -> {"description", "required_skills", "related_subdomains"}

    @property
    def skill_names(self) -> list[str]:
        return list(self.skills)

    def prerequisite_groups(self, skill: str) -> list[list[str]]:
        return self.skills.get(skill, {}).get("prerequisites", [])

    def career_skills(self, career: str) -> list[str]:
        if career not in self.careers:
            raise KeyError(f"Unknown career goal: {career!r}")
        return list(self.careers[career]["required_skills"])

    def missing_prerequisites(self, skill: str, known_skills: set[str]) -> list[list[str]]:
        """Prerequisite groups of ``skill`` that the learner does not yet satisfy."""
        return [group for group in self.prerequisite_groups(skill) if not set(group) & known_skills]

    def prerequisite_depth(self, skill: str, _visiting: frozenset = frozenset()) -> int:
        """Longest chain of prerequisites below ``skill`` (0 = foundational)."""
        if skill in _visiting:
            raise KnowledgeConfigError(f"Prerequisite cycle involving {skill!r}")
        groups = self.prerequisite_groups(skill)
        if not groups:
            return 0
        visiting = _visiting | {skill}
        # For an "any of" group the easiest alternative defines the depth.
        return 1 + max(min(self.prerequisite_depth(s, visiting) for s in group) for group in groups)


def load_knowledge(path: Path = KNOWLEDGE_CONFIG_PATH) -> SkillKnowledge:
    """Load and validate the knowledge file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    skills, careers = data.get("skills", {}), data.get("careers", {})
    if not skills or not careers:
        raise KnowledgeConfigError("skill_knowledge.json must define 'skills' and 'careers'.")

    for name, spec in skills.items():
        if not spec.get("keywords"):
            raise KnowledgeConfigError(f"Skill {name!r} has no keywords.")
        for group in spec.get("prerequisites", []):
            unknown = [s for s in group if s not in skills]
            if unknown:
                raise KnowledgeConfigError(f"Skill {name!r} has unknown prerequisites {unknown}.")
    for name, spec in careers.items():
        unknown = [s for s in spec.get("required_skills", []) if s not in skills]
        if unknown:
            raise KnowledgeConfigError(f"Career {name!r} lists unknown skills {unknown}.")

    knowledge = SkillKnowledge(skills=skills, careers=careers)
    for skill in skills:  # raises on cycles
        knowledge.prerequisite_depth(skill)
    return knowledge


def _keyword_pattern(keywords: list[str]) -> re.Pattern:
    """Regex matching any keyword as a whole word/phrase (case-insensitive)."""
    alternatives = "|".join(re.escape(k.lower()) for k in sorted(keywords, key=len, reverse=True))
    return re.compile(rf"(?<![a-z0-9])(?:{alternatives})(?![a-z0-9])")


def compute_skill_matrix(courses: pd.DataFrame, knowledge: SkillKnowledge) -> pd.DataFrame:
    """Return a (courses x skills) table of evidence strengths in [0, 1].

    Strength is ``TAG_OR_TITLE_STRENGTH`` if a keyword appears in the course's
    skill tags or title, ``DESCRIPTION_STRENGTH`` if it only appears in the
    description, and 0 otherwise.
    """
    tags_text = courses["skills"].map(lambda tags: " | ".join(tags)).str.lower()
    title_text = courses["title"].fillna("").str.lower()
    description_text = courses["description"].fillna("").str.lower()

    matrix = {}
    for skill, spec in knowledge.skills.items():
        pattern = _keyword_pattern(spec["keywords"])
        direct = tags_text.str.contains(pattern) | title_text.str.contains(pattern)
        mentioned = description_text.str.contains(pattern)
        matrix[skill] = np.where(direct, TAG_OR_TITLE_STRENGTH, np.where(mentioned, DESCRIPTION_STRENGTH, 0.0))
    return pd.DataFrame(matrix, index=courses["course_id"].values)


def skills_addressed(skill_row: pd.Series) -> list[str]:
    """Skills a course directly addresses (tag/title evidence)."""
    return [skill for skill, strength in skill_row.items() if strength >= ADDRESSES_THRESHOLD]
