"""The learner profile: every user input that can influence recommendations."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.knowledge_base import SkillKnowledge

PROFICIENCY_LEVELS = ["Beginner", "Intermediate", "Advanced"]
PROFICIENCY_RANK = {level: rank for rank, level in enumerate(PROFICIENCY_LEVELS, start=1)}


@dataclass
class LearnerProfile:
    """Learner context collected on the Profile page.

    Fields marked *strict_* decide whether the matching preference is a HARD
    constraint (courses that fail it are removed) or a SOFT preference (it
    only changes the ranking). Excluded topics and completed courses are
    always hard: the learner has explicitly asked not to see them.
    """

    career_goal: str | None = None
    current_skills: list[str] = field(default_factory=list)
    desired_skills: list[str] = field(default_factory=list)
    proficiency: str = "Beginner"
    interests: str = ""                                    # free-text topics of interest
    preferred_difficulties: list[str] = field(default_factory=list)
    preferred_providers: list[str] = field(default_factory=list)
    min_rating: float | None = None
    excluded_topics: list[str] = field(default_factory=list)
    must_cover_skills: list[str] = field(default_factory=list)
    completed_course_ids: list[str] = field(default_factory=list)

    strict_difficulty: bool = False
    strict_provider: bool = False
    strict_min_rating: bool = False
    strict_must_cover: bool = True

    def validate(self, knowledge: SkillKnowledge) -> list[str]:
        """Normalise the profile in place and return human-readable warnings.

        Unknown values are removed (with a warning) instead of raising, so a
        partially filled profile still produces recommendations.
        """
        warnings: list[str] = []
        if self.career_goal and self.career_goal not in knowledge.careers:
            warnings.append(f"Unknown career goal '{self.career_goal}' was ignored.")
            self.career_goal = None
        if self.proficiency not in PROFICIENCY_RANK:
            warnings.append(f"Unknown proficiency '{self.proficiency}'; using Beginner.")
            self.proficiency = "Beginner"

        for attribute in ("current_skills", "desired_skills", "must_cover_skills"):
            values = getattr(self, attribute)
            unknown = [skill for skill in values if skill not in knowledge.skills]
            if unknown:
                warnings.append(f"Ignored unknown skills in {attribute.replace('_', ' ')}: {', '.join(unknown)}.")
            setattr(self, attribute, list(dict.fromkeys(s for s in values if s in knowledge.skills)))

        valid_difficulties = [d for d in self.preferred_difficulties if d in PROFICIENCY_RANK]
        if len(valid_difficulties) != len(self.preferred_difficulties):
            warnings.append("Ignored unknown difficulty values.")
        self.preferred_difficulties = list(dict.fromkeys(valid_difficulties))

        if self.min_rating is not None and not (1.0 <= float(self.min_rating) <= 5.0):
            warnings.append("Minimum rating must be between 1 and 5; it was ignored.")
            self.min_rating = None

        self.excluded_topics = [t.strip().lower() for t in self.excluded_topics if t and t.strip()]
        self.completed_course_ids = list(dict.fromkeys(self.completed_course_ids))
        return warnings

    @property
    def known_skills(self) -> set[str]:
        return set(self.current_skills)

    @property
    def proficiency_rank(self) -> int:
        return PROFICIENCY_RANK[self.proficiency]
