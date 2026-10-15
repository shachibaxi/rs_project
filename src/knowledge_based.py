"""Knowledge-based recommendation with explicit, explainable rules.

Inputs: the learner profile, the manually defined skill knowledge
(career requirements + prerequisites) and the course-skill evidence matrix.

For every course the recommender computes four rule scores in [0, 1]:

1. Goal relevance   - does the course teach a skill the learner is missing?
2. Level fit        - is the course's difficulty close to the learner's level?
3. Readiness        - does the learner already have the prerequisites of the
                      skills the course teaches?
4. Preferences      - soft difficulty/provider/rating preferences
                      (see ``constraints.soft_preference_score``).

and combines them with the documented weights below. A course that teaches
none of the learner's target skills has relevance 0 and is not recommended
by this method.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.constraints import soft_preference_score
from src.knowledge_base import ADDRESSES_THRESHOLD, SkillKnowledge
from src.profile import LearnerProfile

# Priority of a target skill. A skill the learner explicitly asked for is
# weighted above one implied by the career template.
DESIRED_SKILL_PRIORITY = 1.0
CAREER_GAP_PRIORITY = 0.8
# Goal relevance combines the evidence for each target skill with a
# "noisy-OR": relevance = 1 - prod_s (1 - ALPHA * priority_s * evidence_s).
# One strongly taught desired skill gives ALPHA; each further skill adds
# less and less, so breadth helps without letting heavily tagged courses
# dominate, and the score never needs an artificial cap.
SKILL_EVIDENCE_ALPHA = 0.6

KNOWLEDGE_WEIGHTS = {"relevance": 0.55, "level_fit": 0.20, "readiness": 0.15, "preferences": 0.10}

# Explanations that are cautions rather than reasons start with this text, so
# the interface can show them as notes instead of ticks.
CAVEAT_PREFIX = "Best after learning "

# Unknown difficulty ("Not Calibrated") is neither rewarded nor penalised.
NEUTRAL_LEVEL_FIT = 0.5
# Introductory courses usually explain their own prerequisites, so a missing
# prerequisite costs a Beginner course only half as much readiness.
BEGINNER_PREREQ_TOLERANCE = 0.5


@dataclass
class SkillTargets:
    """Which skills the learner has, needs and why."""

    target_skills: list[str]          # everything the goal + desired skills require
    possessed: list[str]              # target skills the learner already has
    gaps: dict[str, float]            # missing skill -> priority


def compute_targets(profile: LearnerProfile, knowledge: SkillKnowledge) -> SkillTargets:
    """Compare the learner's skills with their career goal and desired skills."""
    career_skills = knowledge.career_skills(profile.career_goal) if profile.career_goal else []
    target_skills = list(dict.fromkeys(career_skills + profile.desired_skills))
    known = profile.known_skills
    gaps: dict[str, float] = {}
    for skill in career_skills:
        if skill not in known:
            gaps[skill] = CAREER_GAP_PRIORITY
    for skill in profile.desired_skills:
        if skill not in known:
            gaps[skill] = DESIRED_SKILL_PRIORITY
    return SkillTargets(
        target_skills=target_skills,
        possessed=[s for s in target_skills if s in known],
        gaps=gaps,
    )


class KnowledgeBasedRecommender:
    """Scores courses with transparent rules; also produces the explanations."""

    def __init__(self, courses: pd.DataFrame, skill_matrix: pd.DataFrame, knowledge: SkillKnowledge):
        self.courses = courses.set_index("course_id", drop=False)
        self.skill_matrix = skill_matrix
        self.knowledge = knowledge

    def score(self, profile: LearnerProfile, targets: SkillTargets | None = None) -> pd.DataFrame:
        """Return one row per course with each rule score and the final knowledge score."""
        targets = targets or compute_targets(profile, self.knowledge)
        index = self.courses.index
        result = pd.DataFrame(index=index)
        if not targets.gaps:
            result["knowledge_score"] = 0.0
            result["relevance"] = 0.0
            return result

        gap_skills = list(targets.gaps)
        evidence = self.skill_matrix.loc[index, gap_skills]
        priorities = pd.Series(targets.gaps)
        weighted = evidence * priorities[gap_skills] * SKILL_EVIDENCE_ALPHA
        result["relevance"] = 1 - (1 - weighted).prod(axis=1)

        # Level fit: distance between course difficulty and learner proficiency.
        distance = (self.courses["difficulty_rank"] - profile.proficiency_rank).abs()
        result["level_fit"] = (1 - distance / 2).fillna(NEUTRAL_LEVEL_FIT)

        result["readiness"] = self._readiness(evidence, profile)

        preferences = soft_preference_score(self.courses.reset_index(drop=True), profile)
        weights = dict(KNOWLEDGE_WEIGHTS)
        if preferences.isna().all():
            weights.pop("preferences")      # no soft preferences set -> re-normalise
        else:
            result["preferences"] = preferences.values
        total = sum(weights.values())
        score = sum(weights[name] * result[name] for name in weights) / total
        # Relevance is a gate: a course unrelated to every target skill is not
        # a knowledge-based recommendation, however well it fits otherwise.
        result["knowledge_score"] = np.where(result["relevance"] > 0, score, 0.0)
        return result

    def _readiness(self, evidence: pd.DataFrame, profile: LearnerProfile) -> pd.Series:
        """Share of the course's taught gap skills whose prerequisites the learner meets.

        A prerequisite also counts as met when the same course teaches it.
        """
        known = profile.known_skills
        course_teaches = self.skill_matrix.loc[evidence.index] >= ADDRESSES_THRESHOLD
        taught_gaps = evidence >= ADDRESSES_THRESHOLD
        unmet_count = pd.Series(0, index=evidence.index)
        for skill in evidence.columns:
            unmet = pd.Series(False, index=evidence.index)
            for group in self.knowledge.missing_prerequisites(skill, known):
                # The group stays unmet unless this course itself teaches one of its skills.
                unmet |= ~course_teaches[group].any(axis=1)
            unmet_count += (taught_gaps[skill] & unmet).astype(int)

        taught_count = taught_gaps.sum(axis=1)
        penalty = (unmet_count / taught_count.where(taught_count > 0)).fillna(0.0)
        is_beginner = self.courses.loc[evidence.index, "difficulty_rank"] == 1
        penalty = penalty.where(~is_beginner, penalty * BEGINNER_PREREQ_TOLERANCE)
        return 1 - penalty

    def explain(self, course_id: str, profile: LearnerProfile, targets: SkillTargets) -> list[str]:
        """Plain-language reasons derived from the same rules used for scoring."""
        reasons: list[str] = []
        course = self.courses.loc[course_id]
        row = self.skill_matrix.loc[course_id]
        direct = [s for s in targets.gaps if row[s] >= ADDRESSES_THRESHOLD]
        mentioned = [s for s in targets.gaps if 0 < row[s] < ADDRESSES_THRESHOLD]
        goal = f"your goal of {profile.career_goal}" if profile.career_goal else "your desired skills"
        if direct:
            reasons.append(f"Teaches {', '.join(direct[:3])}, missing for {goal}.")
        elif mentioned:
            reasons.append(f"Mentions {', '.join(mentioned[:3])} in its description (weaker evidence).")

        if pd.isna(course["difficulty_rank"]):
            reasons.append("Difficulty is not calibrated in the dataset.")
        elif course["difficulty_rank"] == profile.proficiency_rank:
            reasons.append(f"{course['difficulty']} level matches your {profile.proficiency} proficiency.")
        else:
            direction = "above" if course["difficulty_rank"] > profile.proficiency_rank else "below"
            reasons.append(f"{course['difficulty']} level is {direction} your {profile.proficiency} proficiency.")

        known = profile.known_skills
        course_skills = {s for s, v in row.items() if v >= ADDRESSES_THRESHOLD}
        for skill in direct[:2]:
            # A prerequisite the course teaches itself is not a gap for this course
            # (same rule as the readiness score).
            unmet = self.knowledge.missing_prerequisites(skill, known)
            missing = [g for g in unmet if not set(g) & course_skills]
            if missing:
                reasons.append(f"{CAVEAT_PREFIX}{' or '.join(missing[0])} first.")
            elif unmet:
                reasons.append(f"Also covers what {skill} builds on.")
            elif self.knowledge.prerequisite_groups(skill):
                reasons.append(f"You already meet the prerequisites for {skill}.")
        return reasons
