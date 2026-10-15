"""Learning-path generation.

The path is built greedily, step by step; it is a sensible ordering, not a
globally optimised curriculum.

1. Path skills = the learner's skill gaps, plus any missing prerequisite of a
   gap (added automatically, so the path never assumes knowledge the learner
   lacks).
2. Skills are ordered topologically over the manual prerequisite graph
   (Kahn's algorithm): a skill appears only after the skills it depends on.
   Among skills that are ready, lower stages come first, then higher
   priority, then alphabetical order.
3. Each skill gets a stage = 1 + the longest prerequisite chain *inside the
   path*. Target difficulty starts at Beginner for a skill the learner has no
   foundations in (see ``starting_rank``), otherwise at their proficiency, and
   rises one level per stage, capped at Advanced.
4. For each skill, the shared hybrid recommender picks the best course that
   directly teaches it at (preferably) the target difficulty. The request is
   *focused*: the skill is the only target, and skills taught by earlier
   steps count as known, so readiness reflects progress along the path.
   Every step gets its own course (courses already in the path are skipped);
   the dataset's skill tags are too noisy to trust one course to cover
   several stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.knowledge_base import SkillKnowledge
from src.knowledge_based import CAREER_GAP_PRIORITY, compute_targets
from src.profile import PROFICIENCY_LEVELS, LearnerProfile
from src.recommender import SkillBridgeEngine

# Missing prerequisites pulled into the path rank just below career gaps.
PREREQUISITE_PRIORITY = 0.7
CANDIDATES_PER_STEP = 10      # enough to skip courses already used in the path


@dataclass
class PathStep:
    position: int
    stage: int
    skill: str
    reason_in_path: str            # why this skill is in the path at all
    reason_for_position: str       # why it comes here
    target_difficulty: str
    course: dict | None = None     # selected course (None if the catalogue has none)
    course_reasons: list[str] = field(default_factory=list)


@dataclass
class LearningPath:
    goal: str
    steps: list[PathStep]
    notes: list[str] = field(default_factory=list)

    @property
    def course_ids(self) -> list[str]:
        return [s.course["course_id"] for s in self.steps if s.course]


def choose_prerequisite(group: list[str], path_skills: set[str]) -> str:
    """From an any-of group, prefer a skill already in the path, else the first listed."""
    for skill in group:
        if skill in path_skills:
            return skill
    return group[0]


def collect_path_skills(gaps: dict[str, float], known: set[str], knowledge: SkillKnowledge) -> tuple[dict[str, float], dict[str, str]]:
    """Gaps plus the transitive closure of their missing prerequisites."""
    skills = dict(gaps)
    added_for: dict[str, str] = {}
    queue = list(gaps)
    while queue:
        skill = queue.pop(0)
        for group in knowledge.missing_prerequisites(skill, known):
            if set(group) & set(skills):
                continue           # an alternative from this group is already in the path
            prerequisite = choose_prerequisite(group, set(skills))
            skills[prerequisite] = PREREQUISITE_PRIORITY
            added_for[prerequisite] = skill
            queue.append(prerequisite)
    return skills, added_for


def in_path_dependencies(skill: str, path_skills: set[str], knowledge: SkillKnowledge) -> list[str]:
    """Skills in the path that ``skill`` must follow."""
    return sorted({s for group in knowledge.prerequisite_groups(skill) for s in group if s in path_skills})


def topological_order(priorities: dict[str, float], knowledge: SkillKnowledge) -> tuple[list[str], dict[str, int]]:
    """Kahn's algorithm. Returns the ordered skills and each skill's stage."""
    path_skills = set(priorities)
    dependencies = {s: set(in_path_dependencies(s, path_skills, knowledge)) for s in path_skills}
    stage: dict[str, int] = {}
    ordered: list[str] = []
    ready = [s for s in path_skills if not dependencies[s]]
    def stage_of(skill: str) -> int:
        return 1 + max((stage[d] for d in dependencies[skill]), default=0)

    while ready:
        # Lowest stage first (all foundations before anything built on them),
        # then higher priority, then alphabetical for determinism.
        ready.sort(key=lambda s: (stage_of(s), -priorities[s], s))
        skill = ready.pop(0)
        stage[skill] = stage_of(skill)
        ordered.append(skill)
        for other in path_skills - set(ordered) - set(ready):
            if dependencies[other] <= set(ordered):
                ready.append(other)
    if len(ordered) != len(path_skills):     # only possible with a cyclic config
        raise ValueError("Prerequisite cycle in skill_knowledge.json")
    return ordered, stage


def starting_rank(skill: str, profile: LearnerProfile, knowledge: SkillKnowledge) -> int:
    """Level (1-3) at which the learner should start a skill they do not have yet.

    A learner's overall proficiency does not transfer to a skill they have never
    studied: a foundational skill (no prerequisites), or one whose prerequisites
    the learner has not listed, starts at Beginner. Only a skill whose
    prerequisites the learner already has starts at their overall level.
    """
    if not knowledge.prerequisite_groups(skill) or knowledge.missing_prerequisites(skill, profile.known_skills):
        return 1
    return profile.proficiency_rank


def target_difficulty(start_rank: int, stage: int) -> str:
    """Start level for the skill, one level higher per later stage, capped at Advanced."""
    return PROFICIENCY_LEVELS[min(len(PROFICIENCY_LEVELS), start_rank + stage - 1) - 1]


def generate_learning_path(engine: SkillBridgeEngine, profile: LearnerProfile) -> LearningPath:
    notes = profile.validate(engine.knowledge)
    knowledge = engine.knowledge
    targets = compute_targets(profile, knowledge)
    goal = profile.career_goal or ", ".join(profile.desired_skills) or "your goal"
    if not targets.gaps:
        notes.append("No skill gaps: choose a career goal or desired skills you do not have yet.")
        return LearningPath(goal, [], notes)

    priorities, added_for = collect_path_skills(targets.gaps, profile.known_skills, knowledge)
    order, stage = topological_order(priorities, knowledge)

    steps: list[PathStep] = []
    used_courses: set[str] = set()
    for position, skill in enumerate(order, start=1):
        dependencies = in_path_dependencies(skill, set(priorities), knowledge)
        if skill in added_for:
            reason_in_path = f"Prerequisite of {added_for[skill]}, which you have not listed."
        elif priorities[skill] == CAREER_GAP_PRIORITY:
            reason_in_path = f"Required for {profile.career_goal}; not in your current skills."
        else:
            reason_in_path = "You listed it as a desired skill."
        if dependencies:
            reason_for_position = f"Stage {stage[skill]}: builds on {', '.join(dependencies)} earlier in this path."
        elif knowledge.prerequisite_groups(skill):
            reason_for_position = f"Stage {stage[skill]}: you already meet its prerequisites, so it can start now."
        else:
            reason_for_position = f"Stage {stage[skill]}: foundational skill with no prerequisites."
        level = target_difficulty(starting_rank(skill, profile, knowledge), stage[skill])
        step = PathStep(position, stage[skill], skill, reason_in_path, reason_for_position, level)

        learned_so_far = order[:position - 1]
        step.course, step.course_reasons = _pick_course(engine, profile, skill, level, learned_so_far, used_courses)
        if step.course:
            used_courses.add(step.course["course_id"])
        else:
            notes.append(f"No course in the catalogue directly teaches {skill} under your constraints.")
        steps.append(step)
    return LearningPath(goal, steps, notes)


def _pick_course(engine: SkillBridgeEngine, profile: LearnerProfile, skill: str, level: str,
                 learned_so_far: list[str], used: set[str]) -> tuple[dict | None, list[str]]:
    """Best course that teaches ``skill`` at (preferably) ``level`` and is not already in the path."""
    result = engine.recommend_for_skill(profile, skill, CANDIDATES_PER_STEP, level=level, also_known=learned_so_far)
    for _, row in result.table.iterrows():
        if row["course_id"] not in used:
            return row.to_dict(), list(row["explanation"])
    return None, []
