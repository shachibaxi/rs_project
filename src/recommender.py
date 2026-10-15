"""SkillBridge recommendation engine: one object shared by every feature.

The Streamlit pages, the skill-gap analysis, the learning-path generator and
the offline evaluation all call this engine, so there is exactly one
implementation of each recommendation method.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

import pandas as pd

from src.collaborative_filtering import ItemKNNRecommender
from src.constraints import ConstraintReport, apply_hard_constraints
from src.content_based import ContentRecommender
from src.data_loader import PROCESSED_CATALOGUE_PATH, RATED_CATALOGUE_PATH, find_raw_csv
from src.hybrid import DEFAULT_DIVERSITY, DEFAULT_WEIGHTS, hybrid_scores, max_normalise, mmr_order
from src.knowledge_base import ADDRESSES_THRESHOLD, SkillKnowledge, compute_skill_matrix, load_knowledge
from src.knowledge_based import KnowledgeBasedRecommender, SkillTargets, compute_targets
from src.popularity import quality_score, top_rated
from src.preprocessing import build_processed_catalogue, load_processed
from src.profile import LearnerProfile
from src.rated_subset import build_rated_subset
from src.synthetic_data import load_interactions

HYBRID = "Hybrid"
CONTENT = "Content-based"
KNOWLEDGE = "Knowledge-based"
TOP_RATED = "Top-rated baseline"
COLLABORATIVE = "Collaborative (synthetic data)"
METHODS = [HYBRID, KNOWLEDGE, CONTENT, COLLABORATIVE, TOP_RATED]


def _is_older(path, than) -> bool:
    return path.stat().st_mtime < than.stat().st_mtime


@dataclass
class RecommendationResult:
    table: pd.DataFrame                       # ranked courses with scores and explanations
    method: str
    notes: list[str] = field(default_factory=list)
    constraint_report: ConstraintReport | None = None
    components_used: list[str] = field(default_factory=list)
    targets: SkillTargets | None = None


class SkillBridgeEngine:
    """Holds the data and fitted models; produces recommendations for a profile."""

    def __init__(self, courses: pd.DataFrame, rated: pd.DataFrame, knowledge: SkillKnowledge,
                 interactions: pd.DataFrame | None):
        self.courses = courses.reset_index(drop=True)
        self.by_id = self.courses.set_index("course_id", drop=False)
        self.rated = rated
        self.rated_ids = set(rated["course_id"])
        self.knowledge = knowledge
        self.skill_matrix = compute_skill_matrix(self.courses, knowledge)
        self.content = ContentRecommender(self.courses)
        self.knowledge_model = KnowledgeBasedRecommender(self.courses, self.skill_matrix, knowledge)
        self.interactions = interactions
        self.collaborative = (
            ItemKNNRecommender(interactions, self.courses["course_id"].tolist())
            if interactions is not None else None
        )
        # Quality uses genuine ratings only; unrated courses are NaN (no imputation).
        rated_quality = quality_score(rated.set_index("course_id")["rating"])
        self.quality = rated_quality.reindex(self.courses["course_id"])

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls) -> "SkillBridgeEngine":
        """Build processed files if they are missing, then load everything."""
        raw_path = find_raw_csv()
        # Rebuild derived files when missing or older than their source, so a
        # replaced raw CSV is never served from stale processed data.
        if not PROCESSED_CATALOGUE_PATH.exists() or _is_older(PROCESSED_CATALOGUE_PATH, raw_path):
            build_processed_catalogue()
        if not RATED_CATALOGUE_PATH.exists() or _is_older(RATED_CATALOGUE_PATH, PROCESSED_CATALOGUE_PATH):
            build_rated_subset()
        courses = load_processed()
        rated = pd.read_csv(RATED_CATALOGUE_PATH, usecols=["course_id", "rating"])
        return cls(courses, rated, load_knowledge(), load_interactions())

    # ---------------------------------------------------------------- helpers
    def course(self, course_id: str) -> pd.Series:
        return self.by_id.loc[course_id]

    def course_skills(self, course_id: str) -> list[str]:
        row = self.skill_matrix.loc[course_id]
        return [s for s, v in row.items() if v >= ADDRESSES_THRESHOLD]

    def profile_query(self, profile: LearnerProfile, targets: SkillTargets) -> str:
        """Text used by content-based Mode 2: interests + goal + skills still to learn."""
        parts = [profile.interests, " ".join(profile.desired_skills), " ".join(targets.gaps)]
        if profile.career_goal:
            parts.append(profile.career_goal)
        return " ".join(p for p in parts if p).strip()

    def _empty_result_note(self, report: ConstraintReport, found: int, wanted: int) -> list[str]:
        if found >= wanted:
            return []
        note = f"Only {found} course(s) satisfy every requirement (you asked for {wanted})."
        culprit = report.most_restrictive()
        if culprit:
            note += f" '{culprit}' is the most restrictive constraint ({report.removed_by[culprit]} courses excluded); relaxing it would widen the results."
        return [note]

    # ---------------------------------------------------------- recommendations
    def component_scores(self, profile: LearnerProfile, method: str = HYBRID,
                         use_collaborative: bool = True) -> dict:
        """Normalised component scores over the courses that pass the hard constraints.

        Shared by ``recommend`` and by the weight search in the evaluation, so
        both rank with exactly the same signals.
        """
        notes = profile.validate(self.knowledge)
        targets = compute_targets(profile, self.knowledge)
        report = apply_hard_constraints(self.courses, profile, self.skill_matrix, self.rated_ids)
        eligible = report.eligible[report.eligible].index

        components: dict[str, pd.Series] = {}
        query = self.profile_query(profile, targets)
        if method in (HYBRID, CONTENT) and query:
            components["content"] = max_normalise(self.content.score_query(query).loc[eligible])
        if method in (HYBRID, KNOWLEDGE) and targets.gaps:
            kb = self.knowledge_model.score(profile, targets)
            components["knowledge"] = kb.loc[eligible, "knowledge_score"]
        if method == COLLABORATIVE or (method == HYBRID and use_collaborative):
            cf_scores = self._collaborative_scores(profile, notes)
            if cf_scores is not None:
                components["collaborative"] = max_normalise(cf_scores.loc[eligible])
        if method == HYBRID and any(name != "quality" for name in components):
            components["quality"] = self.quality.loc[eligible]
        return {"components": components, "notes": notes, "targets": targets, "report": report,
                "eligible": eligible}

    def recommend(self, profile: LearnerProfile, method: str = HYBRID, n: int = 10,
                  weights: dict[str, float] | None = None, use_collaborative: bool = True,
                  diversity: float | None = None) -> RecommendationResult:
        """Rank courses for ``profile``.

        ``use_collaborative=False`` keeps the hybrid to signals from real data
        (the website does this, because the CF model is trained on synthetic
        learners). ``diversity`` is the MMR lambda for the hybrid (default
        ``DEFAULT_DIVERSITY``; 1.0 = no diversification). Single-technique
        methods are never re-ranked, so they show the technique on its own.
        """
        if n <= 0:
            raise ValueError("Number of recommendations must be positive.")
        if method not in METHODS:
            raise ValueError(f"Unknown method {method!r}; choose one of {METHODS}.")
        parts = self.component_scores(profile, method, use_collaborative)
        components, notes = parts["components"], parts["notes"]
        targets, report = parts["targets"], parts["report"]

        if method == TOP_RATED:
            ranked = top_rated(self.rated.merge(self.courses[["course_id"]]), set(parts["eligible"]), n)
            scores = ranked.set_index("course_id")["score"]
            notes.append("Non-personalised baseline: ranks by average rating only. "
                         "The dataset has no rating counts, so this is not a popularity measure.")
            return self._package(scores, {"quality": scores}, method, profile, targets, report, notes, n)

        # A method needs at least one personalised signal to say anything.
        personal = [c for c in components if c != "quality"]
        if not personal:
            notes.append(self._missing_input_message(method))
            return RecommendationResult(pd.DataFrame(), method, notes, report, [], targets)

        if method == HYBRID:
            scores = hybrid_scores(components, {**DEFAULT_WEIGHTS, **(weights or {})})
            lam = DEFAULT_DIVERSITY if diversity is None else diversity
            order = mmr_order(scores, n, self.content.matrix, self.content.position, lam)
            return self._package(scores, components, method, profile, targets, report, notes, n, order=order)

        scores = components[personal[0]]
        scores = scores[scores > 0]
        return self._package(scores, components, method, profile, targets, report, notes, n)

    def recommend_for_skill(self, profile: LearnerProfile, skill: str, n: int,
                            level: str | None = None, also_known: list[str] | None = None) -> RecommendationResult:
        """Best courses for ONE skill, used by the skill-gap page and the learning path.

        The request is focused: the skill is the only target (so courses are not
        rewarded for touching many other gaps), courses must teach it, skills in
        ``also_known`` count as already learned, and ``level`` (if given) is the
        learner's level for this step, preferred softly unless difficulty is strict.
        The learner's own hard constraints still apply.
        """
        focused = deepcopy(profile)
        focused.career_goal = None
        focused.desired_skills = [skill]
        focused.current_skills = list(dict.fromkeys(profile.current_skills + (also_known or [])))
        focused.must_cover_skills = [skill]
        focused.strict_must_cover = True
        if level is not None:
            focused.proficiency = level
            if not focused.strict_difficulty:
                focused.preferred_difficulties = [level]
        result = self.recommend(focused, HYBRID, n, use_collaborative=False)
        # The focused request treats the skill as a "desired skill"; reword for the caller.
        if not result.table.empty:
            result.table["explanation"] = [
                [f"Directly teaches {skill}." if r.startswith("Teaches ") else r for r in reasons]
                for reasons in result.table["explanation"]
            ]
        return result

    def _collaborative_scores(self, profile: LearnerProfile, notes: list[str]) -> pd.Series | None:
        if self.collaborative is None:
            notes.append("Collaborative filtering is unavailable: synthetic interaction data has not been generated.")
            return None
        if not profile.completed_course_ids:
            notes.append("Collaborative filtering needs at least one completed course in your profile (cold start).")
            return None
        scores = self.collaborative.score(profile.completed_course_ids)
        if scores is None:
            notes.append("None of your completed courses appear in the synthetic interaction data (item cold start).")
        return scores

    def _missing_input_message(self, method: str) -> str:
        if method == KNOWLEDGE:
            return "Knowledge-based recommendations need a career goal or desired skills that you do not already have."
        if method == CONTENT:
            return "Content-based recommendations need interests, desired skills or a career goal."
        if method == COLLABORATIVE:
            return "Add completed courses to your profile to get collaborative recommendations."
        return "Add a career goal, desired skills, interests or completed courses to get recommendations."

    # -------------------------------------------------------------- packaging
    def _package(self, scores: pd.Series, components: dict[str, pd.Series], method: str,
                 profile: LearnerProfile, targets: SkillTargets, report: ConstraintReport,
                 notes: list[str], n: int, order: list[str] | None = None) -> RecommendationResult:
        scores = scores.dropna()
        if order is None:
            # Stable tie-break by course id keeps results deterministic.
            ordered = pd.DataFrame({"course_id": scores.index.to_numpy(), "score": scores.to_numpy()})
            order = ordered.sort_values(["score", "course_id"], ascending=[False, True])["course_id"].tolist()
        top_ids = order[:n]
        ordered = pd.DataFrame({"course_id": top_ids, "score": scores.loc[top_ids].to_numpy()})

        table = self.by_id.loc[top_ids, ["course_id", "title", "provider", "difficulty", "rating",
                                          "url", "skills", "domain", "subdomain"]].reset_index(drop=True)
        table.insert(0, "rank", range(1, len(table) + 1))
        table["score"] = ordered["score"].to_numpy()
        for name, values in components.items():
            table[f"{name}_score"] = values.reindex(top_ids).to_numpy()
        table["matched_skills"] = [
            [s for s in self.course_skills(c) if s in targets.gaps or s in profile.desired_skills] for c in top_ids
        ]
        table["explanation"] = [self.explain(c, method, profile, targets, components) for c in top_ids]
        table["method"] = method

        notes = notes + self._empty_result_note(report, len(table), n)
        return RecommendationResult(table, method, notes, report, list(components), targets)

    def explain(self, course_id: str, method: str, profile: LearnerProfile, targets: SkillTargets,
                components: dict[str, pd.Series]) -> list[str]:
        """Reasons built from the signals that actually produced the score."""
        reasons: list[str] = []
        if "knowledge" in components:
            reasons += self.knowledge_model.explain(course_id, profile, targets)
        if "content" in components:
            # Keep only terms that occur inside one phrase the learner gave (a bigram
            # such as "learning data" can otherwise span two separate skills).
            phrases = [p.lower() for p in [profile.interests, profile.career_goal or "",
                                           *profile.desired_skills, *targets.gaps] if p]
            candidates = [t for t in self.content.shared_terms(self.profile_query(profile, targets), course_id, top=8)
                          if any(t in phrase for phrase in phrases)]
            # Prefer phrases ("data visualization") and drop single words they already cover.
            phrases_first = [t for t in candidates if " " in t]
            words = [t for t in candidates if " " not in t and not any(t in p.split() for p in phrases_first)]
            terms = (phrases_first + words)[:3]
            if terms:
                reasons.append("Covers topics you're after: " + ", ".join(terms) + ".")
        if "collaborative" in components and self.collaborative is not None:
            support = self.collaborative.neighbours_in_history(course_id, profile.completed_course_ids)
            if support:
                titles = "; ".join(self.by_id.at[c, "title"] for c in support)
                reasons.append(f"Synthetic learners who took {titles} also took this.")
        if "quality" in components:
            rating = self.by_id.at[course_id, "rating"]
            reasons.append(f"Rated {rating:.1f}/5 on Coursera." if pd.notna(rating)
                           else "No rating in the dataset; ranked on the other signals only.")
        return reasons

    # ------------------------------------------------------------ similar items
    def similar_courses(self, course_id: str, n: int = 10, profile: LearnerProfile | None = None) -> pd.DataFrame:
        """Content-based Mode 1, optionally restricted by the profile's hard constraints."""
        if profile is None:
            neighbours = self.content.similar_courses(course_id, n)
        else:
            profile.validate(self.knowledge)
            report = apply_hard_constraints(self.courses, profile, self.skill_matrix, self.rated_ids)
            neighbours = self.content.similar_courses(course_id, len(self.courses))
            neighbours = neighbours[neighbours["course_id"].map(report.eligible)].head(n)
        table = self.by_id.loc[neighbours["course_id"], ["course_id", "title", "provider", "difficulty",
                                                         "rating", "url", "skills"]].reset_index(drop=True)
        table.insert(0, "rank", range(1, len(table) + 1))
        table["score"] = neighbours["score"].to_numpy()
        table["explanation"] = [
            ["Shares terms: " + ", ".join(self.content.shared_terms_between(course_id, c)) + "."]
            for c in table["course_id"]
        ]
        return table
