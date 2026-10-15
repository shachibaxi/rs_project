"""Offline evaluation of the SkillBridge recommenders.

Run with:  python -m src.evaluation

Nothing here measures user satisfaction; there was no user study. Three
experiments are run, each with an explicitly stated relevance label:

A. Similar courses (REAL catalogue, PROXY label)
   For sampled query courses, a recommended course counts as relevant when it
   has the same Coursera subdomain as the query. Subdomains are editorial
   labels and are NOT used as a feature by any recommender, so the label is
   independent of the input text, but "same subdomain" is only a proxy for
   "useful similar course".

B. Learner profiles (REAL catalogue, PROXY label)
   Profiles are generated systematically for every career goal x proficiency
   x share of required skills already known. A course counts as relevant when
   its Coursera subdomain is one of the career's related subdomains (a manual
   mapping in skill_knowledge.json, used by no recommender). A graded variant
   also rewards the right level: gain 2 = on-topic at the learner's level,
   1 = on-topic at an adjacent level. The hybrid's weights are chosen by
   career-split cross-validation (chosen on some careers, scored on the
   others), so its reported scores are held-out. Additional diagnostics:
   gap coverage, level fit, diversity and context sensitivity.

C. Interaction prediction (SYNTHETIC data)
   Each synthetic learner's latest 20% of courses are held out; models are
   trained on the rest. Hybrid weights are tuned on a validation split carved
   out of the TRAINING data only. Results describe how well each method
   recovers the synthetic generator's patterns, not real learners.
"""

from __future__ import annotations

import itertools
import json
from copy import deepcopy
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import sparse

from src.collaborative_filtering import ItemKNNRecommender
from src.data_loader import REPORTS_DIR
from src.hybrid import DEFAULT_DIVERSITY, DEFAULT_WEIGHTS, hybrid_scores, mmr_order
from src.profile import PROFICIENCY_LEVELS, PROFICIENCY_RANK, LearnerProfile
from src.recommender import CONTENT, HYBRID, KNOWLEDGE, TOP_RATED, SkillBridgeEngine
from src.knowledge_base import ADDRESSES_THRESHOLD

K = 10
RANDOM_SEED = 7
QUERY_COURSES_EXPERIMENT_A = 400
KNOWN_SKILL_SHARES = [0.0, 1 / 3, 2 / 3]
TEST_SHARE = 0.2
BOOTSTRAP_RESAMPLES = 1000
RANDOM = "Random baseline"
CF = "Item-kNN CF"
WEIGHT_GRID = [0.0, 0.25, 0.5, 1.0]
CF_NEIGHBOUR_GRID = [30, 100, 400]
CF_SHRINKAGE_GRID = [5.0, 20.0, 50.0]
POPULARITY = "Most-completed (synthetic popularity)"
ORACLE = "Generator oracle (upper reference)"

RESULTS_PATH = REPORTS_DIR / "evaluation_results.csv"
PAIRWISE_PATH = REPORTS_DIR / "evaluation_pairwise.csv"
METADATA_PATH = REPORTS_DIR / "evaluation_metadata.json"


# ============================================================== metrics
def precision_at_k(ranked: list, relevant: set, k: int = K) -> float:
    """Share of the top-k that is relevant (denominator is k, even for short lists)."""
    return len(set(ranked[:k]) & relevant) / k


def recall_at_k(ranked: list, relevant: set, k: int = K) -> float:
    """Share of the relevant items found in the top-k."""
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else float("nan")


def ndcg_at_k(ranked: list, relevant: set, k: int = K) -> float:
    """Binary NDCG: DCG = sum rel_i / log2(i + 1) for positions i = 1..k, divided
    by the DCG of an ideal list that puts min(|relevant|, k) hits first."""
    if not relevant:
        return float("nan")
    dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(ranked[:k]) if item in relevant)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal


def reciprocal_rank(ranked: list, relevant: set, k: int = K) -> float:
    """1 / position of the first relevant item in the top-k (0 if none)."""
    for i, item in enumerate(ranked[:k]):
        if item in relevant:
            return 1.0 / (i + 1)
    return 0.0


def hit_at_k(ranked: list, relevant: set, k: int = K) -> float:
    return float(bool(set(ranked[:k]) & relevant))


def catalogue_coverage(lists: list[list], catalogue_size: int) -> float:
    """Share of the catalogue that appears in at least one recommendation list."""
    if catalogue_size <= 0:
        return float("nan")
    return len({item for items in lists for item in items}) / catalogue_size


def intra_list_diversity(ranked: list, vectors: sparse.csr_matrix, position: dict) -> float:
    """Mean pairwise cosine DISTANCE (1 - similarity) between TF-IDF vectors of a list."""
    rows = [position[c] for c in ranked if c in position]
    if len(rows) < 2:
        return float("nan")
    block = vectors[rows]
    similarity = (block @ block.T).toarray()
    upper = similarity[np.triu_indices(len(rows), k=1)]
    return float(np.mean(1.0 - upper))


def jaccard(a: list, b: list) -> float:
    union = set(a) | set(b)
    return len(set(a) & set(b)) / len(union) if union else float("nan")


def ranking_metrics(ranked: list, relevant: set, k: int = K) -> dict[str, float]:
    return {
        f"Precision@{k}": precision_at_k(ranked, relevant, k),
        f"Recall@{k}": recall_at_k(ranked, relevant, k),
        f"NDCG@{k}": ndcg_at_k(ranked, relevant, k),
        f"MRR@{k}": reciprocal_rank(ranked, relevant, k),
        f"HitRate@{k}": hit_at_k(ranked, relevant, k),
    }


def paired_bootstrap(a: np.ndarray, b: np.ndarray, resamples: int = BOOTSTRAP_RESAMPLES,
                     seed: int = RANDOM_SEED) -> dict[str, float]:
    """95% percentile bootstrap CI for mean(a - b), resampling evaluation units with replacement."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    keep = ~(np.isnan(a) | np.isnan(b))
    diff = a[keep] - b[keep]
    if diff.size == 0:
        return {"difference": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    means = rng.choice(diff, size=(resamples, diff.size), replace=True).mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {"difference": float(diff.mean()), "ci_low": float(low), "ci_high": float(high), "n": int(diff.size)}


def summarise(per_unit: pd.DataFrame, experiment: str, source: str) -> list[dict]:
    """Mean of every metric per method; units with an undefined metric (NaN) are skipped."""
    rows = []
    for method, group in per_unit.groupby("method", sort=False):
        for metric in [c for c in group.columns if c not in ("method", "unit")]:
            values = group[metric].dropna()
            rows.append({"experiment": experiment, "data_source": source, "method": method,
                         "metric": metric, "value": float(values.mean()) if len(values) else float("nan"),
                         "n_evaluated": int(len(values))})
    return rows


# ======================================================= Experiment A
def experiment_similar_courses(engine: SkillBridgeEngine) -> tuple[list[dict], pd.DataFrame]:
    courses = engine.courses[engine.courses["subdomain"] != ""]
    rng = np.random.default_rng(RANDOM_SEED)
    queries = rng.choice(courses["course_id"].to_numpy(), size=min(QUERY_COURSES_EXPERIMENT_A, len(courses)), replace=False)
    subdomain = engine.by_id["subdomain"]
    all_ids = engine.courses["course_id"].to_numpy()
    rated_order = engine.rated.sort_values(["rating", "course_id"], ascending=[False, True])["course_id"].tolist()

    records, lists = [], {CONTENT: [], RANDOM: [], TOP_RATED: []}
    for query in queries:
        relevant = set(courses.loc[courses["subdomain"] == subdomain[query], "course_id"]) - {query}
        candidates = {
            CONTENT: engine.content.similar_courses(query, K)["course_id"].tolist(),
            RANDOM: [c for c in rng.choice(all_ids, size=K + 1, replace=False) if c != query][:K],
            TOP_RATED: [c for c in rated_order[:K + 1] if c != query][:K],
        }
        for method, ranked in candidates.items():
            lists[method].append(ranked)
            records.append({"method": method, "unit": query, **ranking_metrics(ranked, relevant),
                            "Diversity (ILD)": intra_list_diversity(ranked, engine.content.matrix, engine.content.position)})
    per_unit = pd.DataFrame(records)
    rows = summarise(per_unit, "A. Similar courses", "Real catalogue - proxy label (same subdomain)")
    for method, method_lists in lists.items():
        rows.append({"experiment": "A. Similar courses", "data_source": "Real catalogue - proxy label (same subdomain)",
                     "method": method, "metric": "Catalogue coverage",
                     "value": catalogue_coverage(method_lists, len(all_ids)), "n_evaluated": len(method_lists)})
    return rows, per_unit


# ======================================================= Experiment B
def generate_evaluation_profiles(engine: SkillBridgeEngine) -> list[tuple[str, LearnerProfile]]:
    """Every career x proficiency x known-skill share (seeded choice of known skills)."""
    rng = np.random.default_rng(RANDOM_SEED)
    profiles = []
    for career in engine.knowledge.careers:
        required = engine.knowledge.career_skills(career)
        for share in KNOWN_SKILL_SHARES:
            known = list(rng.choice(required, size=round(share * len(required)), replace=False))
            for level in PROFICIENCY_LEVELS:
                key = f"{career}|{share:.2f}"
                profiles.append((key, LearnerProfile(career_goal=career, current_skills=known, proficiency=level)))
    return profiles


HYBRID_CV = "Hybrid (cross-validated)"
HYBRID_APP = "Hybrid (cross-validated) + diversity"
HYBRID_HAND = "Hybrid (hand-set weights)"
GRADED = f"NDCG@{K} (topic + level)"
HAND_SET_WEIGHTS = {"content": 0.35, "knowledge": 0.45, "quality": 0.20, "collaborative": 0.15}
HYBRID_GRID = {"content": [0.05, 0.1, 0.2, 0.3], "knowledge": [0.5, 0.6, 0.7, 0.8], "quality": [0.0, 0.1, 0.2]}
CV_FOLDS = 2


def graded_gains(engine: SkillBridgeEngine, profile: LearnerProfile) -> pd.Series:
    """Graded proxy relevance: 2 = career-related subdomain AND exactly the learner's level,
    1 = related subdomain at an adjacent level, 0 otherwise (unknown level counts as 0)."""
    related = set(engine.knowledge.careers[profile.career_goal]["related_subdomains"])
    on_topic = engine.courses["subdomain"].isin(related).to_numpy()
    distance = (engine.courses["difficulty_rank"] - profile.proficiency_rank).abs().to_numpy()
    gains = np.where(on_topic & (distance == 0), 2, np.where(on_topic & (distance == 1), 1, 0))
    return pd.Series(gains, index=engine.courses["course_id"].to_numpy())


def graded_ndcg_at_k(ranked: list, gains: pd.Series, k: int = K) -> float:
    """NDCG with graded gains: DCG = sum gain_i / log2(i + 1)."""
    ideal = np.sort(gains.to_numpy())[::-1][:k]
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal))
    if idcg == 0:
        return float("nan")
    dcg = sum(gains.get(c, 0) / np.log2(i + 2) for i, c in enumerate(ranked[:k]))
    return dcg / idcg


def select_hybrid(engine: SkillBridgeEngine, cached: list[dict]) -> dict:
    """Weights that maximise mean graded NDCG@10 over the ``cached`` (training) profiles.

    Plain ranking (no diversification) is used here, so the weights are chosen
    on relevance alone; the cost of the application's diversification is
    measured separately as its own method row.
    """
    best = None
    names = list(HYBRID_GRID)
    for values in itertools.product(*HYBRID_GRID.values()):
        weights = dict(zip(names, values))
        scores_per_profile = []
        for item in cached:
            scores = hybrid_scores(item["components"], {**weights, "collaborative": 0.0})
            ranked = mmr_order(scores, K, engine.content.matrix, engine.content.position, diversity=1.0)
            scores_per_profile.append(graded_ndcg_at_k(ranked, item["gains"]))
        score = float(np.nanmean(scores_per_profile))
        if best is None or score > best["validation_graded_ndcg"] + 1e-12:
            best = {"weights": weights, "validation_graded_ndcg": score}
    return best


def cross_validate_hybrid(engine: SkillBridgeEngine, profiles: list[tuple[str, LearnerProfile]]) -> dict:
    """Career-split cross-validation: settings are chosen on some careers and used
    only on the other careers, so no profile is ranked with settings tuned on it."""
    cached = []
    for key, profile in profiles:
        parts = engine.component_scores(deepcopy(profile), HYBRID, use_collaborative=False)
        cached.append({"key": key, "career": profile.career_goal, "components": parts["components"],
                       "gains": graded_gains(engine, profile)})
    careers = sorted({item["career"] for item in cached})
    fold_of = {career: i % CV_FOLDS for i, career in enumerate(careers)}
    per_fold = {}
    for fold in range(CV_FOLDS):
        train = [item for item in cached if fold_of[item["career"]] != fold]
        per_fold[fold] = select_hybrid(engine, train)
    final = select_hybrid(engine, cached)          # used by the application (DEFAULT_WEIGHTS)
    return {"fold_of_career": fold_of, "per_fold": per_fold, "final": final}


def experiment_profiles(engine: SkillBridgeEngine) -> tuple[list[dict], pd.DataFrame, pd.DataFrame, dict]:
    rng = np.random.default_rng(RANDOM_SEED)
    all_ids = engine.courses["course_id"].to_numpy()
    subdomain = engine.by_id["subdomain"]
    difficulty_rank = engine.by_id["difficulty_rank"]
    addresses = engine.skill_matrix >= ADDRESSES_THRESHOLD
    methods = [HYBRID_APP, HYBRID_CV, HYBRID_HAND, KNOWLEDGE, CONTENT, TOP_RATED, RANDOM]
    profiles = generate_evaluation_profiles(engine)
    cv = cross_validate_hybrid(engine, profiles)

    records, lists, by_context = [], {m: [] for m in methods}, []
    for key, profile in profiles:
        related = set(engine.knowledge.careers[profile.career_goal]["related_subdomains"])
        relevant = set(all_ids[subdomain.loc[all_ids].isin(related).to_numpy()])
        gains = graded_gains(engine, profile)
        gaps = [s for s in engine.knowledge.career_skills(profile.career_goal) if s not in profile.current_skills]
        held_out = cv["per_fold"][cv["fold_of_career"][profile.career_goal]]
        for method in methods:
            if method == RANDOM:
                ranked = list(rng.choice(all_ids, size=K, replace=False))
            elif method in (HYBRID_CV, HYBRID_APP):
                diversity = 1.0 if method == HYBRID_CV else DEFAULT_DIVERSITY
                ranked = engine.recommend(deepcopy(profile), HYBRID, K, weights=held_out["weights"],
                                          use_collaborative=False, diversity=diversity).table["course_id"].tolist()
            elif method == HYBRID_HAND:
                ranked = engine.recommend(deepcopy(profile), HYBRID, K, weights=HAND_SET_WEIGHTS,
                                          use_collaborative=False, diversity=1.0).table["course_id"].tolist()
            else:
                ranked = engine.recommend(deepcopy(profile), method, K).table.get("course_id", pd.Series(dtype=str)).tolist()
            lists[method].append(ranked)
            by_context.append({"method": method, "context": key, "level": profile.proficiency, "ranked": ranked})
            gap_cover = (addresses.loc[ranked, gaps].any(axis=0).mean() if ranked and gaps else float("nan"))
            level_fit = (float(np.mean([difficulty_rank[c] == PROFICIENCY_RANK[profile.proficiency] for c in ranked]))
                         if ranked else float("nan"))
            records.append({
                "method": method, "unit": f"{key}|{profile.proficiency}",
                GRADED: graded_ndcg_at_k(ranked, gains),
                **ranking_metrics(ranked, relevant),
                "Gap coverage@10": gap_cover,
                "Level fit@10": level_fit,
                "Diversity (ILD)": intra_list_diversity(ranked, engine.content.matrix, engine.content.position),
            })
    per_unit = pd.DataFrame(records)
    source = "Real catalogue - proxy labels (career-related subdomain; graded by level)"
    rows = summarise(per_unit, "B. Learner profiles", source)
    for method in methods:
        rows.append({"experiment": "B. Learner profiles", "data_source": source, "method": method,
                     "metric": "Catalogue coverage", "value": catalogue_coverage(lists[method], len(all_ids)),
                     "n_evaluated": len(lists[method])})

    # Context sensitivity: overlap between the Beginner and Advanced lists for
    # the same goal and skills. 1.0 = context ignored; lower = context matters.
    context = pd.DataFrame(by_context)
    for method in methods:
        subset = context[context["method"] == method]
        overlaps = []
        for _, group in subset.groupby("context"):
            lists_by_level = dict(zip(group["level"], group["ranked"]))
            overlaps.append(jaccard(lists_by_level["Beginner"], lists_by_level["Advanced"]))
        rows.append({"experiment": "B. Learner profiles", "data_source": source, "method": method,
                     "metric": "Beginner-vs-Advanced overlap (Jaccard)", "value": float(np.nanmean(overlaps)),
                     "n_evaluated": len(overlaps)})
    details = {
        "profiles": int(per_unit["unit"].nunique()), "known_skill_shares": KNOWN_SKILL_SHARES,
        "cv_folds": CV_FOLDS, "fold_of_career": cv["fold_of_career"],
        "fold_selections": {str(f): v for f, v in cv["per_fold"].items()},
        "selected_for_application": cv["final"],
        "hybrid_grid": HYBRID_GRID, "application_diversity": DEFAULT_DIVERSITY, "hand_set_weights": HAND_SET_WEIGHTS,
    }
    return rows, per_unit, context, details


# ======================================================= Experiment C
def split_interactions(interactions: pd.DataFrame, test_share: float = TEST_SHARE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per learner, hold out the latest ``test_share`` of courses (at least one)."""
    ordered = interactions.sort_values(["learner_id", "sequence"])
    position_from_end = ordered.groupby("learner_id").cumcount(ascending=False)
    size = ordered.groupby("learner_id")["course_id"].transform("size")
    n_test = np.maximum(1, np.round(size * test_share)).astype(int)
    is_test = position_from_end < n_test
    return ordered[~is_test].copy(), ordered[is_test].copy()


class SyntheticScorer:
    """Dense learner x course score matrices for every component (fast grid search)."""

    def __init__(self, engine: SkillBridgeEngine, train: pd.DataFrame, learners: list[str],
                 cf_params: dict | None = None):
        self.engine = engine
        self.course_ids = engine.courses["course_id"].to_numpy()
        position = engine.content.position
        learner_index = {l: i for i, l in enumerate(learners)}
        rows = train["learner_id"].map(learner_index).to_numpy()
        cols = train["course_id"].map(position).to_numpy()
        self.history = sparse.csr_matrix((np.ones(len(train)), (rows, cols)),
                                         shape=(len(learners), len(self.course_ids)))
        self.seen = self.history.toarray().astype(bool)

        self.cf_params = cf_params or {}
        self.components = {
            "collaborative": self.collaborative_scores(train, **self.cf_params),
            "content": self._row_max_normalise(self._content_scores()),
            "knowledge": self._knowledge_scores(train, learners),
        }
        quality = engine.quality.reindex(self.course_ids).to_numpy()
        self.quality = quality                        # NaN for unrated courses

    def collaborative_scores(self, train: pd.DataFrame, **params) -> np.ndarray:
        """score(u, i) = sum_j history(u, j) * sim_k(i, j), row-max normalised."""
        cf = ItemKNNRecommender(train, list(self.course_ids), **params)
        return self._row_max_normalise(np.asarray((self.history @ cf.similarity.T).todense()))

    def _content_scores(self) -> np.ndarray:
        counts = np.asarray(self.history.sum(axis=1)).ravel()
        counts[counts == 0] = 1
        # Mean TF-IDF vector of each learner's history, L2-normalised; kept
        # sparse because the vocabulary is large.
        centroids = (sparse.diags(1.0 / counts) @ self.history @ self.engine.content.matrix).tocsr()
        norms = np.sqrt(np.asarray(centroids.multiply(centroids).sum(axis=1))).ravel()
        norms[norms == 0] = 1
        centroids = sparse.diags(1.0 / norms) @ centroids
        return (centroids @ self.engine.content.matrix.T).toarray()

    def _knowledge_scores(self, train: pd.DataFrame, learners: list[str]) -> np.ndarray:
        personas = train.drop_duplicates("learner_id").set_index("learner_id")[["persona_career", "persona_level"]]
        cache: dict[tuple, np.ndarray] = {}
        output = np.zeros((len(learners), len(self.course_ids)))
        for i, learner in enumerate(learners):
            career, level = personas.loc[learner]
            key = (career, int(level))
            if key not in cache:
                profile = LearnerProfile(career_goal=career, proficiency=PROFICIENCY_LEVELS[int(level) - 1])
                cache[key] = self.engine.knowledge_model.score(profile)["knowledge_score"].reindex(self.course_ids).to_numpy()
            output[i] = cache[key]
        return output

    @staticmethod
    def _row_max_normalise(matrix: np.ndarray) -> np.ndarray:
        peak = matrix.max(axis=1, keepdims=True)
        peak[peak <= 0] = 1
        return matrix / peak

    def hybrid(self, weights: dict[str, float]) -> np.ndarray:
        """Weighted mean of components; quality weight dropped where a course is unrated."""
        total = sum(weights[c] * self.components[c] for c in self.components)
        weight_sum = np.full(self.seen.shape, sum(weights[c] for c in self.components), dtype=float)
        has_quality = ~np.isnan(self.quality)
        total = total + weights["quality"] * np.where(has_quality, self.quality, 0.0)
        weight_sum = weight_sum + weights["quality"] * has_quality
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(weight_sum > 0, total / weight_sum, 0.0)

    def top_k(self, scores: np.ndarray, k: int = K) -> list[list[str]]:
        masked = np.where(self.seen, -np.inf, scores)
        top = np.argpartition(-masked, k, axis=1)[:, :k]
        ordered = np.take_along_axis(top, np.argsort(-np.take_along_axis(masked, top, axis=1), axis=1), axis=1)
        return [list(self.course_ids[row]) for row in ordered]


def _evaluate_lists(lists: list[list[str]], learners: list[str], held_out: dict[str, set]) -> list[dict]:
    return [ranking_metrics(ranked, held_out[learner]) for ranked, learner in zip(lists, learners)]


def _validation_ndcg(scorer: "SyntheticScorer", scores: np.ndarray, learners: list[str], held_out: dict) -> float:
    lists = scorer.top_k(scores)
    return float(np.nanmean([ndcg_at_k(r, held_out[l]) for r, l in zip(lists, learners) if l in held_out]))


def tune_on_validation(engine: SkillBridgeEngine, train: pd.DataFrame) -> dict:
    """Choose CF hyperparameters, then hybrid weights, on a validation split carved
    out of the TRAINING data. The test split is never used for any choice."""
    inner_train, validation = split_interactions(train)
    learners = sorted(inner_train["learner_id"].unique())
    validation = validation[validation["learner_id"].isin(learners)]
    held_out = validation.groupby("learner_id")["course_id"].apply(set).to_dict()
    scorer = SyntheticScorer(engine, inner_train, learners)

    cf_results = {}
    for k, shrinkage in itertools.product(CF_NEIGHBOUR_GRID, CF_SHRINKAGE_GRID):
        scores = scorer.collaborative_scores(inner_train, n_neighbours=k, shrinkage=shrinkage)
        cf_results[(k, shrinkage)] = _validation_ndcg(scorer, scores, learners, held_out)
    best_k, best_shrinkage = max(cf_results, key=cf_results.get)
    cf_params = {"n_neighbours": best_k, "shrinkage": best_shrinkage}
    scorer.components["collaborative"] = scorer.collaborative_scores(inner_train, **cf_params)

    best, best_score = None, -1.0
    names = ["content", "knowledge", "quality", "collaborative"]
    for combo in itertools.product(WEIGHT_GRID, repeat=len(names)):
        if sum(combo) == 0:
            continue
        weights = dict(zip(names, combo))
        score = _validation_ndcg(scorer, scorer.hybrid(weights), learners, held_out)
        if score > best_score:
            best, best_score = weights, score
    return {
        "cf_params": cf_params,
        "cf_validation_ndcg": {f"k={k}, shrinkage={lam}": round(v, 5) for (k, lam), v in cf_results.items()},
        "hybrid_weights": best,
        "hybrid_validation_ndcg": best_score,
    }


def generator_oracle_scores(engine: SkillBridgeEngine, train: pd.DataFrame, learners: list[str],
                            course_ids: np.ndarray) -> np.ndarray:
    """The synthetic generator's own sampling weights for each learner.

    Not a recommender (it knows the hidden persona and appeal values). It shows
    the best ranking achievable on this noisy synthetic task, which puts the
    real methods' small absolute numbers in context.
    """
    from src import synthetic_data as gen

    rng = np.random.default_rng(gen.RANDOM_SEED)
    appeal = rng.lognormal(mean=0.0, sigma=gen.APPEAL_SIGMA, size=len(engine.courses))
    courses = engine.courses.set_index("course_id").loc[course_ids]
    subdomains, domains = courses["subdomain"].to_numpy(), courses["domain"].to_numpy()
    difficulty = courses["difficulty_rank"].to_numpy()
    appeal = pd.Series(appeal, index=engine.courses["course_id"]).loc[course_ids].to_numpy()
    sub_to_domain = dict(zip(subdomains, domains))
    personas = train.drop_duplicates("learner_id").set_index("learner_id")
    output = np.zeros((len(learners), len(course_ids)))
    for i, learner in enumerate(learners):
        career, level = personas.at[learner, "persona_career"], int(personas.at[learner, "persona_level"])
        related = set(engine.knowledge.careers[career]["related_subdomains"])
        related_domains = {sub_to_domain.get(s) for s in related}
        affinity = np.where(np.isin(subdomains, list(related)), gen.RELATED_SUBDOMAIN_AFFINITY,
                            np.where(np.isin(domains, list(related_domains)), gen.SAME_DOMAIN_AFFINITY, gen.OTHER_AFFINITY))
        level_affinity = np.full(len(course_ids), gen.UNKNOWN_LEVEL_AFFINITY)
        known = ~np.isnan(difficulty)
        level_affinity[known] = [gen.LEVEL_AFFINITY[g] for g in np.abs(difficulty[known] - level).astype(int)]
        output[i] = appeal * affinity * level_affinity
    return output


def experiment_synthetic(engine: SkillBridgeEngine) -> tuple[list[dict], pd.DataFrame, dict]:
    if engine.interactions is None:
        return [], pd.DataFrame(), {"skipped": "synthetic interactions not generated"}
    train, test = split_interactions(engine.interactions)
    # Guard against leakage: no (learner, course) pair may be in both splits.
    overlap = train.merge(test, on=["learner_id", "course_id"])
    assert overlap.empty, "train/test leakage"

    tuning = tune_on_validation(engine, train)
    tuned_weights = tuning["hybrid_weights"]
    learners = sorted(train["learner_id"].unique())
    held_out = test.groupby("learner_id")["course_id"].apply(set).to_dict()
    scorer = SyntheticScorer(engine, train, learners, cf_params=tuning["cf_params"])
    completions = np.bincount(train["course_id"].map(engine.content.position), minlength=len(scorer.course_ids))

    rng = np.random.default_rng(RANDOM_SEED)
    rated_quality = np.nan_to_num(scorer.quality, nan=-1.0)       # unrated never rank in top-rated
    method_scores = {
        "Hybrid (tuned on validation)": scorer.hybrid(tuned_weights),
        "Hybrid (default weights)": scorer.hybrid(DEFAULT_WEIGHTS),
        CF: scorer.components["collaborative"],
        "Content-based (history)": scorer.components["content"],
        "Knowledge-based (persona)": scorer.components["knowledge"],
        POPULARITY: np.tile(completions.astype(float), (len(learners), 1)),
        TOP_RATED: np.tile(rated_quality, (len(learners), 1)),
        RANDOM: rng.random(scorer.seen.shape),
        ORACLE: generator_oracle_scores(engine, train, learners, scorer.course_ids),
    }
    records, lists_by_method = [], {}
    for method, scores in method_scores.items():
        lists = scorer.top_k(scores)
        lists_by_method[method] = lists
        for learner, ranked, metrics in zip(learners, lists, _evaluate_lists(lists, learners, held_out)):
            records.append({"method": method, "unit": learner, **metrics})
    per_unit = pd.DataFrame(records)
    source = "SYNTHETIC interactions (generated, not real learners)"
    rows = summarise(per_unit, "C. Synthetic interactions", source)
    for method, lists in lists_by_method.items():
        rows.append({"experiment": "C. Synthetic interactions", "data_source": source, "method": method,
                     "metric": "Catalogue coverage", "value": catalogue_coverage(lists, len(scorer.course_ids)),
                     "n_evaluated": len(lists)})
    details = {
        "train_interactions": int(len(train)), "test_interactions": int(len(test)),
        "learners": len(learners), **tuning,
        "weight_grid": WEIGHT_GRID, "cf_neighbour_grid": CF_NEIGHBOUR_GRID, "cf_shrinkage_grid": CF_SHRINKAGE_GRID,
    }
    return rows, per_unit, details


# ============================================================ runner
PAIRWISE_COMPARISONS = [
    # (experiment, A, B, metric)
    ("A. Similar courses", CONTENT, TOP_RATED, f"NDCG@{K}"),
    ("A. Similar courses", CONTENT, RANDOM, f"NDCG@{K}"),
    ("B. Learner profiles", HYBRID_CV, CONTENT, GRADED),
    ("B. Learner profiles", HYBRID_CV, KNOWLEDGE, GRADED),
    ("B. Learner profiles", HYBRID_CV, HYBRID_HAND, GRADED),
    ("B. Learner profiles", HYBRID_CV, CONTENT, f"NDCG@{K}"),
    ("B. Learner profiles", HYBRID_CV, TOP_RATED, GRADED),
    ("B. Learner profiles", HYBRID_APP, HYBRID_CV, GRADED),
    ("B. Learner profiles", HYBRID_APP, HYBRID_CV, "Diversity (ILD)"),
    ("C. Synthetic interactions", "Hybrid (tuned on validation)", CF, f"NDCG@{K}"),
    ("C. Synthetic interactions", "Hybrid (tuned on validation)", POPULARITY, f"NDCG@{K}"),
    ("C. Synthetic interactions", CF, POPULARITY, f"NDCG@{K}"),
    ("C. Synthetic interactions", CF, "Content-based (history)", f"NDCG@{K}"),
]


def pairwise_table(per_unit_by_experiment: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for experiment, a, b, metric in PAIRWISE_COMPARISONS:
        per_unit = per_unit_by_experiment.get(experiment)
        if per_unit is None or per_unit.empty:
            continue
        wide = per_unit.pivot(index="unit", columns="method", values=metric)
        stats = paired_bootstrap(wide[a].to_numpy(), wide[b].to_numpy())
        rows.append({"experiment": experiment, "comparison": f"{a} - {b}", "metric": metric, **stats,
                     "ci_excludes_zero": bool(stats["ci_low"] > 0 or stats["ci_high"] < 0)})
    return pd.DataFrame(rows)


def run_all() -> pd.DataFrame:
    engine = SkillBridgeEngine.load()
    rows_a, per_a = experiment_similar_courses(engine)
    rows_b, per_b, _, details_b = experiment_profiles(engine)
    rows_c, per_c, details_c = experiment_synthetic(engine)

    results = pd.DataFrame(rows_a + rows_b + rows_c)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)
    pairwise = pairwise_table({"A. Similar courses": per_a, "B. Learner profiles": per_b,
                               "C. Synthetic interactions": per_c})
    pairwise.to_csv(PAIRWISE_PATH, index=False)
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "k": K, "random_seed": RANDOM_SEED, "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "catalogue_courses": int(len(engine.courses)), "rated_courses": int(len(engine.rated)),
        "experiment_A": {"query_courses": int(per_a["unit"].nunique())},
        "experiment_B": details_b,
        "experiment_C": details_c,
        "default_hybrid_weights": DEFAULT_WEIGHTS,
        "notes": "Offline proxy/synthetic evaluation only; no user-satisfaction evidence.",
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    import time
    started = time.time()
    table = run_all()
    pd.set_option("display.width", 200)
    print(table.pivot_table(index=["experiment", "method"], columns="metric", values="value").round(4).to_string())
    print(pd.read_csv(PAIRWISE_PATH).round(4).to_string())
    print(f"Finished in {time.time() - started:.0f}s")
