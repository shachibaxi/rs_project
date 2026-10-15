"""Weighted hybrid recommendation with diversity re-ranking.

    hybrid(c) = sum_m  w_m * s_m(c)  /  sum_m w_m        over components m AVAILABLE for c

Components (each normalised to [0, 1] before combining):

* content       - TF-IDF cosine similarity to the learner's profile text,
                  divided by the largest value among eligible courses.
* knowledge     - rule score from ``knowledge_based`` (already in [0, 1]).
* quality       - (rating - 1) / 4, from ``courses_rated.csv`` only.
* collaborative - item-kNN score (SYNTHETIC training data), divided by the
                  largest value among eligible courses.

A component is dropped entirely when it cannot be computed for this learner
(e.g. no history -> no collaborative score; no goal or desired skills -> no
knowledge score). For an individual course without a genuine rating the
quality weight is dropped for that course only; no rating is invented.
The remaining weights are re-normalised so scores stay comparable in [0, 1].

Diversity (Maximal Marginal Relevance). The catalogue contains many near-
identical courses (e.g. several "Exploratory Data Analysis" guided projects),
so the final list is chosen greedily from the best candidates:

    next = argmax_c  [ lambda * hybrid(c) - (1 - lambda) * max_{s in selected} cos(c, s) ]

lambda = 1 is plain ranking; smaller values trade a little relevance for less
redundancy.

Hard constraints are applied BEFORE scoring, by restricting the candidates,
so the hybrid can never return a course that violates them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

# Content / knowledge / quality weights were selected by career-split
# cross-validation in src/evaluation.py (experiment B: graded topic-and-level
# relevance, scored only on careers not used for the selection; see
# reports/evaluation_metadata.json). A unit test checks these constants match
# the saved selection. The collaborative weight is not part of that search
# (it needs a learning history) and stays modest because its data are synthetic.
DEFAULT_WEIGHTS = {"content": 0.1, "knowledge": 0.7, "quality": 0.1, "collaborative": 0.15}
# Diversification is a product choice, not a relevance gain: in experiment B it
# lowers graded NDCG@10 slightly and raises intra-list diversity (both measured
# and reported), and it removes near-duplicate courses from the list.
DEFAULT_DIVERSITY = 0.85
MMR_CANDIDATES = 50          # re-rank only among the best 50 candidates


def max_normalise(scores: pd.Series) -> pd.Series:
    """Divide by the maximum so the best eligible course scores 1 (0 stays 0)."""
    peak = scores.max()
    if not np.isfinite(peak) or peak <= 0:
        return scores * 0.0
    return scores / peak


def combine_scores(components: dict[str, pd.Series], weights: dict[str, float]) -> pd.Series:
    """Weighted mean of the available components, per course.

    ``components`` maps name -> Series indexed by course_id; NaN marks a value
    that does not exist for that course (an unrated course's quality).
    """
    if not components:
        raise ValueError("No recommendation component is available for this profile.")
    frame = pd.DataFrame(components)
    weight_row = pd.Series({name: weights[name] for name in frame.columns})
    present = frame.notna()
    weighted_sum = (frame.fillna(0.0) * weight_row).sum(axis=1)
    weight_total = (present * weight_row).sum(axis=1)
    return weighted_sum / weight_total.replace(0, np.nan)


def hybrid_scores(components: dict[str, pd.Series], weights: dict[str, float]) -> pd.Series:
    """Combined score for courses with at least one personalised signal.

    Quality alone must not push an unrelated course into the list, so courses
    whose content, knowledge and collaborative scores are all 0 are dropped.
    """
    scores = combine_scores(components, weights)
    personal = [name for name in components if name != "quality"]
    has_signal = pd.DataFrame({name: components[name] for name in personal}).fillna(0).max(axis=1) > 0
    return scores[has_signal].dropna()


def mmr_order(scores: pd.Series, n: int, vectors: sparse.csr_matrix, position: dict[str, int],
              diversity: float = DEFAULT_DIVERSITY) -> list[str]:
    """Course ids in Maximal-Marginal-Relevance order (see module docstring)."""
    ranked = scores.sort_values(ascending=False, kind="mergesort")
    if diversity >= 1 or len(ranked) <= 1:
        return list(ranked.index[:n])
    candidates = list(ranked.index[:max(MMR_CANDIDATES, n)])
    relevance = ranked.loc[candidates].to_numpy()
    rows = vectors[[position[c] for c in candidates]]
    similarity = (rows @ rows.T).toarray()          # TF-IDF rows are L2-normalised -> cosine

    chosen: list[int] = [0]                          # the most relevant course always comes first
    max_similarity = similarity[0].copy()
    while len(chosen) < min(n, len(candidates)):
        value = diversity * relevance - (1 - diversity) * max_similarity
        value[chosen] = -np.inf
        best = int(np.argmax(value))
        chosen.append(best)
        max_similarity = np.maximum(max_similarity, similarity[best])
    return [candidates[i] for i in chosen]
