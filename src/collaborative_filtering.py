"""Item-based collaborative filtering (item-kNN) on implicit feedback.

Trained ONLY on the synthetic interaction dataset (see ``synthetic_data.py``),
so its output demonstrates the algorithm, not real learner behaviour.

Algorithm
---------
* R is the binary learner x course matrix (1 = learner completed the course).
* Item similarity is cosine similarity between course columns:
      sim(i, j) = |U_i ∩ U_j| / sqrt(|U_i| * |U_j|)
  where U_i is the set of learners who took course i.
* Similarities resting on very few shared learners are unreliable (two
  courses each taken once by the same learner get similarity 1.0), so they
  are shrunk towards 0 by the number of co-occurrences c_ij:
      sim'(i, j) = sim(i, j) * c_ij / (c_ij + lambda)
* Only the k most similar neighbours of each course are kept (k-NN), which
  removes weak, noisy similarities.
* The score of a candidate course i for learner u is the sum of its
  similarities to the courses u has taken:
      score(u, i) = sum_{j in history(u)} sim_k(i, j)

Cold start: a learner with no known history gets no CF scores (returned as
``None``); a course nobody took has no neighbours and scores 0, flagged as
such. Other methods cover both cases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

# Chosen by grid search on a validation split of the synthetic TRAINING data
# (src/evaluation.py, results in reports/evaluation_metadata.json). Large
# neighbourhoods and strong shrinkage work best because the synthetic matrix is
# very sparse (about 4.7 learners per course).
DEFAULT_NEIGHBOURS = 400
DEFAULT_SHRINKAGE = 50.0  # lambda: a similarity backed by 50 shared learners keeps half its value


class ItemKNNRecommender:
    def __init__(self, interactions: pd.DataFrame, catalogue_ids: list[str], n_neighbours: int = DEFAULT_NEIGHBOURS,
                 shrinkage: float = DEFAULT_SHRINKAGE):
        self.catalogue_ids = np.asarray(catalogue_ids)
        self.position = {course_id: i for i, course_id in enumerate(self.catalogue_ids)}
        self.n_neighbours = n_neighbours
        self.shrinkage = shrinkage

        known = interactions[interactions["course_id"].isin(self.position)]
        known = known.drop_duplicates(["learner_id", "course_id"])   # repeated completions count once
        learners = known["learner_id"].unique()
        learner_position = {learner: i for i, learner in enumerate(learners)}
        rows = known["learner_id"].map(learner_position).to_numpy()
        cols = known["course_id"].map(self.position).to_numpy()
        self.matrix = sparse.csr_matrix(
            (np.ones(len(known), dtype=np.float32), (rows, cols)),
            shape=(len(learners), len(self.catalogue_ids)),
        )
        self.n_learners = len(learners)
        self.item_counts = np.asarray(self.matrix.sum(axis=0)).ravel()
        self.similarity = self._top_k_item_similarity()

    def _top_k_item_similarity(self) -> sparse.csr_matrix:
        item_user = self.matrix.T.tocsr()
        co_occurrence = (item_user @ item_user.T).toarray()      # c_ij = shared learners
        norms = np.sqrt(self.item_counts)
        norms[norms == 0] = 1.0
        cosine = co_occurrence / np.outer(norms, norms)
        denominator = co_occurrence + self.shrinkage
        full = np.divide(cosine * co_occurrence, denominator,
                         out=np.zeros_like(cosine), where=denominator > 0)
        np.fill_diagonal(full, 0.0)                     # a course is not its own neighbour
        k = min(self.n_neighbours, full.shape[1] - 1)
        if k <= 0:
            return sparse.csr_matrix(full.shape, dtype=np.float32)
        # Keep the k largest similarities in every row.
        cutoff_idx = np.argpartition(-full, k - 1, axis=1)[:, :k]
        pruned = np.zeros_like(full)
        rows = np.arange(full.shape[0])[:, None]
        pruned[rows, cutoff_idx] = full[rows, cutoff_idx]
        return sparse.csr_matrix(pruned)

    def has_interactions(self, course_id: str) -> bool:
        return course_id in self.position and self.item_counts[self.position[course_id]] > 0

    def score(self, history: list[str]) -> pd.Series | None:
        """CF scores for every catalogue course, or ``None`` for a cold-start learner."""
        rows = [self.position[c] for c in history if self.has_interactions(c)]
        if not rows:
            return None
        history_vector = np.zeros(len(self.catalogue_ids), dtype=np.float32)
        history_vector[rows] = 1.0
        # sim is the k-NN list of each candidate i; summing row i over the
        # history gives score(u, i).
        scores = self.similarity @ history_vector
        scores[rows] = 0.0                               # never re-recommend taken courses
        return pd.Series(np.asarray(scores).ravel(), index=self.catalogue_ids)

    def neighbours_in_history(self, course_id: str, history: list[str], top: int = 2) -> list[str]:
        """Courses from the history that most strongly support recommending ``course_id``."""
        if course_id not in self.position:
            return []
        row = self.similarity.getrow(self.position[course_id]).toarray().ravel()
        supporting = [(row[self.position[c]], c) for c in history if c in self.position and row[self.position[c]] > 0]
        return [c for _, c in sorted(supporting, reverse=True)[:top]]
