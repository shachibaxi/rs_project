"""Content-based filtering with TF-IDF and cosine similarity.

Each course is represented by a TF-IDF vector built from its title, skill
tags, description and provider. Two modes are supported:

* Mode 1 - item to item: nearest neighbours of a selected course
  (scikit-learn ``NearestNeighbors`` with cosine distance).
* Mode 2 - profile to item: the learner's interests, goal and target skills
  are turned into a query document and compared with every course.

Coursera's domain/subdomain labels are deliberately NOT included in the text,
because the offline evaluation uses subdomains as a proxy relevance label.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.neighbors import NearestNeighbors

# Repeating a field gives its words more weight in the TF-IDF vector. Titles
# and skill tags are short, specific summaries of the course, while the long
# description contains much generic marketing text.
TITLE_REPEAT = 2
SKILLS_REPEAT = 2

TFIDF_SETTINGS = dict(
    stop_words="english",
    ngram_range=(1, 2),   # unigrams + bigrams, e.g. "machine learning"
    min_df=2,             # ignore terms seen in only one course (typos, names)
    max_df=0.5,           # ignore terms in over half the courses ("course", "learn")
    sublinear_tf=True,    # 1 + log(tf): dampens long descriptions repeating a word
)


def normalise_text(text: str) -> str:
    """Lower-case and keep letters, digits and a few symbols used in tech names (c++, c#, .net)."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#.\s-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_course_document(row: pd.Series) -> str:
    """Combine the text fields of one course into a single document."""
    skills = " ".join(row["skills"]) if isinstance(row["skills"], list) else ""
    parts = [row["title"]] * TITLE_REPEAT + [skills] * SKILLS_REPEAT + [row["description"], row["provider"]]
    return normalise_text(" ".join(part for part in parts if part))


class ContentRecommender:
    """TF-IDF course model shared by every component that needs text similarity."""

    def __init__(self, courses: pd.DataFrame):
        self.course_ids = courses["course_id"].to_numpy()
        self.position = {course_id: i for i, course_id in enumerate(self.course_ids)}
        documents = courses.apply(build_course_document, axis=1)
        self.vectorizer = TfidfVectorizer(**TFIDF_SETTINGS)
        self.matrix: sparse.csr_matrix = self.vectorizer.fit_transform(documents)
        self.terms = self.vectorizer.get_feature_names_out()
        self.neighbours = NearestNeighbors(metric="cosine", algorithm="brute").fit(self.matrix)

    # ---- Mode 1: similar courses -------------------------------------------------
    def similar_courses(self, course_id: str, n: int = 10) -> pd.DataFrame:
        """Return the ``n`` most similar courses, excluding the course itself."""
        if course_id not in self.position:
            raise KeyError(f"Unknown course id: {course_id}")
        # Ask for extra neighbours: the course itself (and exact text duplicates)
        # come back first and are removed below.
        k = min(n + 5, len(self.course_ids))
        distances, indices = self.neighbours.kneighbors(self.matrix[self.position[course_id]], n_neighbors=k)
        result = pd.DataFrame({
            "course_id": self.course_ids[indices[0]],
            "score": 1.0 - distances[0],          # cosine similarity = 1 - cosine distance
        })
        result = result[result["course_id"] != course_id].drop_duplicates("course_id").head(n)
        return result.reset_index(drop=True)

    # ---- Mode 2: profile to course -----------------------------------------------
    def query_vector(self, query_text: str) -> sparse.csr_matrix:
        return self.vectorizer.transform([normalise_text(query_text)])

    def score_query(self, query_text: str) -> pd.Series:
        """Cosine similarity of the query with every course (TF-IDF rows are L2-normalised,
        so the dot product equals cosine similarity)."""
        if not query_text.strip():
            return pd.Series(0.0, index=self.course_ids)
        scores = linear_kernel(self.query_vector(query_text), self.matrix).ravel()
        return pd.Series(scores, index=self.course_ids)

    def score_from_history(self, course_ids: list[str]) -> pd.Series:
        """Cosine similarity with the mean TF-IDF vector of courses the learner took."""
        rows = [self.position[c] for c in course_ids if c in self.position]
        if not rows:
            return pd.Series(0.0, index=self.course_ids)
        centroid = np.asarray(self.matrix[rows].mean(axis=0))
        norm = np.linalg.norm(centroid)
        if norm == 0:
            return pd.Series(0.0, index=self.course_ids)
        scores = self.matrix @ (centroid.ravel() / norm)
        return pd.Series(np.asarray(scores).ravel(), index=self.course_ids)

    # ---- Explanations ------------------------------------------------------------
    def shared_terms(self, query_text: str, course_id: str, top: int = 4) -> list[str]:
        """Terms contributing most to the query-course similarity."""
        if course_id not in self.position or not query_text.strip():
            return []
        contribution = self.query_vector(query_text).multiply(self.matrix[self.position[course_id]])
        contribution = contribution.tocoo()
        order = np.argsort(contribution.data)[::-1][:top]
        return [self.terms[contribution.col[i]] for i in order]

    def shared_terms_between(self, course_a: str, course_b: str, top: int = 4) -> list[str]:
        contribution = self.matrix[self.position[course_a]].multiply(self.matrix[self.position[course_b]]).tocoo()
        order = np.argsort(contribution.data)[::-1][:top]
        return [self.terms[contribution.col[i]] for i in order]
