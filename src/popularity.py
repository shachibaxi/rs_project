"""Top-rated baseline (non-personalised).

The dataset has an average rating per course but NO number of ratings and NO
enrolment counts, so a true popularity baseline (or an IMDb-style weighted
rating, which needs the vote count) cannot be built honestly. This baseline
simply ranks courses by their average rating and is labelled "Top-rated",
not "Popular".

It reads ``courses_rated.csv`` only: courses without a genuine rating are not
ranked, never given an invented rating.
"""

from __future__ import annotations

import pandas as pd

# Ratings in the dataset lie on Coursera's 1-5 star scale (min 1.0 and max 5.0
# were verified in the raw file by src/rated_subset.py).
RATING_SCALE_MIN, RATING_SCALE_MAX = 1.0, 5.0


def quality_score(ratings: pd.Series) -> pd.Series:
    """Map ratings onto [0, 1] using the fixed rating scale (NaN stays NaN)."""
    return (ratings - RATING_SCALE_MIN) / (RATING_SCALE_MAX - RATING_SCALE_MIN)


def top_rated(rated_courses: pd.DataFrame, eligible_ids: set[str] | None = None, n: int = 10) -> pd.DataFrame:
    """Return the ``n`` highest-rated courses, optionally restricted to ``eligible_ids``.

    Ties (many courses share e.g. 4.8) are broken alphabetically by course id
    so the output is deterministic; the order inside a tie carries no meaning.
    """
    candidates = rated_courses
    if eligible_ids is not None:
        candidates = candidates[candidates["course_id"].isin(eligible_ids)]
    ranked = candidates.sort_values(["rating", "course_id"], ascending=[False, True]).head(n).copy()
    ranked["score"] = quality_score(ranked["rating"])
    return ranked
