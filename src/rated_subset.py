"""Build ``courses_rated.csv``: the clean catalogue restricted to genuine ratings.

Rating-based components (the top-rated baseline and the hybrid "quality"
signal) read this file so they never see a missing or "Not Calibrated" rating.
Every other component keeps using ``courses_clean.csv``.

Ratings are copied through unchanged; rows are only removed, never edited.
"""

from __future__ import annotations

import pandas as pd

from src.data_loader import PROCESSED_CATALOGUE_PATH, RATED_CATALOGUE_PATH, load_raw_catalogue


def observed_rating_range(raw: pd.DataFrame) -> tuple[float, float]:
    """Return (min, max) of the numeric ratings in the *raw* file.

    The range is read from the data rather than assumed. Computing it on the
    raw file (not on the subset being filtered) keeps the check independent.
    """
    numeric = pd.to_numeric(raw["Course Rating"], errors="coerce").dropna()
    if numeric.empty:
        raise ValueError("The raw dataset contains no numeric ratings.")
    return float(numeric.min()), float(numeric.max())


def build_rated_subset(
    clean_path=PROCESSED_CATALOGUE_PATH,
    output_path=RATED_CATALOGUE_PATH,
) -> dict:
    """Filter the clean catalogue to rows with a valid rating and save it.

    A rating is valid when it parses as a finite number and lies inside the
    range observed in the raw data. Missing values, text such as
    "Not Calibrated" and anything outside the range are removed.
    """
    # Read the rating as text first so we can see exactly what is in the file.
    clean = pd.read_csv(clean_path, dtype={"rating": str}, keep_default_na=False)
    rating_text = clean["rating"].str.strip()
    rating_value = pd.to_numeric(rating_text, errors="coerce")

    low, high = observed_rating_range(load_raw_catalogue())
    # between() is False for NaN and for +/-inf, so this also rejects non-finite values.
    is_valid = rating_value.between(low, high)

    rated = clean[is_valid].copy()
    rated["rating"] = rating_value[is_valid].astype(float)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rated.to_csv(output_path, index=False)

    removed_values = rating_text[~is_valid].replace("", "<empty>").value_counts().to_dict()
    return {
        "rows_before": int(len(clean)),
        "rows_removed": int((~is_valid).sum()),
        "removed_values": removed_values,
        "rows_after": int(len(rated)),
        "legitimate_range_observed_in_raw": (low, high),
    }


def verify_rated_subset(output_path=RATED_CATALOGUE_PATH) -> dict:
    """Reload the saved file and check type, completeness and range."""
    rated = pd.read_csv(output_path)
    low, high = observed_rating_range(load_raw_catalogue())
    return {
        "loaded_rows": int(len(rated)),
        "rating_dtype": str(rated["rating"].dtype),
        "missing_or_invalid_remaining": int(
            rated["rating"].isna().sum() + (~rated["rating"].between(low, high)).sum()
        ),
        "min_rating": float(rated["rating"].min()),
        "max_rating": float(rated["rating"].max()),
        "duplicate_course_ids": int(rated["course_id"].duplicated().sum()),
    }


if __name__ == "__main__":
    print(build_rated_subset())
    print(verify_rated_subset())
