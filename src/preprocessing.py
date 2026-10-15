"""Clean the raw Coursera catalogue into an analysis-ready table.

The raw file is never modified. The cleaned table is written to
``data/processed/courses_clean.csv``.

Format facts discovered by inspecting the raw file (see README):
* ``Skills`` is a list of tags separated by two or more spaces. The final
  chunk additionally ends with Coursera's *domain* and *subdomain* slugs
  separated by single spaces, e.g. ``"... SQL information-technology data-management"``.
* ``Course Rating`` is numeric except for the literal ``"Not Calibrated"``.
* ``Difficulty Level`` uses Beginner / Intermediate / Advanced / Conversant /
  Not Calibrated.
* Some descriptions contain U+FFFD replacement characters where apostrophes
  were lost in the original scrape.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import PROCESSED_CATALOGUE_PATH, REPORTS_DIR, load_raw_catalogue

# Coursera's eleven top-level subject domains (taken from the slugs present in
# the dataset). Used only to recognise where the skill tags stop.
COURSERA_DOMAINS = {
    "arts-and-humanities", "business", "computer-science", "data-science",
    "health", "information-technology", "language-learning", "life-sciences",
    "math-and-logic", "personal-development", "physical-science-and-engineering",
    "social-sciences",
}

# Ordinal difficulty. "Conversant" is a Coursera label without a published
# definition; we treat it as equivalent to Intermediate for *ranking* (a manual
# decision, documented in the README) but keep the original label for display.
# "Not Calibrated" means the level is unknown and is stored as NaN.
DIFFICULTY_RANK = {"Beginner": 1, "Intermediate": 2, "Conversant": 2, "Advanced": 3}

RATING_MIN, RATING_MAX = 1.0, 5.0
SHORT_DESCRIPTION_CHARS = 30  # below this a description is flagged as uninformative
# A handful of raw rows have description text spilled into the Skills column;
# real skill tags are short, so longer "tags" are discarded as parsing debris.
MAX_SKILL_TAG_CHARS = 50

SLUG_PATTERN = re.compile(r"^[a-z]+(?:-[a-z]+)*$")
APOSTROPHE_PATTERN = re.compile(r"(?<=[A-Za-z])\ufffd(?=(?:s|t|ll|re|ve|d|m)\b)")
COURSE_SLUG_PATTERN = re.compile(r"coursera\.org/learn/([^/?#]+)")


def repair_text(text: object) -> str:
    """Normalise whitespace and repair lost apostrophes.

    A replacement character before a typical contraction or possessive ending
    (``You\ufffdll``, ``Coursera\ufffds``) was an apostrophe; any other
    replacement character becomes a space.
    """
    if text is None or (isinstance(text, float) and np.isnan(text)) or text is pd.NA:
        return ""
    value = str(text)
    value = APOSTROPHE_PATTERN.sub("'", value)
    value = value.replace("\ufffd", " ")
    return re.sub(r"\s+", " ", value).strip()


def parse_skills_field(raw_skills: object) -> tuple[list[str], str, str]:
    """Split the raw ``Skills`` string into (skill tags, domain, subdomain)."""
    if raw_skills is None or raw_skills is pd.NA or (isinstance(raw_skills, float) and np.isnan(raw_skills)):
        return [], "", ""
    chunks = [chunk.strip() for chunk in re.split(r"\s{2,}", repair_text_keep_spacing(raw_skills)) if chunk.strip()]
    if not chunks:
        return [], "", ""

    # The final chunk is "<last skill> <domain> <subdomain>".
    last_tokens = chunks[-1].split(" ")
    domain, subdomain = "", ""
    if len(last_tokens) >= 2 and SLUG_PATTERN.match(last_tokens[-1]) and last_tokens[-2] in COURSERA_DOMAINS:
        domain, subdomain = last_tokens[-2], last_tokens[-1]
        last_tokens = last_tokens[:-2]
    chunks[-1] = " ".join(last_tokens).strip()

    tags: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        tag = chunk.strip().lower()
        if tag and tag not in seen and len(tag) <= MAX_SKILL_TAG_CHARS:
            seen.add(tag)
            tags.append(tag)
    return tags, domain, subdomain


def repair_text_keep_spacing(text: object) -> str:
    """Like :func:`repair_text` but keeps runs of spaces (they delimit skill tags)."""
    value = APOSTROPHE_PATTERN.sub("'", str(text))
    return value.replace("\ufffd", " ").strip()


def parse_rating(raw_rating: object) -> float:
    """Convert a rating string to float; invalid or out-of-range values become NaN."""
    value = pd.to_numeric(raw_rating, errors="coerce")
    if pd.isna(value) or not (RATING_MIN <= float(value) <= RATING_MAX):
        return float("nan")
    return float(value)


def course_id_from_url(url: str, fallback_index: int) -> str:
    """Use the Coursera URL slug as a stable, human-readable course id."""
    match = COURSE_SLUG_PATTERN.search(url or "")
    return match.group(1).lower() if match else f"course-{fallback_index}"


def clean_catalogue(raw: pd.DataFrame) -> pd.DataFrame:
    """Return the cleaned catalogue (one row per unique course URL)."""
    frame = raw.copy()

    # 1. Exact duplicate rows, then duplicate URLs (the URL identifies a course).
    frame = frame.drop_duplicates()
    frame["Course URL"] = frame["Course URL"].fillna("").astype(str).str.strip()
    frame["Course Name"] = frame["Course Name"].map(repair_text)
    frame = frame[frame["Course Name"] != ""]
    has_url = frame["Course URL"] != ""
    frame = pd.concat([
        frame[has_url].drop_duplicates(subset="Course URL", keep="first"),
        frame[~has_url].drop_duplicates(subset=["Course Name", "University"], keep="first"),
    ]).reset_index(drop=True)

    parsed = frame["Skills"].map(parse_skills_field)
    cleaned = pd.DataFrame({
        "course_id": [course_id_from_url(url, i) for i, url in enumerate(frame["Course URL"])],
        "title": frame["Course Name"],
        "provider": frame["University"].map(repair_text).replace("", "Unknown provider"),
        "difficulty": frame["Difficulty Level"].map(repair_text).replace("", "Not Calibrated"),
        "rating": frame["Course Rating"].map(parse_rating),
        "url": frame["Course URL"],
        "description": frame["Course Description"].map(repair_text),
        "skills": parsed.map(lambda item: item[0]),
        "domain": parsed.map(lambda item: item[1]),
        "subdomain": parsed.map(lambda item: item[2]),
    })
    cleaned["difficulty_rank"] = cleaned["difficulty"].map(DIFFICULTY_RANK)  # NaN = unknown
    cleaned["has_rating"] = cleaned["rating"].notna()
    cleaned["url_valid"] = cleaned["url"].str.match(r"^https://www\.coursera\.org/learn/[^/\s]+/?$")
    cleaned["short_description"] = cleaned["description"].str.len() < SHORT_DESCRIPTION_CHARS

    # A course_id collision would only happen for malformed URLs; make ids unique.
    duplicated_ids = cleaned["course_id"].duplicated(keep=False)
    if duplicated_ids.any():
        cleaned.loc[duplicated_ids, "course_id"] = [
            f"{cid}-{i}" for i, cid in zip(cleaned.index[duplicated_ids], cleaned.loc[duplicated_ids, "course_id"])
        ]
    return cleaned


def save_processed(cleaned: pd.DataFrame, path: Path = PROCESSED_CATALOGUE_PATH) -> None:
    """Write the cleaned catalogue to CSV (skill lists joined with '|')."""
    path.parent.mkdir(parents=True, exist_ok=True)
    output = cleaned.copy()
    output["skills"] = output["skills"].map("|".join)
    output.to_csv(path, index=False)


def load_processed(path: Path = PROCESSED_CATALOGUE_PATH) -> pd.DataFrame:
    """Read the cleaned catalogue back, restoring list and numeric types."""
    frame = pd.read_csv(path, keep_default_na=False, na_values={"rating": [""], "difficulty_rank": [""]})
    frame["skills"] = frame["skills"].map(lambda value: [s for s in str(value).split("|") if s])
    for column in ("has_rating", "url_valid", "short_description"):
        frame[column] = frame[column].astype(str).str.lower() == "true"
    frame["rating"] = pd.to_numeric(frame["rating"], errors="coerce")
    frame["difficulty_rank"] = pd.to_numeric(frame["difficulty_rank"], errors="coerce")
    return frame


def profile_dataset(raw: pd.DataFrame, cleaned: pd.DataFrame) -> dict:
    """Compute the dataset-inspection statistics reported in the README and UI."""
    raw_ratings = pd.to_numeric(raw["Course Rating"], errors="coerce")
    return {
        "raw_rows": int(len(raw)),
        "raw_columns": list(raw.columns),
        "raw_missing_values": {column: int(raw[column].isna().sum()) for column in raw.columns},
        "raw_exact_duplicate_rows": int(raw.duplicated().sum()),
        "raw_duplicate_urls": int(raw["Course URL"].duplicated().sum()),
        "raw_non_numeric_ratings": raw.loc[raw_ratings.isna(), "Course Rating"].value_counts().to_dict(),
        "clean_rows": int(len(cleaned)),
        "duplicate_titles_with_different_urls": int(cleaned["title"].duplicated().sum()),
        "difficulty_distribution": cleaned["difficulty"].value_counts().to_dict(),
        "rating_summary": {
            key: round(float(value), 3)
            for key, value in cleaned["rating"].describe().items()
        },
        "rated_courses": int(cleaned["has_rating"].sum()),
        "providers": int(cleaned["provider"].nunique()),
        "top_providers": cleaned["provider"].value_counts().head(10).to_dict(),
        "domains": cleaned["domain"].replace("", "unknown").value_counts().to_dict(),
        "subdomains": int(cleaned.loc[cleaned["subdomain"] != "", "subdomain"].nunique()),
        "courses_without_subdomain": int((cleaned["subdomain"] == "").sum()),
        "courses_without_skill_tags": int((cleaned["skills"].map(len) == 0).sum()),
        "description_length_chars": {
            key: round(float(value), 1)
            for key, value in cleaned["description"].str.len().describe().items()
        },
        "short_descriptions": int(cleaned["short_description"].sum()),
        "descriptions_with_repaired_characters": int(raw["Course Description"].fillna("").str.contains("�").sum()),
        "skill_tags_per_course": {
            key: round(float(value), 2)
            for key, value in cleaned["skills"].map(len).describe().items()
        },
        "unique_skill_tags": int(len({tag for tags in cleaned["skills"] for tag in tags})),
        "valid_coursera_urls": int(cleaned["url_valid"].sum()),
    }


def build_processed_catalogue() -> tuple[pd.DataFrame, dict]:
    """Run the whole pipeline: load raw -> clean -> save -> profile."""
    raw = load_raw_catalogue()
    cleaned = clean_catalogue(raw)
    save_processed(cleaned)
    profile = profile_dataset(raw, cleaned)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "dataset_profile.json").write_text(json.dumps(profile, indent=2, default=str), encoding="utf-8")
    return cleaned, profile


if __name__ == "__main__":
    _, dataset_profile = build_processed_catalogue()
    print(json.dumps(dataset_profile, indent=2, default=str))
