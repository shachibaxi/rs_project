"""Locate and load the raw Coursera catalogue CSV.

All paths are built relative to the project root so the project works on any
machine without editing absolute paths.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"
CONFIG_DIR = PROJECT_ROOT / "config"
REPORTS_DIR = PROJECT_ROOT / "reports"

KNOWLEDGE_CONFIG_PATH = CONFIG_DIR / "skill_knowledge.json"
PROCESSED_CATALOGUE_PATH = PROCESSED_DIR / "courses_clean.csv"
# Subset of the clean catalogue with a genuine numeric rating. Used only where
# a rating is required (the top-rated baseline and the hybrid quality signal).
RATED_CATALOGUE_PATH = PROCESSED_DIR / "courses_rated.csv"
SYNTHETIC_INTERACTIONS_PATH = SAMPLE_DIR / "synthetic_interactions.csv"

DATASET_URL = "https://www.kaggle.com/datasets/khusheekapoor/coursera-courses-dataset-2021"

# Columns observed in the Kaggle file (inspected, not assumed). Only the first
# two are strictly required: without a title and a link a course cannot be
# shown. Everything else is optional and handled when missing.
REQUIRED_COLUMNS = ["Course Name", "Course URL"]
OPTIONAL_COLUMNS = [
    "University",
    "Difficulty Level",
    "Course Rating",
    "Course Description",
    "Skills",
]


class DatasetError(Exception):
    """Raised when the catalogue cannot be found or is unusable."""


def find_raw_csv(raw_dir: Path = RAW_DIR) -> Path:
    """Return the catalogue CSV inside ``data/raw``.

    The Kaggle download is named ``Coursera.csv``; if it was renamed we fall
    back to the only CSV in the folder.
    """
    preferred = raw_dir / "Coursera.csv"
    if preferred.exists():
        return preferred
    csv_files = sorted(raw_dir.glob("*.csv"))
    if len(csv_files) == 1:
        return csv_files[0]
    if not csv_files:
        raise DatasetError(
            f"No CSV found in {raw_dir}. Download the dataset from {DATASET_URL}, "
            "unzip it and place Coursera.csv in data/raw/."
        )
    names = ", ".join(p.name for p in csv_files)
    raise DatasetError(f"Several CSV files in {raw_dir} ({names}); keep only Coursera.csv.")


def load_raw_catalogue(csv_path: Path | None = None) -> pd.DataFrame:
    """Read the raw catalogue and validate its columns.

    The file contains some bytes that are not valid UTF-8 (the original
    scrape lost curly apostrophes), so undecodable bytes are replaced with
    U+FFFD instead of crashing; preprocessing then repairs them.
    """
    path = Path(csv_path) if csv_path is not None else find_raw_csv()
    if not path.exists():
        raise DatasetError(f"Dataset file not found: {path}")

    try:
        frame = pd.read_csv(path, dtype=str, encoding="utf-8", encoding_errors="replace")
    except pd.errors.EmptyDataError as error:
        raise DatasetError(f"Dataset file is empty: {path}") from error

    if frame.empty:
        raise DatasetError(f"Dataset file has a header but no course rows: {path}")

    frame.columns = [column.strip() for column in frame.columns]
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise DatasetError(
            f"Dataset is missing required column(s) {missing}. Found columns: {list(frame.columns)}"
        )

    # Optional columns that are absent are added as empty so downstream code
    # can rely on them existing; preprocessing treats empty as "unknown".
    for column in OPTIONAL_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame


def file_fingerprint(path: Path) -> str:
    """Cheap change detector (modification time + size) used for cache keys."""
    stat = Path(path).stat()
    return f"{stat.st_mtime_ns}-{stat.st_size}"
