"""Data loading, validation, cleaning and the rated subset."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_loader import (PROJECT_ROOT, RAW_DIR, REQUIRED_COLUMNS, DatasetError, find_raw_csv,
                             load_raw_catalogue)
from src.preprocessing import (clean_catalogue, load_processed, parse_rating, parse_skills_field, repair_text)
from src.rated_subset import build_rated_subset, verify_rated_subset

HEADER = "Course Name,University,Difficulty Level,Course Rating,Course URL,Course Description,Skills\n"


def write_csv(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def test_real_csv_loads_with_expected_columns():
    raw = load_raw_catalogue()
    assert len(raw) > 3000
    for column in REQUIRED_COLUMNS:
        assert column in raw.columns


def test_paths_are_relative_to_project_root():
    assert RAW_DIR.is_relative_to(PROJECT_ROOT)
    assert find_raw_csv().parent == RAW_DIR


def test_missing_required_column_raises(tmp_path):
    path = write_csv(tmp_path / "bad.csv", "Course Name,University\nA,B\n")
    with pytest.raises(DatasetError, match="Course URL"):
        load_raw_catalogue(path)


def test_optional_columns_may_be_missing(tmp_path):
    path = write_csv(tmp_path / "min.csv", "Course Name,Course URL\nIntro,https://www.coursera.org/learn/intro\n")
    raw = load_raw_catalogue(path)
    cleaned = clean_catalogue(raw)
    assert cleaned.loc[0, "provider"] == "Unknown provider"
    assert cleaned.loc[0, "difficulty"] == "Not Calibrated"
    assert pd.isna(cleaned.loc[0, "rating"]) and cleaned.loc[0, "skills"] == []


def test_empty_file_and_header_only_raise(tmp_path):
    with pytest.raises(DatasetError):
        load_raw_catalogue(write_csv(tmp_path / "empty.csv", ""))
    with pytest.raises(DatasetError, match="no course rows"):
        load_raw_catalogue(write_csv(tmp_path / "header.csv", HEADER))


def test_missing_dataset_folder_gives_download_instructions(tmp_path):
    with pytest.raises(DatasetError, match="kaggle"):
        find_raw_csv(tmp_path)


def test_duplicates_removed_and_invalid_ratings_become_nan(tmp_path):
    row = 'Intro,Uni,Beginner,{rating},https://www.coursera.org/learn/{slug},"Desc",python  sql data-science data-analysis\n'
    text = HEADER + row.format(rating="4.5", slug="a") * 2 + row.format(rating="Not Calibrated", slug="b") \
        + row.format(rating="7.2", slug="c") + row.format(rating="abc", slug="d")
    cleaned = clean_catalogue(load_raw_catalogue(write_csv(tmp_path / "d.csv", text)))
    assert len(cleaned) == 4                       # exact duplicate row removed
    assert cleaned["course_id"].is_unique
    assert cleaned.set_index("course_id")["rating"].isna().to_dict() == {"a": False, "b": True, "c": True, "d": True}


def test_parse_rating_and_skills_and_text_repair():
    assert parse_rating("4.7") == 4.7
    assert pd.isna(parse_rating("Not Calibrated")) and pd.isna(parse_rating("0.5"))
    tags, domain, subdomain = parse_skills_field("Strategy  Innovation business business-strategy")
    assert tags == ["strategy", "innovation"] and domain == "business" and subdomain == "business-strategy"
    assert parse_skills_field(None) == ([], "", "")
    assert repair_text("You�ll learn") == "You'll learn"
    assert repair_text("in�SQL") == "in SQL"


def test_processed_catalogue_is_clean():
    courses = load_processed()
    assert courses["course_id"].is_unique
    assert courses["url"].is_unique
    rated = courses["rating"].dropna()
    assert rated.between(1, 5).all()


def test_rated_subset_has_only_genuine_ratings():
    report = verify_rated_subset()
    assert report["missing_or_invalid_remaining"] == 0
    assert report["rating_dtype"] == "float64"
    rated = pd.read_csv(PROJECT_ROOT / "data/processed/courses_rated.csv")
    raw = pd.read_csv(find_raw_csv(), dtype=str).drop_duplicates(subset="Course URL")
    merged = rated.merge(raw, left_on="url", right_on="Course URL")
    assert len(merged) == len(rated)
    assert (merged["rating"] == pd.to_numeric(merged["Course Rating"])).all()   # ratings copied unchanged


def test_rated_subset_filters_a_messy_file(tmp_path):
    clean = pd.DataFrame({"course_id": ["a", "b", "c", "d"], "rating": ["4.5", "", "Not Calibrated", "9"]})
    clean.to_csv(tmp_path / "clean.csv", index=False)
    summary = build_rated_subset(tmp_path / "clean.csv", tmp_path / "rated.csv")
    assert summary["rows_before"] == 4 and summary["rows_after"] == 1 and summary["rows_removed"] == 3
    assert pd.read_csv(tmp_path / "rated.csv")["course_id"].tolist() == ["a"]
