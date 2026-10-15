"""Recommendation methods, constraints, hybrid scoring, skill gap and learning path."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.hybrid import DEFAULT_WEIGHTS, combine_scores
from src.knowledge_base import ADDRESSES_THRESHOLD, KnowledgeConfigError, SkillKnowledge, load_knowledge
from src.knowledge_based import compute_targets
from src.learning_path import generate_learning_path, in_path_dependencies
from src.profile import PROFICIENCY_RANK, LearnerProfile
from src.recommender import COLLABORATIVE, CONTENT, HYBRID, KNOWLEDGE, METHODS, TOP_RATED
from src.skill_gap import analyse_skill_gap


def data_scientist(**overrides) -> LearnerProfile:
    base = dict(career_goal="Data Scientist", current_skills=["Python Programming"], proficiency="Beginner",
                interests="data analysis machine learning", completed_course_ids=["python-data"])
    base.update(overrides)
    return LearnerProfile(**base)


# ------------------------------------------------------------- every method
@pytest.mark.parametrize("method", METHODS)
def test_each_method_returns_valid_unique_results(engine, method):
    # diversity=1.0 switches MMR off so every method must be sorted by its score.
    result = engine.recommend(data_scientist(), method, 10, diversity=1.0)
    table = result.table
    assert len(table) == 10, result.notes
    assert table["course_id"].is_unique
    assert table["rank"].tolist() == list(range(1, 11))
    assert table["score"].between(0, 1).all()
    assert (table["score"].diff().dropna() <= 1e-12).all()        # sorted best first
    assert table["url"].str.startswith("https://www.coursera.org/learn/").all()
    assert table["explanation"].map(len).gt(0).all()
    assert "python-data" not in set(table["course_id"])          # completed course never returned


def test_top_rated_uses_only_genuinely_rated_courses(engine):
    table = engine.recommend(LearnerProfile(), TOP_RATED, 20).table
    assert table["rating"].notna().all()
    assert set(table["course_id"]) <= engine.rated_ids


def test_similar_courses_exclude_the_selected_course(engine):
    course_id = engine.courses["course_id"].iloc[0]
    table = engine.similar_courses(course_id, 10)
    assert course_id not in set(table["course_id"])
    assert len(table) == 10 and table["course_id"].is_unique


# -------------------------------------------------------------- constraints
def test_hard_constraints_are_respected(engine):
    profile = data_scientist(preferred_difficulties=["Beginner"], strict_difficulty=True,
                             min_rating=4.6, strict_min_rating=True, excluded_topics=["excel"],
                             must_cover_skills=["Machine Learning"], strict_must_cover=True)
    for method in (HYBRID, KNOWLEDGE, CONTENT, TOP_RATED):
        table = engine.recommend(profile, method, 15).table
        assert (table["difficulty"] == "Beginner").all(), method
        assert (table["rating"] >= 4.6).all(), method
        assert not table["title"].str.lower().str.contains("excel").any(), method
        addresses = engine.skill_matrix.loc[table["course_id"], "Machine Learning"] >= ADDRESSES_THRESHOLD
        assert addresses.all(), method


def test_strict_provider_and_soft_provider(engine):
    strict = engine.recommend(data_scientist(preferred_providers=["IBM"], strict_provider=True), HYBRID, 10).table
    assert (strict["provider"] == "IBM").all()
    soft = engine.recommend(data_scientist(preferred_providers=["IBM"]), HYBRID, 10).table
    assert len(soft) == 10                                       # soft never removes courses


def test_insufficient_eligible_courses_are_explained(engine):
    profile = LearnerProfile(desired_skills=["MLOps & Model Deployment"], must_cover_skills=["MLOps & Model Deployment"],
                             preferred_difficulties=["Beginner"], strict_difficulty=True)
    result = engine.recommend(profile, HYBRID, 10)
    assert len(result.table) < 10
    assert any("most restrictive" in note for note in result.notes)
    assert result.constraint_report.most_restrictive() in {"Must-cover skills", "Difficulty"}


# --------------------------------------------------------- invalid inputs
def test_invalid_inputs_are_handled(engine):
    profile = LearnerProfile(career_goal="Astronaut", proficiency="Guru", desired_skills=["Telepathy"],
                             interests="astronomy", min_rating=9)
    result = engine.recommend(profile, HYBRID, 5)
    assert len(result.table) == 5
    assert any("Unknown career goal" in n for n in result.notes)
    assert profile.proficiency == "Beginner" and profile.min_rating is None
    with pytest.raises(ValueError):
        engine.recommend(LearnerProfile(), "Magic", 5)
    with pytest.raises(ValueError):
        engine.recommend(data_scientist(), HYBRID, 0)


def test_empty_profile_returns_message_not_crash(engine):
    result = engine.recommend(LearnerProfile(), HYBRID, 5)
    assert result.table.empty and result.notes


def test_collaborative_cold_start(engine):
    result = engine.recommend(data_scientist(completed_course_ids=[]), COLLABORATIVE, 5)
    assert result.table.empty
    assert any("cold start" in n for n in result.notes)


# ----------------------------------------------------------- personalisation
def test_proficiency_changes_the_ranking(engine):
    beginner = engine.recommend(data_scientist(proficiency="Beginner"), KNOWLEDGE, 10).table
    advanced = engine.recommend(data_scientist(proficiency="Advanced"), KNOWLEDGE, 10).table
    assert set(beginner["course_id"]) != set(advanced["course_id"])
    assert (beginner["difficulty"] == "Beginner").mean() > (advanced["difficulty"] == "Beginner").mean()


def test_career_goal_changes_the_ranking(engine):
    ds = engine.recommend(data_scientist(), HYBRID, 10).table
    web = engine.recommend(data_scientist(career_goal="Web Developer", interests=""), HYBRID, 10).table
    assert len(set(ds["course_id"]) & set(web["course_id"])) < 3


# --------------------------------------------------------------- hybrid
def test_combine_scores_weighted_mean_with_missing_component():
    components = {"content": pd.Series({"a": 1.0, "b": 0.5}), "quality": pd.Series({"a": 0.5, "b": np.nan})}
    scores = combine_scores(components, {"content": 0.6, "quality": 0.4})
    assert scores["a"] == pytest.approx(0.6 * 1.0 + 0.4 * 0.5)
    assert scores["b"] == pytest.approx(0.5)                 # unrated: quality weight dropped, not imputed


def test_engine_hybrid_score_matches_its_components(engine):
    table = engine.recommend(data_scientist(completed_course_ids=["python-data"]), HYBRID, 10).table
    components = [c for c in ("content", "knowledge", "quality", "collaborative") if f"{c}_score" in table]
    for _, row in table.iterrows():
        present = [c for c in components if pd.notna(row[f"{c}_score"])]
        expected = sum(DEFAULT_WEIGHTS[c] * row[f"{c}_score"] for c in present) / sum(DEFAULT_WEIGHTS[c] for c in present)
        assert row["score"] == pytest.approx(expected)


def test_missing_optional_metadata_does_not_crash(tiny_engine):
    profile = LearnerProfile(career_goal="ML Person", interests="python")
    for method in (HYBRID, KNOWLEDGE, CONTENT, TOP_RATED):
        result = tiny_engine.recommend(profile, method, 10)
        assert result.table["course_id"].is_unique
    assert tiny_engine.similar_courses("stats-1", 3)["course_id"].ne("stats-1").all()


# -------------------------------------------------------------- skill gap
def test_targets_compare_existing_with_required_skills():
    knowledge = load_knowledge()
    profile = LearnerProfile(career_goal="Data Analyst", current_skills=["SQL & Databases", "Statistics"],
                             desired_skills=["Machine Learning"])
    targets = compute_targets(profile, knowledge)
    required = set(knowledge.career_skills("Data Analyst"))
    assert set(targets.possessed) == {"SQL & Databases", "Statistics"}
    assert set(targets.gaps) == (required - {"SQL & Databases", "Statistics"}) | {"Machine Learning"}
    assert targets.gaps["Machine Learning"] > targets.gaps["Data Analysis"]   # desired outranks career gap


def test_skill_gap_recommendations_teach_the_gap(engine):
    report = analyse_skill_gap(engine, data_scientist(), courses_per_gap=3)
    assert {g.skill for g in report.gaps} == set(report.target_skills) - set(report.possessed)
    assert 0 < report.readiness_percent < 100
    for gap in report.gaps:
        for course_id in gap.recommendations["course_id"]:
            assert engine.skill_matrix.at[course_id, gap.skill] >= ADDRESSES_THRESHOLD


# ---------------------------------------------------------- learning path
@pytest.mark.parametrize("career", ["Machine Learning Engineer", "Web Developer", "Cybersecurity Analyst"])
def test_learning_path_order_and_uniqueness(engine, career):
    profile = LearnerProfile(career_goal=career, proficiency="Beginner")
    path = generate_learning_path(engine, profile)
    assert len(path.course_ids) == len(set(path.course_ids))              # no repeated course
    position = {step.skill: step.position for step in path.steps}
    for step in path.steps:
        for dependency in in_path_dependencies(step.skill, set(position), engine.knowledge):
            assert position[dependency] < step.position                  # prerequisites come first
    assert [s.stage for s in path.steps] == sorted(s.stage for s in path.steps)


def test_learning_path_adds_missing_prerequisites(engine):
    path = generate_learning_path(engine, LearnerProfile(desired_skills=["Computer Vision"]))
    skills = [s.skill for s in path.steps]
    assert skills[-1] == "Computer Vision"
    assert {"Deep Learning", "Machine Learning"} <= set(skills)


def test_learning_path_without_prerequisites_or_courses(tiny_engine):
    path = generate_learning_path(tiny_engine, LearnerProfile(career_goal="ML Person"))
    by_skill = {s.skill: s for s in path.steps}
    assert by_skill["Quantum Basket Weaving"].course is None             # no course teaches it: no crash
    assert any("Quantum Basket Weaving" in n for n in path.notes)


# ------------------------------------------------------- knowledge config
def test_knowledge_config_validation(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"skills": {"A": {"keywords": ["a"], "prerequisites": [["Missing"]]}}, "careers": {"X": {"required_skills": ["A"]}}}')
    with pytest.raises(KnowledgeConfigError):
        load_knowledge(bad)
    cyclic = SkillKnowledge(skills={"A": {"keywords": ["a"], "prerequisites": [["B"]]},
                                    "B": {"keywords": ["b"], "prerequisites": [["A"]]}}, careers={})
    with pytest.raises(KnowledgeConfigError):
        cyclic.prerequisite_depth("A")


def test_every_career_skill_exists_in_catalogue(engine):
    supply = (engine.skill_matrix >= ADDRESSES_THRESHOLD).sum()
    assert (supply > 0).all(), supply[supply == 0]
    assert set(PROFICIENCY_RANK) == {"Beginner", "Intermediate", "Advanced"}


def test_website_ranking_can_exclude_synthetic_collaborative_signal(engine):
    profile = data_scientist(completed_course_ids=["python-data"])
    with_cf = engine.recommend(profile, HYBRID, 10)
    without_cf = engine.recommend(profile, HYBRID, 10, use_collaborative=False)
    assert "collaborative" in with_cf.components_used
    assert "collaborative" not in without_cf.components_used
    assert "python-data" not in set(without_cf.table["course_id"])     # completed course still excluded


def test_mmr_diversification_reduces_redundancy(engine):
    from src.evaluation import intra_list_diversity

    plain = engine.recommend(data_scientist(), HYBRID, 10, use_collaborative=False, diversity=1.0).table
    diverse = engine.recommend(data_scientist(), HYBRID, 10, use_collaborative=False, diversity=0.7).table
    assert diverse["course_id"].iloc[0] == plain["course_id"].iloc[0]          # best course always first
    ild = lambda t: intra_list_diversity(t["course_id"].tolist(), engine.content.matrix, engine.content.position)
    assert ild(diverse) > ild(plain)
    assert diverse["course_id"].is_unique


def test_mmr_order_on_toy_vectors():
    import numpy as np
    from scipy import sparse
    from src.hybrid import mmr_order

    # a and b are identical texts, c is different; b is slightly more relevant than c.
    vectors = sparse.csr_matrix(np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
    position = {"a": 0, "b": 1, "c": 2}
    scores = pd.Series({"a": 1.0, "b": 0.9, "c": 0.8})
    assert mmr_order(scores, 3, vectors, position, diversity=1.0) == ["a", "b", "c"]
    assert mmr_order(scores, 3, vectors, position, diversity=0.7) == ["a", "c", "b"]
