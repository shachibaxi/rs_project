"""Evaluation metrics, splits, synthetic-data labelling and saved results."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src.evaluation import (RESULTS_PATH, catalogue_coverage, experiment_similar_courses, hit_at_k,
                            intra_list_diversity, jaccard, ndcg_at_k, paired_bootstrap, precision_at_k,
                            recall_at_k, reciprocal_rank, split_interactions)
from src.synthetic_data import generate_interactions, load_interactions


def test_ranking_metrics_hand_computed():
    ranked, relevant = ["a", "x", "b", "y"], {"a", "b", "c"}
    assert precision_at_k(ranked, relevant, k=4) == pytest.approx(2 / 4)
    assert recall_at_k(ranked, relevant, k=4) == pytest.approx(2 / 3)
    assert reciprocal_rank(["x", "a"], relevant, k=4) == pytest.approx(1 / 2)
    assert hit_at_k(["x", "y"], relevant, k=2) == 0.0
    dcg = 1 / math.log2(2) + 1 / math.log2(4)
    ideal = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(ranked, relevant, k=4) == pytest.approx(dcg / ideal)
    assert ndcg_at_k(["a", "b", "c"], relevant, k=3) == pytest.approx(1.0)


def test_metrics_handle_empty_inputs():
    assert math.isnan(recall_at_k(["a"], set()))
    assert math.isnan(ndcg_at_k(["a"], set()))
    assert precision_at_k([], {"a"}) == 0.0 and reciprocal_rank([], {"a"}) == 0.0
    assert math.isnan(catalogue_coverage([], 0))
    assert math.isnan(jaccard([], []))


def test_coverage_diversity_and_jaccard():
    assert catalogue_coverage([["a", "b"], ["b", "c"]], 10) == pytest.approx(0.3)
    vectors = sparse.csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]))
    position = {"a": 0, "b": 1, "c": 2}
    # pairs: a-b distance 1, a-c distance 0, b-c distance 1 -> mean 2/3
    assert intra_list_diversity(["a", "b", "c"], vectors, position) == pytest.approx(2 / 3)
    assert math.isnan(intra_list_diversity(["a"], vectors, position))
    assert jaccard(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3)


def test_paired_bootstrap():
    same = paired_bootstrap(np.array([0.2, 0.4, 0.6]), np.array([0.2, 0.4, 0.6]))
    assert same["difference"] == 0 and same["ci_low"] == 0 and same["ci_high"] == 0
    better = paired_bootstrap(np.full(50, 0.9), np.full(50, 0.1))
    assert better["ci_low"] > 0
    assert math.isnan(paired_bootstrap(np.array([np.nan]), np.array([1.0]))["difference"])


def test_split_holds_out_latest_courses_without_leakage(engine):
    train, test = split_interactions(engine.interactions)
    assert train.merge(test, on=["learner_id", "course_id"]).empty
    assert set(test["learner_id"]) == set(engine.interactions["learner_id"])        # every learner tested
    last_train = train.groupby("learner_id")["sequence"].max()
    first_test = test.groupby("learner_id")["sequence"].min()
    assert (first_test > last_train.reindex(first_test.index)).all()               # test = latest courses


def test_synthetic_data_is_labelled_and_refused_otherwise(engine, tmp_path):
    assert engine.interactions["is_synthetic"].all()
    assert engine.interactions["learner_id"].str.startswith("synthetic_").all()
    unlabelled = engine.interactions.drop(columns="is_synthetic")
    unlabelled.to_csv(tmp_path / "fake.csv", index=False)
    with pytest.raises(ValueError, match="synthetic"):
        load_interactions(tmp_path / "fake.csv")
    assert load_interactions(tmp_path / "missing.csv") is None


def test_synthetic_generator_is_reproducible(engine):
    first = generate_interactions(engine.courses, engine.knowledge, n_learners=20, seed=1)
    second = generate_interactions(engine.courses, engine.knowledge, n_learners=20, seed=1)
    pd.testing.assert_frame_equal(first, second)
    assert not first.duplicated(["learner_id", "course_id"]).any()


def test_saved_results_match_a_fresh_computation(engine):
    """Guards against hand-edited or stale numbers in reports/evaluation_results.csv."""
    saved = pd.read_csv(RESULTS_PATH)
    fresh = pd.DataFrame(experiment_similar_courses(engine)[0])
    merged = fresh.merge(saved, on=["experiment", "method", "metric"], suffixes=("_fresh", "_saved"))
    assert len(merged) == len(fresh)
    assert np.allclose(merged["value_fresh"], merged["value_saved"], equal_nan=True)


def test_saved_results_label_their_data_source():
    saved = pd.read_csv(RESULTS_PATH)
    synthetic = saved[saved["experiment"].str.startswith("C.")]
    assert not synthetic.empty and synthetic["data_source"].str.contains("SYNTHETIC").all()
    real = saved[~saved["experiment"].str.startswith("C.")]
    assert real["data_source"].str.contains("proxy").all()


def test_results_notebook_executes_without_errors():
    """The report notebook must run end to end on the saved results."""
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    from src.data_loader import PROJECT_ROOT

    path = PROJECT_ROOT / "notebooks" / "evaluation_results.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbclient.NotebookClient(notebook, timeout=300, kernel_name="python3",
                            resources={"metadata": {"path": str(path.parent)}}).execute()
    errors = [o for cell in notebook.cells if cell.cell_type == "code"
              for o in cell.get("outputs", []) if o.get("output_type") == "error"]
    assert not errors


def test_application_weights_match_the_cross_validated_selection():
    """DEFAULT_WEIGHTS must be the weights the evaluation selected, not hand-edited numbers."""
    import json

    from src.evaluation import METADATA_PATH
    from src.hybrid import DEFAULT_WEIGHTS

    selected = json.loads(METADATA_PATH.read_text(encoding="utf-8"))["experiment_B"]["selected_for_application"]
    for name, weight in selected["weights"].items():
        assert DEFAULT_WEIGHTS[name] == pytest.approx(weight)


def test_graded_ndcg_hand_computed():
    from src.evaluation import graded_ndcg_at_k

    gains = pd.Series({"a": 2, "b": 1, "c": 0, "d": 2})
    # ideal order a/d (2), d/a (2), b (1); ranked: c, a, b
    dcg = 0 + 2 / math.log2(3) + 1 / math.log2(4)
    idcg = 2 / math.log2(2) + 2 / math.log2(3) + 1 / math.log2(4)
    assert graded_ndcg_at_k(["c", "a", "b"], gains, k=3) == pytest.approx(dcg / idcg)
