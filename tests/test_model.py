import numpy as np

from src.config import FEATURES, TARGET
from src.train import compute_metrics, get_candidates, make_pipeline


def test_candidates_include_at_least_two_models():
    cands = get_candidates()
    assert len(cands) >= 2
    for est, grid in cands.values():
        assert all(k.startswith("model__") for k in grid)
        assert hasattr(est, "fit")


def test_pipeline_trains_and_predicts(trained_model, synthetic_df):
    X = synthetic_df[FEATURES]
    proba = trained_model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    acc = (trained_model.predict(X) == synthetic_df[TARGET]).mean()
    assert acc > 0.7  # signal exists in synthetic data


def test_every_candidate_fits(synthetic_df):
    X, y = synthetic_df[FEATURES], synthetic_df[TARGET]
    for est, _ in get_candidates().values():
        assert make_pipeline(est).fit(X, y).predict(X).shape == (len(X),)


def test_compute_metrics_perfect_and_keys():
    y = np.array([0, 0, 1, 1])
    m = compute_metrics(y, y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert set(m) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert all(v == 1.0 for v in m.values())
