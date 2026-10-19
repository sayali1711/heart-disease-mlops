"""Shared fixtures. Tests use synthetic data so they run offline and fast."""
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.config import FEATURES, RAW_COLUMNS, TARGET
from src.train import make_pipeline


def make_synthetic(n: int = 240, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(29, 78, n),
        "sex": rng.integers(0, 2, n),
        "cp": rng.integers(1, 5, n),
        "trestbps": rng.integers(94, 200, n),
        "chol": rng.integers(126, 564, n),
        "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n),
        "thalach": rng.integers(71, 202, n),
        "exang": rng.integers(0, 2, n),
        "oldpeak": rng.uniform(0, 6, n).round(1),
        "slope": rng.integers(1, 4, n),
        "ca": rng.integers(0, 4, n),
        "thal": rng.choice([3, 6, 7], n),
    }).astype(float)
    score = 0.04 * df["age"] + 1.2 * df["exang"] + 0.8 * df["oldpeak"] - 0.03 * df["thalach"]
    df[TARGET] = (score + rng.normal(0, 0.5, n) > score.median()).astype(int)
    return df


@pytest.fixture(scope="session")
def synthetic_df():
    return make_synthetic()


@pytest.fixture
def raw_df(synthetic_df):
    """Looks like the raw UCI file: string '?' placeholders and 0-4 target."""
    df = synthetic_df.copy()
    df["num"] = df[TARGET] * 2  # 0 or 2 -> must be binarised
    df = df.drop(columns=[TARGET])[RAW_COLUMNS].astype(object)
    df.loc[0, "ca"] = "?"
    df.loc[1, "thal"] = "?"
    return df


@pytest.fixture(scope="session")
def trained_model(synthetic_df):
    X, y = synthetic_df[FEATURES], synthetic_df[TARGET]
    return make_pipeline(LogisticRegression(max_iter=500)).fit(X, y)


@pytest.fixture
def model_path(tmp_path, trained_model):
    path = tmp_path / "model.joblib"
    joblib.dump(trained_model, path)
    return path
