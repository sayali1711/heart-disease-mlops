import numpy as np
import pandas as pd
import pytest

from src.config import FEATURES, TARGET
from src.data_processing import build_preprocessor, clean, split_xy


def test_clean_binarises_target_and_handles_question_marks(raw_df):
    out = clean(raw_df)
    assert set(out[TARGET].unique()) <= {0, 1}
    assert "num" not in out.columns
    assert out["ca"].isna().sum() >= 1          # '?' became NaN
    assert out["thal"].isna().sum() >= 1
    assert all(pd.api.types.is_numeric_dtype(out[c]) for c in FEATURES)


def test_clean_drops_duplicates(raw_df):
    doubled = pd.concat([raw_df, raw_df], ignore_index=True)
    assert len(clean(doubled)) == len(clean(raw_df))


def test_clean_raises_on_missing_columns(raw_df):
    with pytest.raises(ValueError):
        clean(raw_df.drop(columns=["age"]))


def test_split_xy(synthetic_df):
    X, y = split_xy(synthetic_df)
    assert list(X.columns) == FEATURES
    assert len(X) == len(y)


def test_preprocessor_output_has_no_nans(synthetic_df):
    X, _ = split_xy(synthetic_df)
    X = X.copy()
    X.loc[X.index[:5], "chol"] = np.nan
    X.loc[X.index[5:8], "ca"] = np.nan
    Z = build_preprocessor().fit_transform(X)
    Z = Z.toarray() if hasattr(Z, "toarray") else Z
    assert not np.isnan(Z).any()
    assert Z.shape[0] == len(X) and Z.shape[1] > len(FEATURES)  # one-hot expands columns


def test_preprocessor_ignores_unseen_category(synthetic_df):
    X, _ = split_xy(synthetic_df)
    prep = build_preprocessor().fit(X)
    unseen = X.head(1).copy()
    unseen["thal"] = 99.0
    assert prep.transform(unseen).shape[0] == 1
