"""Data acquisition, cleaning and the reusable preprocessing transformer."""
import logging
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES, CLEAN_PATH, DATA_URL, FEATURES, NUMERIC_FEATURES,
    RAW_COLUMNS, RAW_PATH, TARGET,
)

logger = logging.getLogger(__name__)


def download_raw(url: str = DATA_URL, dest: Path = RAW_PATH) -> Path:
    """Download the Cleveland subset of the UCI Heart Disease data as a CSV with header."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s", url)
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 (trusted URL)
        text = resp.read().decode("utf-8")
    tmp = dest.with_suffix(".tmp")
    tmp.write_text(",".join(RAW_COLUMNS) + "\n" + text.strip() + "\n")
    tmp.replace(dest)
    logger.info("Saved raw data to %s", dest)
    return dest


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce types, mark '?' as missing, drop duplicates and binarise the target.

    Missing values are intentionally *kept* here and imputed inside the sklearn
    pipeline so that imputation statistics are learned on training folds only
    (prevents data leakage).
    """
    df = df.copy()
    df = df.replace("?", np.nan)

    if "num" in df.columns:  # original multi-class label 0-4 -> binary
        df[TARGET] = (pd.to_numeric(df["num"], errors="coerce") > 0).astype(int)
        df = df.drop(columns=["num"])

    missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[FEATURES + [TARGET]].drop_duplicates().reset_index(drop=True)
    return df


def ensure_clean_data(refresh: bool = False) -> pd.DataFrame:
    """Return the cleaned dataset, downloading/cleaning it first if necessary."""
    if refresh or not CLEAN_PATH.exists():
        if refresh or not RAW_PATH.exists():
            download_raw()
        cleaned = clean(load_raw())
        CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
        cleaned.to_csv(CLEAN_PATH, index=False)
        logger.info("Saved cleaned data (%d rows) to %s", len(cleaned), CLEAN_PATH)
        return cleaned
    return pd.read_csv(CLEAN_PATH)


def split_xy(df: pd.DataFrame):
    return df[FEATURES], df[TARGET]


def build_preprocessor() -> ColumnTransformer:
    """Impute + scale numeric features, impute + one-hot encode categorical ones."""
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    data = ensure_clean_data(refresh=True)
    print(data.shape)
    print(data[TARGET].value_counts())
