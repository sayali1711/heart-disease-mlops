"""Central configuration: paths, schema and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
RAW_PATH = ROOT / "data" / "raw" / "heart_raw.csv"
CLEAN_PATH = ROOT / "data" / "processed" / "heart_clean.csv"

MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
FIGURES_DIR = ROOT / "reports" / "figures"

RAW_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "target"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

EXPERIMENT_NAME = "heart-disease-classification"
