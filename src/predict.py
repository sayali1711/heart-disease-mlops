"""Command-line inference.

Examples:
    python -m src.predict --json sample_input.json
    python -m src.predict --csv data/processed/heart_clean.csv --n 5
"""
import argparse
import json

import joblib
import pandas as pd

from src.config import FEATURES, MODEL_PATH


def load_model(path=MODEL_PATH):
    return joblib.load(path)


def predict_frame(model, df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES].astype(float)
    proba = model.predict_proba(X)[:, 1]
    out = df.copy()
    out["prediction"] = (proba >= 0.5).astype(int)
    out["probability_disease"] = proba.round(4)
    return out


def main():
    p = argparse.ArgumentParser(description="Heart disease inference")
    p.add_argument("--json", help="JSON file with one patient record")
    p.add_argument("--csv", help="CSV file with many patient records")
    p.add_argument("--n", type=int, default=5, help="rows to score from CSV")
    p.add_argument("--model", default=str(MODEL_PATH))
    args = p.parse_args()

    model = load_model(args.model)
    if args.json:
        with open(args.json) as fh:
            df = pd.DataFrame([json.load(fh)])
    elif args.csv:
        df = pd.read_csv(args.csv).head(args.n)
    else:
        p.error("provide --json or --csv")
    print(predict_frame(model, df)[FEATURES + ["prediction", "probability_disease"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
