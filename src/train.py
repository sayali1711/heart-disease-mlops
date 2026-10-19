"""Train, tune, evaluate and track models with MLflow.

Run:  python -m src.train
View: mlflow ui --backend-store-uri ./mlruns
"""
import json
import logging
import os
import tempfile
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import (  # noqa: E402
    GridSearchCV, StratifiedKFold, cross_validate, train_test_split,
)
from sklearn.pipeline import Pipeline  # noqa: E402

from src.config import (  # noqa: E402
    CV_FOLDS, EXPERIMENT_NAME, METRICS_PATH, MODEL_DIR, MODEL_PATH,
    RANDOM_STATE, TEST_SIZE,
)
from src.data_processing import build_preprocessor, ensure_clean_data, split_xy  # noqa: E402

logger = logging.getLogger("train")

CV_SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def make_pipeline(estimator) -> Pipeline:
    """Preprocessing + estimator in ONE object -> identical transforms at train & serve time."""
    return Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])


def get_candidates() -> dict:
    """Candidate models and their hyper-parameter grids."""
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            {"model__C": [0.01, 0.1, 1, 10], "model__class_weight": [None, "balanced"]},
        ),
        "random_forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "model__n_estimators": [100, 300],
                "model__max_depth": [3, 5, None],
                "model__min_samples_leaf": [1, 3, 5],
            },
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "model__n_estimators": [50, 150],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3],
            },
        ),
    }


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def _log_plot(fig, filename: str, tmpdir: str):
    path = os.path.join(tmpdir, filename)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(path, artifact_path="plots")


def train_one(name, estimator, grid, X_train, X_test, y_train, y_test):
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags({"model_family": name, "stage": "experiment"})
        mlflow.log_params({
            "cv_folds": CV_FOLDS, "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE, "search_scoring": "roc_auc",
            "n_train": len(X_train), "n_test": len(X_test),
        })

        search = GridSearchCV(
            make_pipeline(estimator), grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True,
        )
        search.fit(X_train, y_train)
        best = search.best_estimator_
        mlflow.log_params({k: str(v) for k, v in search.best_params_.items()})
        logger.info("%s best params: %s", name, search.best_params_)

        # Cross-validated metrics of the tuned pipeline (training data only)
        cv_res = cross_validate(best, X_train, y_train, cv=cv, scoring=CV_SCORING, n_jobs=-1)
        cv_metrics = {}
        for metric in CV_SCORING:
            vals = cv_res[f"test_{metric}"]
            cv_metrics[f"cv_{metric}_mean"] = float(vals.mean())
            cv_metrics[f"cv_{metric}_std"] = float(vals.std())
        mlflow.log_metrics(cv_metrics)

        # Held-out test metrics
        y_pred = best.predict(X_test)
        y_proba = best.predict_proba(X_test)[:, 1]
        test_metrics = {f"test_{k}": v for k, v in compute_metrics(y_test, y_pred, y_proba).items()}
        mlflow.log_metrics(test_metrics)

        # Plots
        with tempfile.TemporaryDirectory() as tmp:
            fig, ax = plt.subplots(figsize=(5, 4))
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap="Blues")
            ax.set_title(f"Confusion matrix - {name}")
            _log_plot(fig, "confusion_matrix.png", tmp)

            fig, ax = plt.subplots(figsize=(5, 4))
            RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax, name=name)
            ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
            ax.set_title(f"ROC curve - {name}")
            _log_plot(fig, "roc_curve.png", tmp)

        signature = infer_signature(X_train, best.predict_proba(X_train))
        mlflow.sklearn.log_model(
            best, artifact_path="model", signature=signature, input_example=X_train.head(3),
        )

        return {
            "name": name, "run_id": run.info.run_id, "best_params": search.best_params_,
            "cv": cv_metrics, "test": test_metrics, "estimator": best,
        }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = ensure_clean_data()
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE,
    )
    logger.info("Train=%d Test=%d  positive rate=%.2f", len(X_train), len(X_test), y.mean())

    results = [
        train_one(name, est, grid, X_train, X_test, y_train, y_test)
        for name, (est, grid) in get_candidates().items()
    ]

    # Select on cross-validated ROC-AUC (NOT on the test set) to avoid selection bias
    winner = max(results, key=lambda r: r["cv"]["cv_roc_auc_mean"])
    logger.info("Selected model: %s (CV ROC-AUC=%.4f, test ROC-AUC=%.4f)",
                winner["name"], winner["cv"]["cv_roc_auc_mean"], winner["test"]["test_roc_auc"])

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(winner["estimator"], MODEL_PATH)

    summary = {
        "selected_model": winner["name"],
        "selected_run_id": winner["run_id"],
        "best_params": {k: str(v) for k, v in winner["best_params"].items()},
        "cv_metrics": winner["cv"],
        "test_metrics": winner["test"],
        "all_models": {
            r["name"]: {"cv_roc_auc": r["cv"]["cv_roc_auc_mean"],
                        "test_roc_auc": r["test"]["test_roc_auc"]} for r in results
        },
    }
    Path(METRICS_PATH).write_text(json.dumps(summary, indent=2))

    # Summary run: comparison plot + final packaged artifacts
    with mlflow.start_run(run_name="final_model_summary"):
        mlflow.set_tags({"stage": "final", "selected_model": winner["name"]})
        mlflow.log_param("selected_model", winner["name"])
        mlflow.log_param("source_run_id", winner["run_id"])
        mlflow.log_metrics({k: v for k, v in winner["test"].items()})
        with tempfile.TemporaryDirectory() as tmp:
            fig, ax = plt.subplots(figsize=(7, 4))
            names = [r["name"] for r in results]
            ax.bar([i - 0.2 for i in range(len(names))],
                   [r["cv"]["cv_roc_auc_mean"] for r in results], 0.4, label="CV ROC-AUC")
            ax.bar([i + 0.2 for i in range(len(names))],
                   [r["test"]["test_roc_auc"] for r in results], 0.4, label="Test ROC-AUC")
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=15)
            ax.set_ylim(0.5, 1.0)
            ax.legend()
            ax.set_title("Model comparison")
            _log_plot(fig, "model_comparison.png", tmp)
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="packaged")
        mlflow.log_artifact(str(METRICS_PATH), artifact_path="packaged")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
