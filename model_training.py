"""
src/model_training.py
---------------------
Train and evaluate multiple ML classifiers on the CICIoT2023 dataset.

Models:
  1. Random Forest
  2. Decision Tree
  3. Logistic Regression
  4. K-Nearest Neighbours
  5. XGBoost

Each model is evaluated on accuracy, precision, recall, and macro F1.
The best model is saved as a .pkl file.
"""

import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Model definitions
# ─────────────────────────────────────────────────────────────────────────────

def get_models(random_state: int = 42) -> dict:
    """Return a dict of {model_name: sklearn_estimator}."""
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=5,
            n_jobs=-1,
            random_state=random_state,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=20,
            min_samples_split=5,
            random_state=random_state,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            solver="saga",
            multi_class="auto",
            n_jobs=-1,
            random_state=random_state,
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=5,
            n_jobs=-1,
        ),
    }

    # XGBoost is optional — skip gracefully if not installed
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="mlogloss",
            n_jobs=-1,
            random_state=random_state,
            verbosity=0,
        )
    except ImportError:
        log.warning("XGBoost not installed — skipping. Install with: pip install xgboost")

    return models


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helper
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, X_val: pd.DataFrame, y_val: pd.Series,
             class_names: list[str] | None = None) -> dict:
    """
    Compute evaluation metrics on a validation/test set.

    Returns
    -------
    dict with accuracy, precision, recall, f1, predictions, probabilities
    """
    y_pred = model.predict(X_val)
    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_val)

    metrics = {
        "accuracy":  accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred, average="macro", zero_division=0),
        "recall":    recall_score(y_val, y_pred, average="macro", zero_division=0),
        "f1":        f1_score(y_val, y_pred, average="macro", zero_division=0),
        "predictions":   y_pred,
        "probabilities": y_prob,
    }

    log.info(f"  Accuracy : {metrics['accuracy']:.4f}")
    log.info(f"  Precision: {metrics['precision']:.4f}")
    log.info(f"  Recall   : {metrics['recall']:.4f}")
    log.info(f"  F1 (macro): {metrics['f1']:.4f}")

    report = classification_report(y_val, y_pred,
                                   target_names=class_names,
                                   zero_division=0)
    log.info(f"\n{report}")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val:   pd.DataFrame,
    y_val:   pd.Series,
    class_names: list[str] | None = None,
    random_state: int = 42,
) -> tuple[dict, dict]:
    """
    Train every model and evaluate on the validation set.

    Returns
    -------
    trained_models : {model_name: fitted_estimator}
    results        : {model_name: metrics_dict}
    """
    models         = get_models(random_state)
    trained_models = {}
    results        = {}

    log.info("═" * 55)
    log.info("  Model Training")
    log.info("═" * 55)

    for name, model in models.items():
        log.info(f"\n── {name} ───────────────────────────────────────────")
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0
        log.info(f"  Training time: {elapsed:.1f}s")

        metrics = evaluate(model, X_val, y_val, class_names=class_names)
        metrics["train_time_s"] = round(elapsed, 2)

        trained_models[name] = model
        results[name]        = metrics

    return trained_models, results


# ─────────────────────────────────────────────────────────────────────────────
# Save / Load best model
# ─────────────────────────────────────────────────────────────────────────────

def save_model(model, name: str, path: Path | None = None) -> Path:
    """Serialise a model to disk using joblib."""
    if path is None:
        safe_name = name.replace(" ", "_").lower()
        path = MODELS_DIR / f"{safe_name}.pkl"
    joblib.dump(model, path)
    log.info(f"Model saved → {path}")
    return path


def load_model(path: str | Path):
    """Load a serialised model."""
    return joblib.load(path)


def save_best_model(trained_models: dict, results: dict) -> tuple[str, Path]:
    """Save the model with the highest validation F1 score."""
    best_name = max(results, key=lambda n: results[n]["f1"])
    best_path = save_model(trained_models[best_name], best_name,
                           MODELS_DIR / "best_model.pkl")
    log.info(f"\n🏆 Best model: '{best_name}'  F1={results[best_name]['f1']:.4f}")
    return best_name, best_path


# ─────────────────────────────────────────────────────────────────────────────
# Feature importance extraction
# ─────────────────────────────────────────────────────────────────────────────

def get_feature_importance(model, feature_names: list[str]) -> pd.Series | None:
    """
    Extract feature importance from tree-based models.
    Returns a pandas Series sorted descending, or None if unavailable.
    """
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=feature_names)
        return imp.sort_values(ascending=False)
    log.info("  Model does not expose feature_importances_")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Results summary table
# ─────────────────────────────────────────────────────────────────────────────

def results_to_dataframe(results: dict) -> pd.DataFrame:
    """Convert results dict to a tidy DataFrame for reporting."""
    rows = []
    for name, m in results.items():
        rows.append({
            "Model":          name,
            "Accuracy":       round(m["accuracy"],  4),
            "Precision":      round(m["precision"], 4),
            "Recall":         round(m["recall"],    4),
            "F1 (macro)":     round(m["f1"],        4),
            "Train Time (s)": m.get("train_time_s", "—"),
        })
    df = pd.DataFrame(rows).sort_values("F1 (macro)", ascending=False).reset_index(drop=True)
    return df


if __name__ == "__main__":
    # Quick smoke test with synthetic data
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=5000, n_features=20, n_classes=5,
                                n_informative=10, random_state=42)
    X_tr, X_v = X[:4000], X[4000:]
    y_tr, y_v = y[:4000], y[4000:]

    X_tr = pd.DataFrame(X_tr, columns=[f"f{i}" for i in range(20)])
    X_v  = pd.DataFrame(X_v,  columns=[f"f{i}" for i in range(20)])
    y_tr, y_v = pd.Series(y_tr), pd.Series(y_v)

    trained, res = train_all_models(X_tr, y_tr, X_v, y_v)
    print(results_to_dataframe(res).to_string(index=False))
    save_best_model(trained, res)
