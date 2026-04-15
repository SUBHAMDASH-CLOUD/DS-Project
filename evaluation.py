"""
src/evaluation.py
-----------------
Final evaluation of the best model on the held-out test set,
plus reporting utilities.
"""

import logging
from pathlib import Path

import pandas as pd
import numpy as np

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")


def evaluate_on_test(model, X_test: pd.DataFrame, y_test: pd.Series,
                     class_map: dict, model_name: str = "") -> dict:
    """
    Full evaluation of a model on the test set.
    Includes confusion matrix and ROC curves via visualization.py.
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        classification_report
    )
    from src.visualization import plot_confusion_matrix, plot_roc_curves

    class_names = [class_map[i] for i in sorted(class_map)]

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall":    recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1":        f1_score(y_test, y_pred, average="macro", zero_division=0),
    }

    log.info(f"\n{'═'*55}")
    log.info(f"  Test Set Results — {model_name}")
    log.info(f"{'═'*55}")
    log.info(f"  Accuracy : {metrics['accuracy']:.4f}")
    log.info(f"  Precision: {metrics['precision']:.4f}")
    log.info(f"  Recall   : {metrics['recall']:.4f}")
    log.info(f"  F1       : {metrics['f1']:.4f}")

    report = classification_report(y_test, y_pred,
                                   target_names=class_names,
                                   zero_division=0)
    log.info(f"\n{report}")

    plot_confusion_matrix(y_test, y_pred, class_names=class_names)

    if y_prob is not None:
        plot_roc_curves(y_test, y_prob, class_map=class_map, model_name=model_name)

    return metrics


def generate_report(results: dict, best_model_name: str,
                    test_metrics: dict,
                    feature_names: list[str],
                    class_map: dict,
                    output_path: str = "reports/data_science_report.md") -> Path:
    """
    Generate a Markdown report summarising the entire data science pipeline.
    """
    from src.model_training import results_to_dataframe

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    table_df  = results_to_dataframe(results)
    table_md  = table_df.to_markdown(index=False)
    n_classes = len(class_map)
    classes   = "\n".join(f"  - `{v}`" for v in class_map.values())

    report = f"""# CICIoT2023 — Data Science Report

## 1. Dataset Overview

| Property | Value |
|----------|-------|
| Source | CIC IoT Dataset 2023 (UNB) |
| Attack Types | 33 |
| Attack Categories | 7 (+ Benign) |
| Total Classes | {n_classes} |
| Features (after engineering) | {len(feature_names)} |

### Classes Detected
{classes}

---

## 2. Data Cleaning Summary

| Step | Action |
|------|--------|
| Duplicate Removal | Exact duplicate rows removed |
| Infinite Values | Replaced with NaN, then filled with column median |
| Missing Values | Median imputation for numeric features |
| Constant Features | Columns with near-zero variance removed |
| Outlier Clipping | Values clipped to [0.1%, 99.9%] quantiles |

---

## 3. Feature Engineering

| Step | Action |
|------|--------|
| Label Encoding | String labels → integers via LabelEncoder |
| Variance Filtering | VarianceThreshold(0.01) applied |
| Correlation Filtering | Columns with |r| ≥ 0.95 removed |
| Scaling | StandardScaler (fit on train, apply to val/test) |
| Data Split | 70% Train / 15% Val / 15% Test (stratified) |

**Selected features ({len(feature_names)}):**
```
{', '.join(feature_names)}
```

---

## 4. Model Comparison (Validation Set)

{table_md}

---

## 5. Best Model — Test Set Evaluation

**Model:** `{best_model_name}`

| Metric | Score |
|--------|-------|
| Accuracy | {test_metrics['accuracy']:.4f} |
| Precision (macro) | {test_metrics['precision']:.4f} |
| Recall (macro) | {test_metrics['recall']:.4f} |
| F1 Score (macro) | {test_metrics['f1']:.4f} |

---

## 6. Generated Figures

| Figure | Description |
|--------|-------------|
| `class_distribution.png` | Attack class imbalance |
| `correlation_heatmap.png` | Feature correlations |
| `feature_distributions.png` | Per-class feature distributions |
| `protocol_breakdown.png` | Protocol usage per attack type |
| `confusion_matrix.png` | Normalised confusion matrix (best model) |
| `roc_curves_*.png` | One-vs-Rest ROC curves |
| `model_comparison.png` | Side-by-side model metrics |
| `feature_importance_*.png` | Top feature importances |

---

## 7. Conclusion

The **{best_model_name}** achieved the highest macro F1 score of **{test_metrics['f1']:.4f}**
on the held-out test set, demonstrating excellent ability to distinguish between
benign IoT traffic and {n_classes - 1} attack categories.

Tree-based ensemble methods consistently outperform linear models on this dataset,
likely due to the highly non-linear decision boundaries in network traffic features.

---

## 8. Citation

> E. C. P. Neto, S. Dadkhah, R. Ferreira, A. Zohourian, R. Lu, A. A. Ghorbani.
> "CICIoT2023: A real-time dataset and benchmark for large-scale attacks in IoT environment,"
> Sensor (2023). https://www.mdpi.com/1424-8220/23/13/5941
"""

    path.write_text(report)
    log.info(f"Report saved → {path}")
    return path