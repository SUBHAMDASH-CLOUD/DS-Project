"""
src/visualization.py
--------------------
EDA and evaluation visualisations for the CICIoT2023 dataset.
All functions save figures to the `figures/` directory.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for script use
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize

log = logging.getLogger(__name__)

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

PALETTE = "tab20"
plt.style.use("seaborn-v0_8-whitegrid")


def _save(fig, name: str):
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Figure saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# EDA plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_class_distribution(y: pd.Series, title: str = "Attack Class Distribution"):
    """Bar chart of label frequencies."""
    counts = y.value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(5, len(counts) * 0.45)))
    colors  = plt.cm.get_cmap(PALETTE)(np.linspace(0, 1, len(counts)))
    bars    = ax.barh(counts.index.astype(str), counts.values, color=colors)

    for bar, val in zip(bars, counts.values):
        ax.text(val + max(counts.values)*0.005, bar.get_y() + bar.get_height()/2,
                f"{val:,}", va="center", fontsize=9)

    ax.set_xlabel("Sample Count", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    _save(fig, "class_distribution.png")


def plot_missing_values(df: pd.DataFrame):
    """Heatmap of missing-value fractions per column."""
    missing_pct = df.isna().mean().sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0]
    if missing_pct.empty:
        log.info("No missing values — skipping missing-values plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    sns.barplot(x=missing_pct.index, y=missing_pct.values * 100,
                palette="Reds_r", ax=ax)
    ax.set_ylabel("Missing %")
    ax.set_xlabel("Feature")
    ax.set_title("Missing Value Percentage per Feature", fontweight="bold")
    ax.tick_params(axis="x", rotation=60, labelsize=8)
    fig.tight_layout()
    _save(fig, "missing_values.png")


def plot_correlation_heatmap(df: pd.DataFrame, top_n: int = 25):
    """Correlation heatmap for the top-N most-varied numeric features."""
    num_df = df.select_dtypes(include="number").drop(columns=["label"], errors="ignore")
    if num_df.shape[1] > top_n:
        top_cols = num_df.std().nlargest(top_n).index
        num_df   = num_df[top_cols]

    corr = num_df.corr()
    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0,
                linewidths=0.3, annot=False, fmt=".2f",
                ax=ax, vmin=-1, vmax=1)
    ax.set_title(f"Feature Correlation Heatmap (Top {top_n} features)",
                 fontsize=14, fontweight="bold", pad=12)
    fig.tight_layout()
    _save(fig, "correlation_heatmap.png")


def plot_feature_distributions(df: pd.DataFrame, features: list[str] | None = None,
                                n_cols: int = 4):
    """KDE/histogram for selected numeric features, coloured by label."""
    if features is None:
        num_cols = df.select_dtypes(include="number").drop(
            columns=["label"], errors="ignore"
        ).columns.tolist()
        # pick the features with highest variance
        features = df[num_cols].std().nlargest(12).index.tolist()

    n_rows = -(-len(features) // n_cols)       # ceiling division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
    axes = axes.flatten()

    for ax, col in zip(axes, features):
        if "label" in df.columns:
            for lbl, grp in df.groupby("label"):
                ax.hist(grp[col].dropna(), bins=40, alpha=0.5,
                        label=str(lbl), density=True)
        else:
            ax.hist(df[col].dropna(), bins=40, color="steelblue")
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("")
        ax.set_yticks([])

    # Hide unused axes
    for ax in axes[len(features):]:
        ax.set_visible(False)

    fig.suptitle("Feature Distributions by Class", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "feature_distributions.png")


def plot_protocol_breakdown(df: pd.DataFrame):
    """Stacked bar showing protocol usage per attack class."""
    proto_cols = [c for c in ["TCP", "UDP", "ICMP", "HTTP", "HTTPS", "ARP", "DNS"]
                  if c in df.columns]
    if not proto_cols or "label" not in df.columns:
        return

    pivot = df.groupby("label")[proto_cols].mean()
    pivot.plot(kind="bar", stacked=True, colormap="tab20",
               figsize=(14, 6))
    plt.title("Protocol Usage per Attack Class (normalised)", fontweight="bold")
    plt.xlabel("Attack Class")
    plt.ylabel("Mean Presence (0-1)")
    plt.xticks(rotation=45, ha="right")
    plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    fig = plt.gcf()
    _save(fig, "protocol_breakdown.png")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, class_names: list[str] | None = None):
    """Normalised confusion matrix."""
    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        display_labels=class_names,
        normalize="true",
        cmap="Blues",
        ax=ax,
        colorbar=False,
        values_format=".2f",
    )
    ax.set_title("Normalised Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
    fig.tight_layout()
    _save(fig, "confusion_matrix.png")


def plot_feature_importance(importance: pd.Series, top_n: int = 20, model_name: str = ""):
    """Horizontal bar chart of feature importances."""
    importance = importance.nlargest(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.38)))
    colors = plt.cm.get_cmap("RdYlGn")(np.linspace(0.25, 0.85, top_n))
    ax.barh(importance.index, importance.values, color=colors)
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Top-{top_n} Feature Importances — {model_name}",
                 fontweight="bold")
    fig.tight_layout()
    _save(fig, f"feature_importance_{model_name.replace(' ', '_')}.png")


def plot_model_comparison(results: dict):
    """
    Bar chart comparing multiple models on Accuracy, F1, Precision, Recall.

    results: {model_name: {'accuracy': ..., 'f1': ..., 'precision': ..., 'recall': ...}}
    """
    models  = list(results.keys())
    metrics = ["accuracy", "f1", "precision", "recall"]
    x       = np.arange(len(models))
    width   = 0.2

    fig, ax = plt.subplots(figsize=(max(10, len(models) * 2), 6))
    colors  = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = [results[m].get(metric, 0) for m in models]
        bars = ax.bar(x + i * width, vals, width, label=metric.capitalize(), color=color)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Classification Metrics",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, "model_comparison.png")


def plot_roc_curves(y_test, y_prob, class_map: dict, model_name: str = ""):
    """
    One-vs-Rest ROC curves for multiclass.
    y_prob: (n_samples, n_classes) probability array
    """
    n_classes = len(class_map)
    y_bin     = label_binarize(y_test, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(10, 8))
    colors  = plt.cm.get_cmap("tab20")(np.linspace(0, 1, n_classes))

    for i, (idx, label) in enumerate(class_map.items()):
        if y_bin.shape[1] <= i:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i], lw=1.5,
                label=f"{label} (AUC={roc_auc:.2f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves (OvR) — {model_name}", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    _save(fig, f"roc_curves_{model_name.replace(' ', '_')}.png")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data_cleaning import load_csv, clean_pipeline

    df = clean_pipeline(load_csv("data/sample_data.csv"))
    plot_class_distribution(df["label"])
    plot_missing_values(df)
    plot_correlation_heatmap(df)
    plot_feature_distributions(df)
    plot_protocol_breakdown(df)
    print("All EDA figures saved to figures/")
