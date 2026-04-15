"""
main.py
-------
🚀 Full Data Science pipeline for the CICIoT2023 dataset.

Steps:
  1. Load data (sample or real)
  2. Inspect raw data
  3. Clean data
  4. EDA visualisations
  5. Feature engineering
  6. Train all models
  7. Compare models
  8. Evaluate best model on test set
  9. Generate Markdown report

Usage:
    python main.py                      # Use sample_data.csv
    python main.py --data data/raw/     # Use all CSVs in data/raw/
    python main.py --data path/to/file.csv
"""

import argparse
import logging
import sys
from pathlib import Path

# ── Ensure src/ is importable ─────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from src.data_cleaning     import (load_csv, load_multiple_csvs,
                                    inspect, clean_pipeline, save_cleaned)
from src.feature_engineering import feature_pipeline
from src.model_training      import (train_all_models, save_best_model,
                                      results_to_dataframe, get_feature_importance)
from src.visualization       import (plot_class_distribution, plot_missing_values,
                                      plot_correlation_heatmap, plot_feature_distributions,
                                      plot_protocol_breakdown, plot_model_comparison,
                                      plot_feature_importance)
from src.evaluation          import evaluate_on_test, generate_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="CICIoT2023 Data Science Pipeline")
    p.add_argument("--data", default="data/sample_data.csv",
                   help="Path to a CSV file or a folder containing CSV files "
                        "(default: data/sample_data.csv)")
    p.add_argument("--no-eda", action="store_true",
                   help="Skip EDA plots (faster, useful for re-runs)")
    p.add_argument("--no-report", action="store_true",
                   help="Skip final Markdown report")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    data_path = Path(args.data)

    print("\n" + "═" * 60)
    print("  CICIoT2023 — End-to-End Data Science Pipeline")
    print("═" * 60 + "\n")

    # ── 1. Generate sample data if it doesn't exist ──────────────────────────
    if not data_path.exists():
        if data_path.suffix == ".csv":
            log.warning(f"'{data_path}' not found. Generating synthetic sample data…")
            import subprocess
            subprocess.run([sys.executable, "data/download_dataset.py", "--sample"],
                           check=True)
        else:
            raise FileNotFoundError(f"Data path not found: {data_path}")

    # ── 2. Load data ─────────────────────────────────────────────────────────
    if data_path.is_dir():
        df_raw = load_multiple_csvs(data_path)
    else:
        df_raw = load_csv(data_path)

    # ── 3. Inspect ───────────────────────────────────────────────────────────
    inspect(df_raw)

    # ── 4. EDA (pre-cleaning) ────────────────────────────────────────────────
    if not args.no_eda:
        log.info("\n── EDA Visualisations (raw data) ───────────────────────")
        plot_missing_values(df_raw)
        plot_class_distribution(df_raw["label"],
                                title="Raw Data — Attack Class Distribution")

    # ── 5. Clean ─────────────────────────────────────────────────────────────
    df_clean = clean_pipeline(df_raw)
    save_cleaned(df_clean, "data/cleaned_data.csv")

    # ── 6. EDA (post-cleaning) ───────────────────────────────────────────────
    if not args.no_eda:
        log.info("\n── EDA Visualisations (cleaned data) ───────────────────")
        plot_correlation_heatmap(df_clean)
        plot_feature_distributions(df_clean)
        plot_protocol_breakdown(df_clean)
        plot_class_distribution(df_clean["label"],
                                title="Cleaned Data — Attack Class Distribution")

    # ── 7. Feature engineering ───────────────────────────────────────────────
    fe = feature_pipeline(df_clean)
    X_train       = fe["X_train"]
    X_val         = fe["X_val"]
    X_test        = fe["X_test"]
    y_train       = fe["y_train"]
    y_val         = fe["y_val"]
    y_test        = fe["y_test"]
    class_map     = fe["class_map"]
    label_encoder = fe["label_encoder"]
    feature_names = fe["feature_names"]
    class_names   = [class_map[i] for i in sorted(class_map)]

    # ── 8. Train all models ──────────────────────────────────────────────────
    trained_models, results = train_all_models(
        X_train, y_train, X_val, y_val, class_names=class_names
    )

    # ── 9. Compare models ────────────────────────────────────────────────────
    results_df = results_to_dataframe(results)
    print("\n" + "─" * 60)
    print("  Validation Set — Model Comparison")
    print("─" * 60)
    print(results_df.to_string(index=False))
    print("─" * 60 + "\n")

    if not args.no_eda:
        plot_model_comparison({k: v for k, v in results.items()
                               if k not in ("predictions", "probabilities")})

    # ── 10. Best model ───────────────────────────────────────────────────────
    best_name, best_path = save_best_model(trained_models, results)
    best_model = trained_models[best_name]

    # Feature importance
    imp = get_feature_importance(best_model, feature_names)
    if imp is not None and not args.no_eda:
        plot_feature_importance(imp, top_n=20, model_name=best_name)

    # ── 11. Test set evaluation ──────────────────────────────────────────────
    test_metrics = evaluate_on_test(
        best_model, X_test, y_test, class_map=class_map, model_name=best_name
    )

    # ── 12. Report ───────────────────────────────────────────────────────────
    if not args.no_report:
        generate_report(
            results=results,
            best_model_name=best_name,
            test_metrics=test_metrics,
            feature_names=feature_names,
            class_map=class_map,
        )

    print("\n" + "═" * 60)
    print("  ✅ Pipeline Complete!")
    print(f"  Best Model : {best_name}")
    print(f"  Test F1    : {test_metrics['f1']:.4f}")
    print(f"  Test Acc   : {test_metrics['accuracy']:.4f}")
    print(f"  Model saved: {best_path}")
    print(f"  Figures    : figures/")
    print(f"  Report     : reports/data_science_report.md")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()