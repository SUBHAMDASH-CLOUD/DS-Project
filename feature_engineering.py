"""
src/feature_engineering.py
--------------------------
Feature engineering for the CICIoT2023 dataset:

  1. Encode categorical label column
  2. Separate features / target
  3. Correlation-based feature filtering
  4. Variance-based feature selection
  5. Normalisation / Standardisation
  6. Train / Validation / Test split
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")

LABEL_COLUMN = "label"
RANDOM_STATE = 42


# ─────────────────────────────────────────────────────────────────────────────
# 1. Label Encoding
# ─────────────────────────────────────────────────────────────────────────────

def encode_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder, dict]:
    """
    Encode string attack labels to integer classes.

    Returns
    -------
    df            : DataFrame with integer 'label' column
    le            : Fitted LabelEncoder (needed for inverse_transform)
    class_map     : dict {int → label_string}
    """
    le = LabelEncoder()
    df = df.copy()
    df[LABEL_COLUMN] = le.fit_transform(df[LABEL_COLUMN].astype(str))
    class_map = dict(enumerate(le.classes_))
    log.info(f"[encode_labels] {len(class_map)} classes: {list(le.classes_)}")
    return df, le, class_map


# ─────────────────────────────────────────────────────────────────────────────
# 2. Feature / Target Separation
# ─────────────────────────────────────────────────────────────────────────────

def split_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate feature matrix X and target vector y."""
    y = df[LABEL_COLUMN]
    X = df.drop(columns=[LABEL_COLUMN])
    log.info(f"[split_X_y] X: {X.shape}, y: {y.shape}  ({y.nunique()} classes)")
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# 3. Correlation-based Feature Removal
# ─────────────────────────────────────────────────────────────────────────────

def remove_highly_correlated(X: pd.DataFrame,
                              threshold: float = 0.95) -> pd.DataFrame:
    """
    Drop one of each pair of features with Pearson |r| ≥ threshold.
    Keeps the first column in each pair.
    """
    corr_matrix = X.corr().abs()
    upper       = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [col for col in upper.columns if any(upper[col] >= threshold)]
    X = X.drop(columns=to_drop)
    log.info(f"[remove_corr] Dropped {len(to_drop)} highly-correlated features "
             f"(|r| ≥ {threshold}): {to_drop}")
    return X


# ─────────────────────────────────────────────────────────────────────────────
# 4. Variance Threshold
# ─────────────────────────────────────────────────────────────────────────────

def remove_low_variance(X: pd.DataFrame,
                         threshold: float = 0.01) -> pd.DataFrame:
    """Remove features with variance below `threshold`."""
    selector = VarianceThreshold(threshold=threshold)
    X_sel    = selector.fit_transform(X)
    kept     = X.columns[selector.get_support()]
    dropped  = list(set(X.columns) - set(kept))
    X_out    = pd.DataFrame(X_sel, columns=kept, index=X.index)
    log.info(f"[remove_low_var] Dropped {len(dropped)} low-variance features: {dropped}")
    return X_out


# ─────────────────────────────────────────────────────────────────────────────
# 5. Normalisation
# ─────────────────────────────────────────────────────────────────────────────

def scale_features(X_train: pd.DataFrame,
                   X_val:   pd.DataFrame,
                   X_test:  pd.DataFrame
                   ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Fit StandardScaler on training data and apply to all splits.
    Returns scaled DataFrames and the fitted scaler.
    """
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train),
                              columns=X_train.columns, index=X_train.index)
    X_val_s   = pd.DataFrame(scaler.transform(X_val),
                              columns=X_val.columns,   index=X_val.index)
    X_test_s  = pd.DataFrame(scaler.transform(X_test),
                              columns=X_test.columns,  index=X_test.index)
    log.info("[scale_features] StandardScaler fitted on training set, applied to val/test.")
    return X_train_s, X_val_s, X_test_s, scaler


# ─────────────────────────────────────────────────────────────────────────────
# 6. Train / Val / Test Split
# ─────────────────────────────────────────────────────────────────────────────

def split_data(X: pd.DataFrame, y: pd.Series,
               test_size: float = 0.15,
               val_size:  float = 0.15,
               stratify:  bool  = True
               ) -> tuple:
    """
    Split into train / validation / test (default 70 / 15 / 15).

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    strat = y if stratify else None

    # First split off the test set
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=strat
    )

    # Then split the remainder into train / val
    val_adjusted = val_size / (1 - test_size)
    strat2 = y_tmp if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_adjusted, random_state=RANDOM_STATE, stratify=strat2
    )

    log.info(f"[split_data] Train: {len(X_train):,}  Val: {len(X_val):,}  "
             f"Test: {len(X_test):,}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ─────────────────────────────────────────────────────────────────────────────
# 7. Full Feature Engineering Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def feature_pipeline(
    df: pd.DataFrame,
    corr_threshold:   float = 0.95,
    var_threshold:    float = 0.01,
    test_size:        float = 0.15,
    val_size:         float = 0.15,
) -> dict:
    """
    End-to-end feature engineering.

    Returns a dict with keys:
        X_train, X_val, X_test, y_train, y_val, y_test,
        scaler, label_encoder, class_map, feature_names
    """
    log.info("═" * 55)
    log.info("  Starting Feature Engineering Pipeline")
    log.info("═" * 55)

    df, le, class_map = encode_labels(df)
    X, y = split_X_y(df)

    X = remove_low_variance(X, threshold=var_threshold)
    X = remove_highly_correlated(X, threshold=corr_threshold)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, test_size=test_size, val_size=val_size
    )

    X_train, X_val, X_test, scaler = scale_features(X_train, X_val, X_test)

    log.info(f"  Final feature count : {X_train.shape[1]}")
    log.info("═" * 55)

    return {
        "X_train":       X_train,
        "X_val":         X_val,
        "X_test":        X_test,
        "y_train":       y_train,
        "y_val":         y_val,
        "y_test":        y_test,
        "scaler":        scaler,
        "label_encoder": le,
        "class_map":     class_map,
        "feature_names": list(X_train.columns),
    }


if __name__ == "__main__":
    from data_cleaning import load_csv, clean_pipeline

    sample = Path("data/sample_data.csv")
    df_raw   = load_csv(sample)
    df_clean = clean_pipeline(df_raw)
    result   = feature_pipeline(df_clean)
    print("\nClass map:", result["class_map"])
    print("Training features:", result["feature_names"])
