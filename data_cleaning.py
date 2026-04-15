"""
src/data_cleaning.py
--------------------
All data cleaning and preprocessing steps for the CICIoT2023 dataset.

Steps:
  1. Load one or multiple CSV files
  2. Inspect shape, dtypes, and missing values
  3. Handle missing values (NaN) and infinite values (Inf)
  4. Remove duplicate rows
  5. Drop constant and quasi-constant features
  6. Clip extreme outliers
  7. Save cleaned data
"""

import os
import glob
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Column name normalisation map ─────────────────────────────────────────────
RENAME_MAP = {
    "Tot sum":  "Tot_sum",
    "Tot size": "Tot_size",
}

LABEL_COLUMN = "label"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a single CSV file."""
    path = Path(path)
    log.info(f"Loading: {path.name}")
    df = pd.read_csv(path, low_memory=False)
    df.rename(columns=RENAME_MAP, inplace=True)
    df.columns = df.columns.str.strip()
    log.info(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


def load_multiple_csvs(folder: str | Path, pattern: str = "*.csv") -> pd.DataFrame:
    """Load and concatenate all CSVs in a folder."""
    folder = Path(folder)
    files = sorted(folder.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {folder}")
    log.info(f"Found {len(files)} CSV files in '{folder}'")
    frames = [load_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    log.info(f"Combined shape: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Inspection
# ─────────────────────────────────────────────────────────────────────────────

def inspect(df: pd.DataFrame) -> dict:
    """Print and return a summary of the DataFrame."""
    n_rows, n_cols = df.shape
    n_dup  = df.duplicated().sum()
    n_nan  = df.isna().sum().sum()
    n_inf  = np.isinf(df.select_dtypes(include="number")).sum().sum()

    summary = {
        "rows":       n_rows,
        "cols":       n_cols,
        "duplicates": int(n_dup),
        "NaN_total":  int(n_nan),
        "Inf_total":  int(n_inf),
        "dtypes":     df.dtypes.value_counts().to_dict(),
    }

    log.info("── Dataset Inspection ──────────────────────────────────")
    log.info(f"  Rows        : {n_rows:,}")
    log.info(f"  Columns     : {n_cols}")
    log.info(f"  Duplicates  : {n_dup:,}")
    log.info(f"  NaN values  : {n_nan:,}")
    log.info(f"  Inf values  : {n_inf:,}")
    if LABEL_COLUMN in df.columns:
        vc = df[LABEL_COLUMN].value_counts()
        log.info(f"  Label dist  :\n{vc.to_string()}")
    log.info("────────────────────────────────────────────────────────")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# 3. Cleaning steps
# ─────────────────────────────────────────────────────────────────────────────

def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows."""
    before = len(df)
    df = df.drop_duplicates()
    after  = len(df)
    log.info(f"[drop_duplicates] Removed {before - after:,} rows → {after:,} remaining")
    return df.reset_index(drop=True)


def handle_infinite_values(df: pd.DataFrame, strategy: str = "nan") -> pd.DataFrame:
    """
    Replace ±Inf with NaN (default) or with column max/min.
    strategy: 'nan' | 'clip'
    """
    num_cols = df.select_dtypes(include="number").columns
    n_inf = np.isinf(df[num_cols]).sum().sum()

    if strategy == "nan":
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
        log.info(f"[handle_inf] Replaced {n_inf:,} Inf values with NaN")
    elif strategy == "clip":
        for col in num_cols:
            finite_max = df[col].replace([np.inf, -np.inf], np.nan).max()
            finite_min = df[col].replace([np.inf, -np.inf], np.nan).min()
            df[col] = df[col].clip(lower=finite_min, upper=finite_max)
        log.info(f"[handle_inf] Clipped {n_inf:,} Inf values to col min/max")
    return df


def handle_missing_values(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    """
    Fill NaN in numeric columns.
    strategy: 'median' | 'mean' | 'zero' | 'drop'
    """
    num_cols = df.select_dtypes(include="number").columns
    n_nan_before = df[num_cols].isna().sum().sum()

    if strategy == "drop":
        df = df.dropna()
        log.info(f"[handle_nan] Dropped rows with NaN. New shape: {df.shape}")
    else:
        fill_vals = {
            "median": df[num_cols].median(),
            "mean":   df[num_cols].mean(),
            "zero":   pd.Series(0, index=num_cols),
        }.get(strategy, df[num_cols].median())

        df[num_cols] = df[num_cols].fillna(fill_vals)
        n_nan_after = df[num_cols].isna().sum().sum()
        log.info(f"[handle_nan] Filled {n_nan_before:,} NaN values (strategy='{strategy}'). "
                 f"Remaining: {n_nan_after}")
    return df.reset_index(drop=True)


def drop_constant_features(df: pd.DataFrame, threshold: float = 0.001) -> pd.DataFrame:
    """
    Drop columns where the fraction of unique values is below `threshold`,
    i.e. near-constant features that carry no predictive signal.
    The label column is always kept.
    """
    num_cols   = [c for c in df.select_dtypes(include="number").columns
                  if c != LABEL_COLUMN]
    to_drop    = []

    for col in num_cols:
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio < threshold:
            to_drop.append(col)

    if to_drop:
        df = df.drop(columns=to_drop)
        log.info(f"[drop_constant] Dropped {len(to_drop)} near-constant columns: {to_drop}")
    else:
        log.info("[drop_constant] No constant columns found.")
    return df


def clip_outliers(df: pd.DataFrame, lower_q: float = 0.001,
                  upper_q: float = 0.999) -> pd.DataFrame:
    """
    Clip numeric feature values to [lower_q, upper_q] quantiles.
    Helps reduce the effect of extreme outliers without removing rows.
    """
    num_cols = [c for c in df.select_dtypes(include="number").columns
                if c != LABEL_COLUMN]

    lower = df[num_cols].quantile(lower_q)
    upper = df[num_cols].quantile(upper_q)
    df[num_cols] = df[num_cols].clip(lower=lower, upper=upper, axis=1)

    log.info(f"[clip_outliers] Clipped outliers at [{lower_q*100:.1f}%, {upper_q*100:.1f}%] quantiles")
    return df


def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce numeric-looking object columns to numeric."""
    for col in df.columns:
        if col == LABEL_COLUMN:
            continue
        if df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().mean() > 0.9:
                df[col] = converted
                log.info(f"[fix_dtypes] Converted '{col}' object → numeric")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. Full cleaning pipeline
# ─────────────────────────────────────────────────────────────────────────────

def clean_pipeline(
    df: pd.DataFrame,
    nan_strategy:   str   = "median",
    inf_strategy:   str   = "nan",
    clip:           bool  = True,
    drop_constants: bool  = True,
    lower_q:        float = 0.001,
    upper_q:        float = 0.999,
) -> pd.DataFrame:
    """
    Execute the full cleaning pipeline in order.

    Parameters
    ----------
    df            : Raw DataFrame
    nan_strategy  : How to handle NaN — 'median', 'mean', 'zero', or 'drop'
    inf_strategy  : How to handle Inf — 'nan' (replace) or 'clip'
    clip          : Whether to clip outliers
    drop_constants: Whether to drop near-constant columns
    lower_q, upper_q: Quantile bounds for outlier clipping

    Returns
    -------
    Cleaned DataFrame
    """
    log.info("═" * 55)
    log.info("  Starting Data Cleaning Pipeline")
    log.info("═" * 55)

    df = fix_dtypes(df)
    df = drop_duplicates(df)
    df = handle_infinite_values(df, strategy=inf_strategy)
    df = handle_missing_values(df, strategy=nan_strategy)

    if drop_constants:
        df = drop_constant_features(df)

    if clip:
        df = clip_outliers(df, lower_q=lower_q, upper_q=upper_q)

    log.info("═" * 55)
    log.info(f"  Cleaning complete. Final shape: {df.shape[0]:,} × {df.shape[1]}")
    log.info("═" * 55)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. Save
# ─────────────────────────────────────────────────────────────────────────────

def save_cleaned(df: pd.DataFrame, path: str | Path = "data/cleaned_data.csv") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info(f"Cleaned data saved → {path}  ({df.shape[0]:,} rows × {df.shape[1]} cols)")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample = Path("data/sample_data.csv")
    if not sample.exists():
        print("Generating sample data first...")
        import subprocess, sys
        subprocess.run([sys.executable, "data/download_dataset.py", "--sample"])

    df_raw = load_csv(sample)
    inspect(df_raw)
    df_clean = clean_pipeline(df_raw)
    save_cleaned(df_clean, "data/cleaned_data.csv")
