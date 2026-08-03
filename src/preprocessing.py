# src/preprocessing.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import os
import pickle

import sys
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.data_loader import load_raw_data



# CONFIGURATION

# Columns to always drop
DROP_COLUMNS = [
    "TransactionID",   # Just an identifier, no predictive value
]

# Missing value threshold
# Columns with more than this % missing will be dropped
MISSING_THRESHOLD = 0.9   # 90%

# Target column
TARGET = "isFraud"

# Output path for preprocessed data
PREPROCESSED_DIR = project_root / "data" / "preprocessed"



# DROP HIGH MISSING VALUE COLUMNS

def drop_high_missing_columns(df, threshold=MISSING_THRESHOLD):
    """
    Drop columns where missing value percentage
    exceeds the given threshold.

    """
    # Calculate missing percentage per column
    missing_pct = df.isnull().mean()

    # Find columns exceeding threshold
    cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()

    # Remove target from drop list (safety check)
    if TARGET in cols_to_drop:
        cols_to_drop.remove(TARGET)

    print(f"\nDropping columns with > {threshold*100}% missing values")
    print(f"Columns dropped  : {len(cols_to_drop)}")
    print(f"Columns kept     : {df.shape[1] - len(cols_to_drop)}")
    print(f"Dropped list     : {cols_to_drop}")

    df = df.drop(columns=cols_to_drop)

    return df, cols_to_drop



# DROP IDENTIFIER COLUMNS

def drop_identifier_columns(df):
    """
    Drop columns that are identifiers and
    carry no predictive information.

    """
    # Only drop columns that exist in df
    cols_to_drop = [col for col in DROP_COLUMNS if col in df.columns]

    print(f"\nDropping identifier columns")
    print(f"Dropped : {cols_to_drop}")

    df = df.drop(columns=cols_to_drop)

    return df



# ENCODE M FEATURES (T/F → 1/0)

def encode_m_features(df):
    """
    Encode M features.

    Binary M features:
        T -> 1
        F -> 0

    M4:
        M0, M1, M2 are categorical
        handled later by LabelEncoder

    """
    print("\nEncoding M features")

    # Binary M features
    binary_features = [
        "M1", "M2", "M3",
        "M5", "M6", "M7",
        "M8", "M9"
    ]

    # Only encode columns that exist in df
    binary_features = [
        col for col in binary_features
        if col in df.columns
    ]

    for col in binary_features:
        df[col] = df[col].map({
            "T": 1,
            "F": 0
        })

    print(f"Binary encoded   : {binary_features}")

    # M4 is intentionally skipped
    # because it has categorical values: M0, M1, M2
    # It will be handled by encode_categorical_features()
    if "M4" in df.columns:
        print("M4 kept for Label Encoding")

    return df



# ENCODE CATEGORICAL FEATURES

def encode_categorical_features(df, encoders=None, fit=True):
    """
    Encode categorical (object type) features
    using Label Encoding.

    From EDA:
    Categorical features include:
    - ProductCD      : W, C, R, H, S
    - card4          : visa, mastercard, etc.
    - card6          : debit, credit, etc.
    - P_emaildomain
    - R_emaildomain
    - M4             : M0, M1, M2 (string categories)
    - id_12, id_15, id_16, etc.
    - DeviceType     : desktop, mobile
    - DeviceInfo     : many unique values

    """
    # Get categorical columns (object dtype)
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # Remove target if present
    if TARGET in cat_cols:
        cat_cols.remove(TARGET)

    print(f"\nEncoding categorical features")
    print(f"Categorical columns found : {len(cat_cols)}")
    print(f"Columns : {cat_cols}")

    if encoders is None:
        encoders = {}

    for col in cat_cols:
        if fit:
            le = LabelEncoder()

            # Fill NaN temporarily for encoding
            # NaN → string "missing" before encoding
            df[col] = df[col].fillna("missing")
            le.fit(df[col])
            df[col] = le.transform(df[col])
            encoders[col] = le

        else:
            # Transform mode - use existing encoder
            le = encoders[col]
            df[col] = df[col].fillna("missing")

            # Handle unseen labels gracefully
            df[col] = df[col].apply(
                lambda x: x if x in le.classes_ else "missing"
            )
            df[col] = le.transform(df[col])

    return df, encoders



# IMPUTE MISSING VALUES

def impute_missing_values(df, impute_values=None, fit=True):
    """
    Impute missing values using median strategy.

    - Numerical columns → median imputation
    - After encoding categoricals, all cols are numerical

    From EDA:
    - C features  : No missing values
    - D features  : Up to 93% missing
    - V features  : Up to 86% missing
    - id features : Up to 99% missing
    - M features  : Up to 59% missing (after encoding)

    Strategy:
    - Median for all (robust to outliers and skewness)

    """
    print(f"\nImputing missing values")

    # Separate target before imputation
    target_col = None
    if TARGET in df.columns:
        target_col = df[TARGET].copy()
        df = df.drop(columns=[TARGET])

    if fit:
        impute_values = {}

        for col in df.columns:
            if df[col].isnull().any():
                median_val = df[col].median()
                impute_values[col] = median_val
                df[col] = df[col].fillna(median_val)

    else:
        # Transform mode - use pre-computed values
        for col in df.columns:
            if col in impute_values and df[col].isnull().any():
                df[col] = df[col].fillna(impute_values[col])

    # Put target back
    if target_col is not None:
        df[TARGET] = target_col

    missing_after = df.isnull().sum().sum()
    print(f"Missing values remaining : {missing_after}")

    return df, impute_values



# FEATURE TRANSFORMATION

def transform_features(df):
    """
    Apply feature transformations based on EDA findings.

    Transformations:
    1. TransactionAmt → log1p
       From EDA: mean=135, max=31937, heavily right-skewed

    2. TransactionDT  → kept as-is
       Temporal features handled in feature_engineering.py

    """
    print(f"\nApplying feature transformations")

    # Log transform TransactionAmt
    if "TransactionAmt" in df.columns:
        df["TransactionAmt"] = np.log1p(df["TransactionAmt"])
        print(f"TransactionAmt → log1p applied")

    return df


# STANDARD SCALING (Optional)

def scale_features(df, scaler=None, fit=True):
    """
    Apply Standard Scaling to numerical features.

        - Use with Linear Models (e.g. Logistic Regression, SVM, Neural Netwroks etc)
        - No need to use with Tree-Based Models (e.g. Random Forest, XGBoost, LightGBM etc)

    NOTE:
        - Target column (isFraud) is NOT scaled
        - TransactionID already dropped
        - Only numerical columns are scaled

    """
    print(f"\nApplying Standard Scaling")

    # Separate target
    target_col = None
    if TARGET in df.columns:
        target_col = df[TARGET].copy()
        df = df.drop(columns=[TARGET])

    # Get numerical columns only
    num_cols = df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    print(f"Columns to scale : {len(num_cols)}")

    if fit:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
    else:
        df[num_cols] = scaler.transform(df[num_cols])

    # Put target back
    if target_col is not None:
        df[TARGET] = target_col

    print(f"Scaling complete")

    return df, scaler



# SAVE FUNCTIONS

def save_preprocessed_data(df, filename):
    """
    Save the preprocessed DataFrame to the
    data/preprocessed directory.

    """
    # Create directory if it doesn't exist
    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    output_path = PREPROCESSED_DIR / filename
    df.to_parquet(
    output_path,
    index=False,
    engine="pyarrow",
    compression="snappy"
)

    print(f"\nPreprocessed data saved")
    print(f"Path  : {output_path}")
    print(f"Shape : {df.shape}")


def save_encoders(encoders, impute_values, dropped_cols, scaler=None):
    """
    Save encoders, imputation values, dropped columns
    and scaler for use during prediction on new data.

    """
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Save label encoders
    with open(models_dir / "label_encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)

    # Save imputation values
    with open(models_dir / "impute_values.pkl", "wb") as f:
        pickle.dump(impute_values, f)

    # Save dropped columns list
    with open(models_dir / "dropped_columns.pkl", "wb") as f:
        pickle.dump(dropped_cols, f)

    # Save scaler only if provided
    if scaler is not None:
        with open(models_dir / "scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        print(f"Scaler saved        : models/scaler.pkl")

    print(f"Encoders saved      : models/label_encoders.pkl")
    print(f"Impute vals saved   : models/impute_values.pkl")
    print(f"Dropped cols saved  : models/dropped_columns.pkl")



# MAIN PIPELINE

def preprocess(save=True, apply_scaling=False):
    """
    Full preprocessing pipeline.

    Order of operations:
    1. Load raw data
    2. Drop high missing columns  (> 90%)
    3. Drop identifier columns    (TransactionID)
    4. Encode M features          (T/F → 1/0)
    5. Encode categorical features(Label Encoding)
    6. Impute missing values      (Median)
    7. Transform features         (log1p on TransactionAmt)
    8. Scale features             (StandardScaler - optional)
    9. Save preprocessed data

    """
    print("=" * 60)
    print("         PREPROCESSING PIPELINE")
    print("=" * 60)

    # ── Step 1: Load Data ──────────────────────────
    print(f"\n[Step 1] Loading raw data...")
    df = load_raw_data()
    print(f"         Shape after loading   : {df.shape}")

    # ── Step 2: Drop High Missing Columns ──────────
    print(f"\n[Step 2] Dropping high missing columns...")
    df, dropped_cols = drop_high_missing_columns(df)
    print(f"         Shape after dropping  : {df.shape}")

    # ── Step 3: Drop Identifier Columns ────────────
    print(f"\n[Step 3] Dropping identifier columns...")
    df = drop_identifier_columns(df)
    print(f"         Shape after id drop   : {df.shape}")

    # ── Step 4: Encode M Features ──────────────────
    print(f"\n[Step 4] Encoding M features...")
    df = encode_m_features(df)

    # ── Step 5: Encode Categorical Features ────────
    print(f"\n[Step 5] Encoding categorical features...")
    df, encoders = encode_categorical_features(df, fit=True)

    # ── Step 6: Impute Missing Values ──────────────
    print(f"\n[Step 6] Imputing missing values...")
    df, impute_values = impute_missing_values(df, fit=True)

    # ── Step 7: Transform Features ─────────────────
    print(f"\n[Step 7] Transforming features...")
    df = transform_features(df)

    # ── Step 8: Scale Features (Optional) ──────────
    scaler = None
    if apply_scaling:
        print(f"\n[Step 8] Scaling features...")
        df, scaler = scale_features(df, fit=True)
    else:
        print(f"\n[Step 8] Scaling skipped (tree-based models)")

    # ── Step 9: Save ───────────────────────────────
    if save:
        print(f"\n[Step 9] Saving outputs...")
        if apply_scaling:
            save_preprocessed_data(
                df,
                "train_preprocessed_scaled.parquet"
            )
        else:
            save_preprocessed_data(
                df,
                "train_preprocessed.parquet"
            )
        save_encoders(encoders, impute_values, dropped_cols, scaler)

    print("\n" + "=" * 60)
    print("         PREPROCESSING COMPLETE")
    print(f"         Final Shape : {df.shape}")
    print("=" * 60)

    return df, encoders, impute_values, dropped_cols, scaler


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # ── For Tree-Based Models ──────────────────
    df, encoders, impute_values, dropped_cols, scaler = preprocess(
        save=True,
        apply_scaling=False
    )

    # ── For Linear Models (uncomment if needed) ─
    # df, encoders, impute_values, dropped_cols, scaler = preprocess(
    #     save=True,
    #     apply_scaling=True
    # )

    print("\n── Sample of preprocessed data ──")
    print(df.head())

    print("\n── Data types ──")
    print(df.dtypes.value_counts())

    print("\n── Target distribution ──")
    print(df["isFraud"].value_counts())