# src/feature_engineering.py

import numpy as np
import pandas as pd

import sys
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))


# CONFIGURATION

TARGET = "isFraud"

# Top email domains found in EDA
# P_emaildomain top 10:
# gmail, yahoo, hotmail, anonymous,
# aol, comcast, icloud, outlook, msn, att
# Domains not in this list → grouped as "other"

TOP_P_EMAIL_DOMAINS = [
    "gmail.com",
    "yahoo.com",
    "hotmail.com",
    "anonymous.com",
    "aol.com",
    "comcast.net",
    "icloud.com",
    "outlook.com",
    "msn.com",
    "att.net"
]

TOP_R_EMAIL_DOMAINS = [
    "gmail.com",
    "hotmail.com",
    "anonymous.com",
    "yahoo.com",
    "aol.com",
    "outlook.com",
    "comcast.net",
    "yahoo.com.mx",
    "icloud.com",
    "msn.com"
]

# Preprocessed data path
PREPROCESSED_DIR = project_root / "data" / "preprocessed"

# Output path
ENGINEERED_DIR = project_root / "data" / "preprocessed"



# TRANSACTION TIME FEATURES

def create_time_features(df):
    """
    Extract time-based features from TransactionDT.

    From EDA:
    - TransactionDT is elapsed time in seconds
    - Not a real timestamp
    - But we can extract:
        → hour of day (0-23)   : fraud patterns vary by hour
        → day of week (0-6)    : fraud patterns vary by day
        → day of month (0-30)  : fraud patterns vary by day
        → week number          : fraud patterns vary by week

    """
    print("\n[Feature 1] Creating time features from TransactionDT...")

    if "TransactionDT" not in df.columns:
        print("            TransactionDT not found, skipping...")
        return df

    # Reference start date
    # IEEE competition data starts around 2017
    START_DATE = pd.Timestamp("2017-01-01")

    # Convert elapsed seconds to actual datetime
    df["transaction_date"] = (
        START_DATE +
        pd.to_timedelta(df["TransactionDT"], unit="s")
    )

    # Extract time features
    df["tx_hour"]         = df["transaction_date"].dt.hour
    df["tx_day_of_week"]  = df["transaction_date"].dt.dayofweek
    df["tx_day_of_month"] = df["transaction_date"].dt.day
    df["tx_week"]         = df["transaction_date"].dt.isocalendar().week.astype(int)
    df["tx_month"]        = df["transaction_date"].dt.month

    # Is it a weekend? (5=Saturday, 6=Sunday)
    df["tx_is_weekend"] = (
        df["tx_day_of_week"] >= 5
    ).astype(int)

    # Is it night time? (0-6 AM)
    df["tx_is_night"] = (
        df["tx_hour"].between(0, 6)
    ).astype(int)

    # Drop the helper datetime column
    df = df.drop(columns=["transaction_date"])

    new_features = [
        "tx_hour",
        "tx_day_of_week",
        "tx_day_of_month",
        "tx_week",
        "tx_month",
        "tx_is_weekend",
        "tx_is_night"
    ]

    print(f"            Created : {new_features}")

    return df



# EMAIL DOMAIN FEATURES

def create_email_features(df):
    """
    Create features from email domains.

    From EDA:
    - Few domains dominate (gmail, yahoo, hotmail)
    - Many rare domains exist
    - Rare domains → group as 'other'
    - anonymous.com is interesting (privacy-conscious users)

    New features:
    1. P_email_domain_grouped : Top domains kept, rest = 'other'
    2. R_email_domain_grouped : Same for recipient
    3. email_match            : Do sender/receiver use same domain?
    4. P_is_anonymous         : Is sender using anonymous.com?
    5. R_is_anonymous         : Is receiver using anonymous.com?

    NOTE:
    - These are created BEFORE label encoding
    - So we work on original raw domain strings
    - But preprocessing already encoded them...
    - So we recreate from raw TransactionDT logic

    IMPORTANT:
    - This function works on PREPROCESSED df
    - email columns are already label encoded as integers
    - So we create binary flags instead

    """
    print("\n[Feature 2] Creating email domain features...")

    # Since email domains are already label encoded
    # We create aggregate/frequency features instead

    # Frequency encoding for P_emaildomain
    # How often does this email domain appear?
    if "P_emaildomain" in df.columns:
        p_freq = df["P_emaildomain"].value_counts()
        df["P_email_freq"] = df["P_emaildomain"].map(p_freq)
        print("            P_email_freq created")

    # Frequency encoding for R_emaildomain
    if "R_emaildomain" in df.columns:
        r_freq = df["R_emaildomain"].value_counts()
        df["R_email_freq"] = df["R_emaildomain"].map(r_freq)
        print("            R_email_freq created")

    # Do sender and receiver use same email domain?
    if "P_emaildomain" in df.columns and "R_emaildomain" in df.columns:
        df["email_domain_match"] = (
            df["P_emaildomain"] == df["R_emaildomain"]
        ).astype(int)
        print("            email_domain_match created")

    new_features = [
        "P_email_freq",
        "R_email_freq",
        "email_domain_match"
    ]

    print(f"            Created : {new_features}")

    return df



# CARD IDENTITY FEATURES

def create_card_features(df):
    """
    Create features from card-related columns.

    From EDA:
    - card1  : payment card ID
    - card2  : card numerical feature
    - card3  : card numerical feature
    - card5  : card numerical feature
    - card4  : card network (visa/mastercard) - encoded
    - card6  : card type (debit/credit) - encoded

    New features:
    1. card_identity  : card1 + card2 combination
                        proxy for unique card/user
    2. card1_freq     : how often card1 appears
                        high freq = regular user
                        low freq  = new/suspicious card
    3. card2_freq     : how often card2 appears
    
    """
    print("\n[Feature 3] Creating card features...")

    # Card identity - combination of card1 and card2
    # Creates a proxy for unique card holder
    if "card1" in df.columns and "card2" in df.columns:
        df["card_identity"] = (
            df["card1"].astype(str) +
            "_" +
            df["card2"].astype(str)
        )

        # Frequency of this card identity
        card_id_freq = df["card_identity"].value_counts()
        df["card_identity_freq"] = df["card_identity"].map(card_id_freq)

        # Drop string version (keep only frequency)
        df = df.drop(columns=["card_identity"])
        print("            card_identity_freq created")

    # Frequency of card1
    if "card1" in df.columns:
        card1_freq = df["card1"].value_counts()
        df["card1_freq"] = df["card1"].map(card1_freq)
        print("            card1_freq created")

    # Frequency of card2
    if "card2" in df.columns:
        card2_freq = df["card2"].value_counts()
        df["card2_freq"] = df["card2"].map(card2_freq)
        print("            card2_freq created")

    new_features = [
        "card_identity_freq",
        "card1_freq",
        "card2_freq"
    ]

    print(f"            Created : {new_features}")

    return df



# TRANSACTION AMOUNT FEATURES

def create_amount_features(df):
    """
    Create features based on transaction amount.

    From EDA:
    - TransactionAmt is log1p transformed (in preprocessing)
    - Mean = 135, Max = 31937, highly skewed
    - Fraud transactions have different amount patterns

    New features:
    1. amt_deviation_card1  : how much does this transaction
                              deviate from card1's mean amount?
                              Large deviation = suspicious

    2. amt_to_mean_ratio    : transaction amount vs overall mean
                              ratio > 1 means above average

    3. amt_decimal          : decimal part of original amount
                              .00, .99, .95 patterns in fraud

    NOTE:
    - TransactionAmt is already log1p transformed
    - We work on the transformed values

    """
    print("\n[Feature 4] Creating transaction amount features...")

    if "TransactionAmt" not in df.columns:
        print("            TransactionAmt not found, skipping...")
        return df

    # Amount deviation from card1 mean
    # How unusual is this transaction for this card?
    if "card1" in df.columns:
        card1_mean_amt = df.groupby("card1")["TransactionAmt"].transform("mean")
        card1_std_amt  = df.groupby("card1")["TransactionAmt"].transform("std")

        df["amt_deviation_card1"] = (
          (df["TransactionAmt"] - card1_mean_amt) /
          (card1_std_amt.fillna(0) + 1e-9)
        )
        print("            amt_deviation_card1 created")

    # Transaction amount vs overall mean
    overall_mean = df["TransactionAmt"].mean()
    df["amt_to_mean_ratio"] = df["TransactionAmt"] / (overall_mean + 1e-9)
    print("            amt_to_mean_ratio created")

    new_features = [
        "amt_deviation_card1",
        "amt_to_mean_ratio",
    ]

    print(f"            Created : {new_features}")

    return df



# FREQUENCY ENCODING FEATURES

def create_frequency_features(df):
    """
    Create frequency encoding for high cardinality columns.

    Frequency encoding:
    - Replace category with how often it appears
    - High frequency = common user/card (less suspicious)
    - Low frequency  = rare user/card   (more suspicious)

    From EDA:
    - addr1, addr2 : address features
    - DeviceInfo   : many unique device strings

    Args:
        df : Input DataFrame

    Returns:
        df : DataFrame with frequency encoded features
    """
    print("\n[Feature 5] Creating frequency encoding features...")

    # Columns to frequency encode
    freq_cols = [
        "addr1",
        "addr2",
        "DeviceInfo"
    ]

    # Only encode columns that exist
    freq_cols = [col for col in freq_cols if col in df.columns]

    for col in freq_cols:
        freq_map = df[col].value_counts()
        df[f"{col}_freq"] = df[col].map(freq_map)
        print(f"            {col}_freq created")

    new_features = [f"{col}_freq" for col in freq_cols]
    print(f"            Created : {new_features}")

    return df



# AGGREGATION FEATURES

def create_aggregation_features(df):
    """
    Create aggregation features by grouping.

    These capture behavioral patterns:
    - How many transactions from same card?
    - What is average amount for this card?
    - What is max amount for this card?

    Aggregations:
    1. Group by card1 → count, mean, std, max of amount
    2. Group by addr1 → count, mean of amount

    """
    print("\n[Feature 6] Creating aggregation features...")

    # Group by card1
    if "card1" in df.columns and "TransactionAmt" in df.columns:

        card1_agg = df.groupby("card1")["TransactionAmt"].agg([
            "count",
            "mean",
            "std",
            "max",
            "min"
        ]).reset_index()

        # Replace NaN standard deviation
        card1_agg["std"] = card1_agg["std"].fillna(0)

        card1_agg.columns = [
            "card1",
            "card1_tx_count",
            "card1_amt_mean",
            "card1_amt_std",
            "card1_amt_max",
            "card1_amt_min"
        ]

        df = df.merge(card1_agg, on="card1", how="left")
        print("            card1 aggregations created")

    # Group by addr1
    if "addr1" in df.columns and "TransactionAmt" in df.columns:

        addr1_agg = df.groupby("addr1")["TransactionAmt"].agg([
            "count", "mean"
        ]).reset_index()

        addr1_agg.columns = [
            "addr1",
            "addr1_tx_count",
            "addr1_amt_mean"
        ]

        df = df.merge(addr1_agg, on="addr1", how="left")
        print("            addr1 aggregations created")

    new_features = [
        "card1_tx_count",
        "card1_amt_mean",
        "card1_amt_std",
        "card1_amt_max",
        "card1_amt_min",
        "addr1_tx_count",
        "addr1_amt_mean"
    ]

    print(f"            Created : {new_features}")

    return df



# SAVE ENGINEERED DATA

def save_engineered_data(df, filename="train_engineered.parquet"):
    """
    Save the feature engineered DataFrame.
    
    """
    ENGINEERED_DIR.mkdir(parents=True, exist_ok=True)

    output_path = ENGINEERED_DIR / filename
    df.to_parquet(output_path, index=False)

    print(f"\nEngineered data saved")
    print(f"Path  : {output_path}")
    print(f"Shape : {df.shape}")


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def engineer_features(df=None, save=True):
    """
    Full feature engineering pipeline.

    Order of operations:
    1. Load preprocessed data (if df not provided)
    2. Create time features
    3. Create email features
    4. Create card features
    5. Create amount features
    6. Create frequency features
    7. Create aggregation features
    8. Save engineered data

    Args:
        df   : Preprocessed DataFrame
               If None, loads from data/preprocessed/
        save : If True, save output to CSV

    Returns:
        df   : Feature engineered DataFrame
    """
    print("=" * 60)
    print("       FEATURE ENGINEERING PIPELINE")
    print("=" * 60)

    # ── Step 1: Load Data ──────────────────────────
    if df is None:
        print(f"\n[Step 1] Loading preprocessed data...")
        input_path = PREPROCESSED_DIR / "train_preprocessed.parquet"
        df = pd.read_parquet(input_path)
        print(f"         Shape after loading  : {df.shape}")
    else:
        print(f"\n[Step 1] Using provided DataFrame")
        print(f"         Shape               : {df.shape}")

    # Track original columns
    original_cols = df.shape[1]

    # ── Step 2: Time Features ──────────────────────
    print(f"\n[Step 2] Time features...")
    df = create_time_features(df)

    # ── Step 3: Email Features ─────────────────────
    print(f"\n[Step 3] Email features...")
    df = create_email_features(df)

    # ── Step 4: Card Features ──────────────────────
    print(f"\n[Step 4] Card features...")
    df = create_card_features(df)

    # ── Step 5: Amount Features ────────────────────
    print(f"\n[Step 5] Amount features...")
    df = create_amount_features(df)

    # ── Step 6: Frequency Features ─────────────────
    print(f"\n[Step 6] Frequency features...")
    df = create_frequency_features(df)

    # ── Step 7: Aggregation Features ───────────────
    print(f"\n[Step 7] Aggregation features...")
    df = create_aggregation_features(df)

    # ── Step 8: Save ───────────────────────────────
    if save:
        print(f"\n[Step 8] Saving engineered data...")
        save_engineered_data(df)

    # Summary
    new_cols = df.shape[1] - original_cols

    print("\n" + "=" * 60)
    print("       FEATURE ENGINEERING COMPLETE")
    print(f"       Original features  : {original_cols}")
    print(f"       New features added : {new_cols}")
    print(f"       Final shape        : {df.shape}")
    print("=" * 60)

    return df


# RUN

if __name__ == "__main__":

    df = engineer_features(df=None, save=True)

    print("\n── New features sample ──")
    new_feature_cols = [
        "tx_hour",
        "tx_day_of_week",
        "tx_is_weekend",
        "tx_is_night",
        "P_email_freq",
        "R_email_freq",
        "email_domain_match",
        "card1_freq",
        "card_identity_freq",
        "amt_deviation_card1",
        "amt_to_mean_ratio",
        "addr1_freq",
        "card1_tx_count",
        "card1_amt_mean",
    ]

    # Only show columns that exist
    show_cols = [
        col for col in new_feature_cols
        if col in df.columns
    ]

    print(df[show_cols].head(10))

    print("\n── Data types ──")
    print(df.dtypes.value_counts())

    print("\n── Target distribution ──")
    print(df[TARGET].value_counts())

    print("\n── Missing values after engineering ──")
    missing = df.isnull().sum()
    print(missing[missing > 0])