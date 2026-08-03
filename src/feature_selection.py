# src/feature_selection.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif

import sys
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))



# CONFIGURATION

TARGET = "isFraud"

# Number of top features to keep
TOP_K = 70

# Train/Test split configuration
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Engineered data path
ENGINEERED_DIR = project_root / "data" / "preprocessed"

# Output path
SELECTED_DIR = project_root / "data" / "preprocessed"




# MUTUAL INFORMATION

def calculate_mutual_information(X_train, y_train, random_state=RANDOM_STATE):
    """
    Calculate Mutual Information (MI) score for every feature.

    Mutual Information measures how much information a feature
    provides about the target variable.

    Higher MI score  → More informative feature
    Lower MI score   → Less informative feature

    """

    print("\n[Feature 1] Calculating Mutual Information...")

    
    # Sample data to reduce memory usage
    MI_SAMPLE_SIZE = 100000


    if len(X_train) > MI_SAMPLE_SIZE:

        print(
            f"            Sampling {MI_SAMPLE_SIZE} rows for MI calculation..."
        )

        sample_indices = X_train.sample(
            n=MI_SAMPLE_SIZE,
            random_state=random_state
        ).index

        X_mi = X_train.loc[sample_indices]
        y_mi = y_train.loc[sample_indices]

    else:

        X_mi = X_train
        y_mi = y_train



    # Calculate MI

    mi_scores = mutual_info_classif(
        X=X_mi,
        y=y_mi,
        random_state=random_state,
        n_jobs=2
    )

    mi_df = pd.DataFrame({
        "Feature": X_train.columns,
        "MI Score": mi_scores
    })

    mi_df = (
        mi_df
        .sort_values(
            by="MI Score",
            ascending=False
        )
        .reset_index(drop=True)
    )

    print(f"            Features evaluated : {len(mi_df)}")

    print(
        f"            Highest MI Score   : "
        f"{mi_df.iloc[0]['MI Score']:.6f}"
    )

    print(
        f"            Top Feature        : "
        f"{mi_df.iloc[0]['Feature']}"
    )

    return mi_df




# PLOT MUTUAL INFORMATION

def plot_mutual_information(mi_df, top_n=30, figsize=(10, 8)):
    """
    Plot the top N features ranked by Mutual Information.

    """

    print(f"\n[Feature 2] Plotting Top {top_n} Features...")

    if mi_df.empty:
        raise ValueError("Mutual Information DataFrame is empty.")

    top_features = mi_df.head(top_n)

    plt.figure(figsize=figsize)

    plt.barh(
        top_features["Feature"],
        top_features["MI Score"]
    )

    plt.gca().invert_yaxis()

    plt.xlabel("Mutual Information Score")
    plt.ylabel("Features")

    plt.title(f"Top {top_n} Features by Mutual Information")

    plt.tight_layout()

    plt.show()

    print("            Plot generated successfully.")
    



# SELECT TOP FEATURES

def select_top_features(mi_df, top_k=TOP_K):
    """
    Select top K features based on Mutual Information score.

    """

    print(f"\n[Feature 3] Selecting Top {top_k} Features...")

    if mi_df.empty:
        raise ValueError(
            "Mutual Information DataFrame is empty."
        )

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

    if top_k > len(mi_df):
        print(
            f"            Requested {top_k} features "
            f"but only {len(mi_df)} available."
        )

        top_k = len(mi_df)


    selected_features = (
        mi_df
        .head(top_k)["Feature"]
        .tolist()
    )


    print(
        f"            Features selected : "
        f"{len(selected_features)}"
    )

    print(
        f"            First 5 features  : "
        f"{selected_features[:5]}"
    )


    return selected_features




# ADD MANDATORY FEATURES

def add_mandatory_features(selected_features, mandatory_features):
    """
    Add mandatory deployment features to selected features.

    Mutual Information selects features based on statistical
    importance, but deployment may require some features
    regardless of their MI score.
=
    """

    print("\n[Feature 4] Adding mandatory features...")


    if not selected_features:
        raise ValueError(
            "Selected feature list is empty."
        )


    if not mandatory_features:
        print(
            "            No mandatory features provided."
        )

        return selected_features


    # Combine lists and remove duplicates
    final_features = list(
        dict.fromkeys(
            mandatory_features + selected_features
        )
    )


    added_features = (
        len(final_features)
        -
        len(selected_features)
    )


    print(
        f"            Mandatory features added : "
        f"{added_features}"
    )

    print(
        f"            Final feature count       : "
        f"{len(final_features)}"
    )


    return final_features



# CREATE SELECTED DATASET

def create_selected_dataset(df, selected_features, target=TARGET):
    """
    Create final dataset containing only selected features.

    """

    print("\n[Feature 5] Creating selected dataset...")


    if df.empty:
        raise ValueError(
            "Input dataframe is empty."
        )


    if not selected_features:
        raise ValueError(
            "Selected feature list is empty."
        )


    # Check which selected features actually exist
    existing_features = [
        feature
        for feature in selected_features
        if feature in df.columns
    ]


    missing_features = [
        feature
        for feature in selected_features
        if feature not in df.columns
    ]


    if missing_features:
        print(
            f"            Missing features skipped : "
            f"{len(missing_features)}"
        )


    # Create final feature list
    final_columns = existing_features.copy()


    # Add target column
    if target in df.columns:
        final_columns.append(target)
    else:
        raise ValueError(
            f"Target column '{target}' not found."
        )


    selected_df = df[final_columns].copy()


    print(
        f"            Original shape : {df.shape}"
    )

    print(
        f"            Selected shape : {selected_df.shape}"
    )

    print(
        f"            Features kept  : "
        f"{len(existing_features)}"
    )


    return selected_df



# SAVE SELECTED DATASET

def save_selected_dataset(
    df,
    filename="train_selected.parquet"
):
    """
    Save the feature-selected dataset.

    """

    print("\n[Feature 6] Saving selected dataset...")


    # Create directory if not exists
    SELECTED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    output_path = SELECTED_DIR / filename


    df.to_parquet(
        output_path,
        index=False
    )


    print("\n            Selected data saved")
    print(f"            Path  : {output_path}")
    print(f"            Shape : {df.shape}")
    
    



# MAIN PIPELINE

def feature_selection_pipeline(
    df=None,
    top_k=TOP_K,
    mandatory_features=None,
    save=True
):
    """
    Complete feature selection pipeline.

    Steps:
    1. Load engineered data (if df not provided)
    2. Separate features and target
    3. Train/Test split
    4. Calculate Mutual Information
    5. Select top features
    6. Add mandatory deployment features
    7. Create selected dataset
    8. Save selected dataset

    """

    print("=" * 60)
    print("       FEATURE SELECTION PIPELINE")
    print("=" * 60)


    # Step 1: Load Data

    if df is None:

        print("\n[Step 1] Loading engineered data...")

        input_path = (
            ENGINEERED_DIR /
            "train_engineered.parquet"
        )

        df = pd.read_parquet(input_path)

        print(
            f"            Shape : {df.shape}"
        )

    else:

        print("\n[Step 1] Using provided dataframe")

        print(
            f"            Shape : {df.shape}"
        )


    # Step 2: Split Features and Target

    print("\n[Step 2] Separating features and target...")


    X = df.drop(
        columns=[TARGET]
    )

    y = df[TARGET]


    print(
        f"            Features : {X.shape[1]}"
    )


    # Step 3: Train/Test Split

    print("\n[Step 3] Performing train-test split...")


    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE,

        stratify=y
    )


    print(
        f"            Training size : {X_train.shape}"
    )

    print(
        f"            Testing size  : {X_test.shape}"
    )


    # Step 4: Mutual Information

    mi_df = calculate_mutual_information(
        X_train,
        y_train
    )


    # Step 5: Select Top Features

    selected_features = select_top_features(
        mi_df,
        top_k=top_k
    )


    # Step 6: Add Mandatory Features

    if mandatory_features is not None:

        selected_features = add_mandatory_features(

            selected_features,

            mandatory_features
        )


    # Step 7: Create Selected Dataset

    selected_df = create_selected_dataset(

        df,

        selected_features

    )


    # Step 8: Save

    if save:

        save_selected_dataset(
            selected_df
        )


    # Summary

    print("\n" + "=" * 60)
    print("       FEATURE SELECTION COMPLETE")
    print(
        f"       Original features : {X.shape[1]}"
    )
    print(
        f"       Selected features : {len(selected_features)}"
    )
    print(
        f"       Final shape       : {selected_df.shape}"
    )
    print("=" * 60)


    return selected_df, mi_df





# RUN

if __name__ == "__main__":

    # Features that should always remain
    # because they are realistic for deployment

    mandatory_features = [

        "TransactionAmt",

        "ProductCD",

        "card1",

        "card4",

        "card6",

        "DeviceType",

        "tx_hour",

        "tx_day_of_week",

        "tx_is_weekend",

        "tx_is_night"

    ]


    selected_df, mi_df = feature_selection_pipeline(df=None, top_k=TOP_K, mandatory_features=mandatory_features, save=True)


    print("\n── Top 20 Features by Mutual Information ──")

    print(
        mi_df.head(20)
    )


    print("\n── Selected Dataset Sample ──")

    print(
        selected_df.head()
    )


    print("\n── Data Types ──")

    print(
        selected_df.dtypes.value_counts()
    )


    print("\n── Target Distribution ──")

    print(
        selected_df[TARGET].value_counts()
    )


    print("\n── Missing Values ──")

    missing = selected_df.isnull().sum()

    print(
        missing[missing > 0]
    )