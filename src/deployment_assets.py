# src/deployment_assets.py

import pickle
import json
from pathlib import Path
import sys

import pandas as pd


# PROJECT ROOT

project_root = Path(__file__).resolve().parent.parent

sys.path.append(str(project_root))


# CONFIGURATION

TARGET = "isFraud"

DATA_PATH = (
    project_root
    / "data"
    / "preprocessed"
    / "train_selected.parquet"
)

MODEL_DIR = project_root / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# LOAD DATASET

def load_dataset():
    """
    Load feature-selected training dataset.
    """

    print("=" * 60)
    print("LOADING DATASET")
    print("=" * 60)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_parquet(DATA_PATH)

    print(f"Shape : {df.shape}")

    return df



# SPLIT FEATURES

def get_features(df):
    """
    Remove target column and return features.
    """

    X = df.drop(
        columns=[TARGET]
    )

    print(f"Feature shape : {X.shape}")

    return X



# SAVE FEATURE ORDER

def save_feature_order(X):
    """
    Save exact feature order expected by model.
    """

    feature_order = X.columns.tolist()

    output_path = MODEL_DIR / "feature_order.pkl"

    with open(output_path, "wb") as f:
        pickle.dump(
            feature_order,
            f
        )

    print("\nFeature order saved.")
    print(output_path)



# SAVE DEFAULT VALUES

def save_default_values(X):
    """
    Save default values for missing deployment inputs.

    Numeric:
        Mean

    Categorical:
        Mode
    """

    defaults = {}


    numeric_columns = X.select_dtypes(
        include="number"
    ).columns


    categorical_columns = X.select_dtypes(
        exclude="number"
    ).columns


    # Numeric defaults
    for col in numeric_columns:

        defaults[col] = float(
            X[col].mean()
        )


    # Categorical defaults
    for col in categorical_columns:

        defaults[col] = (
            X[col]
            .mode()[0]
        )


    output_path = MODEL_DIR / "default_values.pkl"


    with open(output_path, "wb") as f:
        pickle.dump(
            defaults,
            f
        )


    print("\nDefault values saved.")
    print(output_path)



# SAVE FEATURE MAPPING

def save_feature_mapping():
    """
    Create categorical feature mappings
    from label encoders.
    """

    encoder_path = MODEL_DIR / "label_encoders.pkl"


    if not encoder_path.exists():

        raise FileNotFoundError(
            f"Missing file: {encoder_path}"
        )


    with open(encoder_path, "rb") as f:

        label_encoders = pickle.load(f)



    feature_mapping = {}


    for feature, encoder in label_encoders.items():

        feature_mapping[feature] = {

            "classes":
                encoder.classes_.tolist(),


            "mapping":
                {
                    value: int(index)
                    for index, value
                    in enumerate(
                        encoder.classes_
                    )
                }
        }



    output_path = (
        MODEL_DIR
        / "feature_mapping.json"
    )


    with open(output_path, "w") as f:

        json.dump(
            feature_mapping,
            f,
            indent=4
        )


    print("\nFeature mapping saved.")
    print(output_path)



# SAVE DEPLOYMENT REPORT

def save_report(X):
    """
    Save deployment feature report.
    """

    report = pd.DataFrame({

        "Feature":
            X.columns,


        "Data Type":
            X.dtypes.astype(str)

    })


    output_path = (
        MODEL_DIR
        / "deployment_features.csv"
    )


    report.to_csv(
        output_path,
        index=False
    )


    print("\nDeployment feature report saved.")
    print(output_path)



# MAIN PIPELINE

def create_deployment_assets():

    print("=" * 60)
    print("CREATING DEPLOYMENT ASSETS")
    print("=" * 60)


    df = load_dataset()


    X = get_features(df)


    save_feature_order(X)


    save_default_values(X)


    save_feature_mapping()


    save_report(X)



    print("\n" + "=" * 60)
    print("DEPLOYMENT ASSETS CREATED SUCCESSFULLY")
    print("=" * 60)



# RUN

if __name__ == "__main__":

    create_deployment_assets()