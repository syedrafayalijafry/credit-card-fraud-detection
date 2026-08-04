# src/predict.py

from datetime import datetime
import pickle
import json
from pathlib import Path
import sys

import pandas as pd
from pandas.api.types import is_string_dtype


# PROJECT ROOT

project_root = Path(__file__).resolve().parent.parent
# PATHS

MODEL_DIR = project_root / "models"


MODEL_PATH = MODEL_DIR / "lightgbm_tuned.pkl"
FEATURE_ORDER_PATH = MODEL_DIR / "feature_order.pkl"
DEFAULT_VALUES_PATH = MODEL_DIR / "default_values.pkl"
MAPPING_PATH = MODEL_DIR / "feature_mapping.json"


# THRESHOLD
BEST_THRESHOLD = 0.78


# LOAD ASSETS

def load_assets():

    print("Loading deployment assets...")


    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)


    with open(FEATURE_ORDER_PATH, "rb") as f:
        feature_order = pickle.load(f)


    with open(DEFAULT_VALUES_PATH, "rb") as f:
        default_values = pickle.load(f)


    with open(MAPPING_PATH, "r") as f:
        feature_mapping = json.load(f)


    print("Assets loaded successfully.")


    return (
        model,
        feature_order,
        default_values,
        feature_mapping
    )



# PREPROCESS INPUT

def preprocess_input(
        data,
        feature_order,
        default_values,
        feature_mapping
):
    """
    Prepare model input.
    Supports raw categorical inputs and encoded inputs.
    """

    df = pd.DataFrame([data])


    # Add missing features

    for feature in feature_order:

        if feature not in df.columns:

            df[feature] = default_values[feature]


    # Keep only required columns

    df = df[feature_order]


    # Encode only string/object columns

    for feature in feature_mapping:

        if feature in df.columns:

            if is_string_dtype(df[feature]):

                mapping = (
                    feature_mapping[feature]["mapping"]
                )

                df[feature] = (
                    df[feature]
                    .map(mapping)
                    .fillna(-1)
                )


    return df



# PREDICTION

def predict_fraud(transaction):

    # Create time-based features

    transaction_date = transaction.pop("transaction_date")
    transaction_time = transaction.pop("transaction_time")

    dt = datetime.strptime(
        f"{transaction_date} {transaction_time}",
        "%Y-%m-%d %H:%M"
    )

    transaction["tx_hour"] = dt.hour

    transaction["tx_day_of_week"] = dt.weekday()

    transaction["tx_is_weekend"] = int(
        dt.weekday() >= 5
    )

    transaction["tx_is_night"] = int(
        dt.hour < 6 or dt.hour >= 22
    )

    (
        model,
        feature_order,
        default_values,
        feature_mapping

    ) = load_assets()

    X = preprocess_input(
        transaction,
        feature_order,
        default_values,
        feature_mapping
    )

    probability = model.predict_proba(X)[0][1]

    prediction = (
        1
        if probability >= BEST_THRESHOLD
        else 0
    )

    if probability >= 0.90 or probability <= 0.10:
        confidence = "High"

    elif probability >= 0.75 or probability <= 0.25:
        confidence = "Medium"

    else:
        confidence = "Low"

    return {
        "fraud_probability": round(probability, 4),
        "threshold": BEST_THRESHOLD,
        "prediction": prediction,
        "confidence": confidence
    }



# SAVE THRESHOLD

def save_threshold():
    """
    Save optimal classification threshold.
    """

    output_path = MODEL_DIR / "threshold.pkl"

    with open(output_path, "wb") as f:
        pickle.dump(
            BEST_THRESHOLD,
            f
        )

    print("\nThreshold saved.")
    print(output_path)


# TEST

if __name__ == "__main__":


    sample_transaction = {

        "TransactionAmt": 100.0,

        "transaction_date": "2026-08-05",

        "transaction_time": "14:30",

        "ProductCD": "W",

        "card4": "visa",

        "card6": "debit",

        "DeviceType": "desktop"

        }


    result = predict_fraud(
            sample_transaction
        )


    print("\nPrediction Result:")
    print(result)
    
    save_threshold()