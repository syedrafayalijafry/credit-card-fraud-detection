# src/threshold_tuning.py

import pickle
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report
)


# PATHS

PROJECT_ROOT = Path.cwd()

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "preprocessed"
    / "train_selected.parquet"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "lightgbm_tuned.pkl"
)



# LOAD MODEL AND DATA

def load_assets():

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    df = pd.read_parquet(DATA_PATH)

    return model, df



# FIND BEST THRESHOLD

def find_best_threshold(
        y_true,
        y_prob
):

    thresholds = np.arange(
        0.05,
        0.95,
        0.01
    )


    results = []


    for threshold in thresholds:

        y_pred = (
            y_prob >= threshold
        ).astype(int)


        results.append({

            "threshold": threshold,

            "precision":
                precision_score(
                    y_true,
                    y_pred,
                    zero_division=0
                ),

            "recall":
                recall_score(
                    y_true,
                    y_pred,
                    zero_division=0
                ),

            "f1":
                f1_score(
                    y_true,
                    y_pred,
                    zero_division=0
                )
        })


    results_df = pd.DataFrame(results)


    best = results_df.loc[
        results_df["f1"].idxmax()
    ]


    return best, results_df



# MAIN

def main():

    model, df = load_assets()


    X = df.drop(
        columns=["isFraud"]
    )

    y = df["isFraud"]


    print("Generating probabilities...")


    y_prob = model.predict_proba(X)[:,1]


    print(
        "ROC-AUC:",
        roc_auc_score(
            y,
            y_prob
        )
    )


    best, results = find_best_threshold(
        y,
        y_prob
    )


    print("\nBest Threshold:")
    print(best)


    print("\nClassification Report:")


    y_pred = (
        y_prob >= best["threshold"]
    ).astype(int)


    print(
        classification_report(
            y,
            y_pred
        )
    )


if __name__ == "__main__":
    main()