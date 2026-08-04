# tests/test_prediction.py

import sys
from pathlib import Path
import pandas as pd

from sklearn.metrics import (
    classification_report,
    roc_auc_score
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(
    str(PROJECT_ROOT)
)


from src.predict import predict_fraud



DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "preprocessed"
    / "train_selected.parquet"
)



def main():

    df = pd.read_parquet(DATA_PATH)


    samples = df.sample(
        1000,
        random_state=42
    )


    results = []


    for _, row in samples.iterrows():

        actual = row["isFraud"]


        transaction = (
            row
            .drop("isFraud")
            .to_dict()
        )


        prediction = predict_fraud(
            transaction
        )


        results.append({

            "Actual":
                actual,

            "Probability":
                prediction["fraud_probability"],

            "Prediction":
                prediction["prediction"]

        })


    results_df = pd.DataFrame(results)


    print(results_df)


    print("\nAccuracy:")
    print(
        (
            results_df["Actual"]
            ==
            results_df["Prediction"]
        )
        .mean()
    )

    
    print(
        classification_report(
            results_df["Actual"],
            results_df["Prediction"]
        )
    )

    
    print(
    "ROC-AUC:",
        roc_auc_score(
            results_df["Actual"],
            results_df["Probability"]
        )
    )

if __name__ == "__main__":
    main()