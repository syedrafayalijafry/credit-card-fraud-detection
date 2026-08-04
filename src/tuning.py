import numpy as np
import pandas as pd

import pickle
import json

from pathlib import Path
import sys

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV
)
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
import lightgbm as lgb

# Add project root
project_root = Path(__file__).resolve().parent.parent

sys.path.append(
    str(project_root)
)



# CONFIGURATION

TARGET = "isFraud"


RANDOM_STATE = 42


TEST_SIZE = 0.2


DATA_PATH = (project_root / "data" / "preprocessed" / "train_selected.parquet")


MODEL_DIR = (project_root / "models")

RESULTS_DIR = (project_root / "results")



# LOAD DATA

def load_training_data(path=DATA_PATH):
    """
    Load feature-selected dataset.

    Dataset contains:
    - Selected features
    - Target column (isFraud)

    """

    print("\nLoading training data...")

    df = pd.read_parquet(path)

    print(f"            Shape : {df.shape}")

    return df



# SPLIT FEATURES AND TARGET

def split_data(df):
    """
    Separate features and target,
    then create train/test split.

    Uses stratify because fraud dataset
    is highly imbalanced.

    """

    print("\nSplitting dataset...")

    X = df.drop(columns=[TARGET])

    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"Train : {X_train.shape}")
    print(f"Test  : {X_test.shape}")

    return X_train, X_test, y_train, y_test



# RANDOMIZED SEARCH

def tune_lightgbm(X_train, y_train):
    """
    Tune LightGBM using RandomizedSearchCV.

    """

    print("\n" + "=" * 60)
    print("LIGHTGBM HYPERPARAMETER TUNING")
    print("=" * 60)


    model = lgb.LGBMClassifier(
        objective="binary",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )


    # Hyperparameter search space
    param_dist = {
        "n_estimators": [200, 300, 500, 700],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "num_leaves": [31, 63, 127],
        "max_depth": [-1, 10, 15, 20],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "min_child_samples": [10, 20, 30,50]
    }


    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=15,
        scoring="roc_auc",
        cv=4,
        verbose=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True
    )


    random_search.fit(
        X_train,
        y_train
    )


    print("\nBest ROC-AUC :")

    print(random_search.best_score_)


    print("\nBest Parameters :")

    print(random_search.best_params_)


    return (
        random_search.best_estimator_,
        random_search.best_params_
    )



# EVALUATE TUNED MODEL

def evaluate_model(model, X_test, y_test):
    """
    Evaluate tuned LightGBM model.

    """

    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    # Predictions
    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    roc_auc = roc_auc_score(y_test, y_prob)

    precision = precision_score(y_test, y_pred)

    recall = recall_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred
    )

    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("\nClassification Report\n")

    print(classification_report(y_test, y_pred))

    results = {
        "Model": "LightGBM (Tuned)",
        "ROC-AUC": roc_auc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }

    return results



# SAVE MODEL

def save_model(
    model,
    filename="lightgbm_tuned.pkl"
):
    """
    Save tuned LightGBM model.
    
    """

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = MODEL_DIR / filename

    with open(model_path, "wb") as file:
        pickle.dump(model, file)

    print("\nModel saved successfully!")

    print(f"Location : {model_path}")
    


# SAVE BEST PARAMETERS

def save_best_parameters(params, filename="lightgbm_best_params.json"):
    """
    Save best hyperparameters.
    """

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = RESULTS_DIR / filename

    with open(output_path, "w") as file:

        json.dump(
            params,
            file,
            indent=4
        )

    print("\nBest parameters saved!")

    print(output_path)
    
    

# HYPERPARAMETER TUNING PIPELINE
# ─────────────────────────────────

def tuning_pipeline(df=None):
    """
    Complete LightGBM hyperparameter tuning pipeline.

    Steps:
    1. Load selected dataset
    2. Split data
    3. Tune LightGBM
    4. Evaluate tuned model
    5. Save tuned model
    6. Save best parameters

    """

    print("=" * 60)
    print("LIGHTGBM HYPERPARAMETER TUNING")
    print("=" * 60)

    # Step 1
    if df is None:
        df = load_training_data()

    # Step 2
    X_train, X_test, y_train, y_test = split_data(df)

    # Step 3
    model, best_params = tune_lightgbm(X_train, y_train)

    # Step 4
    results = evaluate_model(model, X_test, y_test)

    # Step 5
    save_model(model)

    # Step 6
    save_best_parameters(best_params)

    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING COMPLETED")
    print("=" * 60)

    return model, results, best_params




if __name__ == "__main__":

    model, results, best_params = tuning_pipeline()

    print("\nFinal Results\n")

    for key, value in results.items():

        if isinstance(value, float):
            print(f"{key:<12}: {value:.4f}")
        else:
            print(f"{key:<12}: {value}")

    print("\nBest Parameters\n")

    for key, value in best_params.items():
        print(f"{key:<20}: {value}")