import numpy as np
import pandas as pd

import pickle
import json

from pathlib import Path
import sys

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import(
    RandomForestClassifier,
    HistGradientBoostingClassifier
)
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

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



# LOAD DATA

def load_training_data(path=DATA_PATH):
    """
    Load feature-selected dataset.

    Dataset contains:
    - Selected features
    - Target column (isFraud)

    """

    print("\n[Step 1] Loading training data...")

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

    print("\n[Step 2] Splitting data...")


    # Features
    X = df.drop(
        columns=[TARGET]
    )


    # Target
    y = df[TARGET]


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )


    print(
        f"            Training data : {X_train.shape}"
    )

    print(
        f"            Testing data  : {X_test.shape}"
    )


    print("\n            Class distribution:")

    print(
        y_train.value_counts()
    )


    return (
        X_train,
        X_test,
        y_train,
        y_test
    )



# MODEL EVALUATION

def evaluate_model(model, X_test, y_test, model_name):
    """
    
    Evaluate trained model.

    Metrics:
        - ROC-AUC
        - Precision
        - Recall
        - F1 Score

    """
    
    print("\n" + "=" * 60)
    print(f"Evaluating : {model_name}")
    print("=" * 60)

    # Predictions
    y_pred = model.predict(X_test)
    
    # Probability predictions
    # Needed for ROC-AUC

    if hasattr(model, "predict_proba"):

        y_prob = model.predict_proba(X_test)[:, 1]

    else:

        y_prob = model.decision_function(X_test)
        
    # Metrics

    roc_auc = roc_auc_score(y_test,  y_prob)
    precision = precision_score(y_test, y_pred,zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print("\nClassification Report:")
    print(
        classification_report(y_test, y_pred, zero_division=0)
    )

    return {
        "Model": model_name,
        "ROC-AUC": roc_auc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }
    


# LOGISTIC REGRESSION MODEL

def train_logistic_regression(X_train, y_train):
    """
    Train Logistic Regression model.

    Uses:
        - StandardScaler
        - Class balancing

    """
    print("\n[Model 1] Training Logistic Regression...")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier",LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )

    ])


    model.fit(X_train, y_train)

    print("            Logistic Regression trained")

    return model



# DECISION TREE MODEL

def train_decision_tree(
    X_train,
    y_train
):
    """
    Train Decision Tree classifier.

    Uses:
        - max_depth to reduce overfitting
        - class_weight for fraud imbalance

    """

    print("\n[Model 2] Training Decision Tree...")

    model = DecisionTreeClassifier(max_depth=10, class_weight="balanced", random_state=RANDOM_STATE)

    model.fit(X_train, y_train)

    print("            Decision Tree trained")

    return model




# RANDOM FOREST MODEL

def train_random_forest(
    X_train,
    y_train
):
    """
    Train Random Forest classifier.

    Uses:
        - Multiple decision trees
        - Class balancing
        - Parallel processing

    """

    print("\n[Model 3] Training Random Forest...")


    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE
    )


    model.fit(X_train, y_train)

    print("            Random Forest trained")

    return model




# XGBOOST MODEL

def train_xgboost(X_train, y_train):
    """
    Train XGBoost classifier.

    Handles class imbalance using
    scale_pos_weight.

    """

    print("\n[Model 4] Training XGBoost...")


    # Calculate imbalance ratio

    negative = (y_train == 0).sum()
    positive = (y_train == 1).sum()

    scale_weight = negative / positive

    print(f"            scale_pos_weight : {scale_weight:.2f}")


    model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_weight,
        eval_metric="auc",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )


    model.fit(X_train, y_train)


    print("            XGBoost trained")

    return model 




# LIGHTGBM MODEL

def train_lightgbm(X_train, y_train):
    """
    Train LightGBM classifier.

    Handles:
    - large tabular data
    - class imbalance

    """

    print("\n[Model 5] Training LightGBM...")


    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    print("            LightGBM trained")


    return model



# CATBOOST MODEL
def train_catboost(X_train, y_train):
    """
    Train CatBoost classifier.

    Handles class imbalance automatically.

    """

    print("\n[Model 6] Training CatBoost...")


    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_seed=RANDOM_STATE,
        verbose=100
    )

    model.fit(X_train, y_train)

    print("            CatBoost trained")

    return model



# MODEL COMPARISON

def compare_models(results):
    """
    Compare all trained models.

    """

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="ROC-AUC",
        ascending=False
    ).reset_index(drop=True)

    print(results_df)

    return results_df



# SAVE THE BEST MODEL

def save_best_model(models, results_df):
    """
    Save the best performing model.
    
    """
    
    print("\n" + "=" * 60)
    print("SAVING BEST MODEL")
    print("=" * 60)

    # Create models directory
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Best model name
    best_model_name = results_df.iloc[0]["Model"]

    print(f"Best Model : {best_model_name}")

    # Retrieve trained model
    best_model = models[best_model_name]

    # Save model
    model_path = MODEL_DIR / "best_model.pkl"

    with open(model_path, "wb") as file:
        pickle.dump(best_model, file)

    print(f"Model saved : {model_path}")

    # Save comparison results
    results_path = MODEL_DIR / "model_results.csv"

    results_df.to_csv(results_path, index=False
    )

    print(f"Results saved : {results_path}")

    # Save metadata
    metadata = {

        "best_model": best_model_name,

        "roc_auc":
            float(results_df.iloc[0]["ROC-AUC"]),

        "precision":
            float(results_df.iloc[0]["Precision"]),

        "recall":
            float(results_df.iloc[0]["Recall"]),

        "f1_score":
            float(results_df.iloc[0]["F1 Score"])

    }

    metadata_path = MODEL_DIR / "model_metadata.json"

    with open(
        metadata_path,
        "w"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    print(f"Metadata saved : {metadata_path}")
    


# TRAINING PIPELINE

def train_pipeline():
    """
    
    Complete model training pipeline.

    Steps:
    1. Load selected dataset
    2. Split train/test
    3. Train all models
    4. Evaluate all models
    5. Compare results
    6. Save best model
    
    """
    
    print("=" * 60)
    print("        MODEL TRAINING PIPELINE")
    print("=" * 60)


    # Step 1
    print("\n[Step 1] Loading selected dataset...")

    df = load_training_data()


    # Step 2
    print("\n[Step 2] Splitting dataset...")

    X_train, X_test, y_train, y_test = split_data(df)


    # Store models & results
    models = {}
    results = []
    
    
    # Logistic Regression
    
    lr_model = train_logistic_regression(X_train, y_train)
    models['Logistic Regression'] = lr_model
    lr_result = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    results.append(lr_result)
    
    
    # Decision Tree
    
    dt_model = train_decision_tree(X_train, y_train)
    models['Decision Tree'] = dt_model
    dt_result = evaluate_model(dt_model, X_test, y_test, "Decision Tree")
    results.append(dt_model)
    
    
    # Random Forest
    
    rf_model = train_random_forest(X_train, y_train)
    models['Random Forest'] = rf_model
    rf_result = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    results.append(rf_result)
    
    
    # XGBoost
    
    xgb_model = train_xgboost(X_train, y_train)
    models['XGBoost'] = xgb_model
    xgb_result = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    results.append(xgb_result)
    
    
    # LightGBM
    
    lgb_model = train_lightgbm(X_train, y_train)
    models['LightGBM'] = lgb_model
    lgb_result = (lgb_model, X_test, y_test, "LightGBM")
    results.append(lgb_result)
    
    
    # CatBoost
    
    cat_model = train_catboost(X_train, y_train)
    models['CatBoost'] = cat_model
    cat_result = evaluate_model(cat_model, X_test, y_test, "CatBoost")
    results.append(cat_result)
    
    
    # Step 3
    print("\n[Step 3] Comparing models...")

    results_df = compare_models(
        results
    )
    
    
    # Step 4
    print("\n[Step 4] Saving best model...")

    save_best_model(
        models,
        results_df
    )


    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETED")
    print("=" * 60)


    return results_df

        

if __name__ == "__main__":

    results_df = train_pipeline()

    print("\nFinal Ranking")
    print(results_df)