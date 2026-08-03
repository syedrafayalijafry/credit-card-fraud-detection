import pandas as pd
from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Raw data directory
RAW_DATA_DIR = BASE_DIR / "data" / "raw"

def load_raw_data():
    """
    Load and merge the IEEE-CIS training datasets.

    Returns
    -------
    pandas.DataFrame
        Merged training dataframe.
    """
    train_transaction = pd.read_csv(RAW_DATA_DIR / "train_transaction.csv")
    
    train_identity = pd.read_csv(RAW_DATA_DIR / "train_identity.csv")
    
    raw_df = train_transaction.merge(
        train_identity,
        how="left",
        on="TransactionID"
    )
    
    return raw_df

if __name__ == "__main__":
    
    print("Loading and Merging Datsets...\n")
   
    df = load_raw_data()
    
    print("Dataset Loaded Successfully\n")
    
    print(f"Dataset Shape: {df.shape}\n")
    
    print("Dataset Preview:\n")
    
    print(df.head())