# backend/app.py

from pathlib import Path
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


from src.predict import predict_fraud



# Import prediction pipeline

from src.predict import predict_fraud



# FASTAPI APP

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Predicts whether a transaction is fraudulent using a LightGBM model.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://credit-card-fraud-detection-murex.vercel.app"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# INPUT SCHEMA

class Transaction(BaseModel):

    TransactionAmt: float

    transaction_date: str

    transaction_time: str

    ProductCD: str

    card4: str

    card6: str



# ROOT ENDPOINT

@app.get("/")
def home():

    return {
        "message":
        "Credit Card Fraud Detection API is running"
    }



# PREDICTION ENDPOINT

@app.post("/predict")
def predict(transaction: Transaction):

    result = predict_fraud(
        transaction.dict()
    )


    return {

    "fraud_probability":
        result["fraud_probability"],

    "threshold":
        result["threshold"],

    "prediction":
        result["prediction"],

    "status":
        (
            "Fraud"
            if result["prediction"] == 1
            else "Legitimate"
        ),

    "confidence":
        result["confidence"]

}