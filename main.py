from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import pickle
import os


# ---------------------------------------------------------
# 1. Create FastAPI application
# ---------------------------------------------------------

app = FastAPI(
    title="Loan Approval Prediction API",
    description="FastAPI backend for predicting loan approval using a trained ML model",
    version="1.0.0"
)


# ---------------------------------------------------------
# 2. Check model files
# ---------------------------------------------------------

if not os.path.exists("model.pkl"):
    raise FileNotFoundError(
        "model.pkl not found. Run train_model.py first."
    )

if not os.path.exists("scaler.pkl"):
    raise FileNotFoundError(
        "scaler.pkl not found. Run train_model.py first."
    )


# ---------------------------------------------------------
# 3. Load model and scaler
# ---------------------------------------------------------

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


# ---------------------------------------------------------
# 4. Request data model
# ---------------------------------------------------------

class LoanApplication(BaseModel):

    income: float = Field(
        ...,
        gt=0,
        description="Applicant annual income"
    )

    credit_score: float = Field(
        ...,
        ge=300,
        le=900,
        description="Credit score"
    )

    loan_amount: float = Field(
        ...,
        gt=0,
        description="Requested loan amount"
    )

    employment_years: float = Field(
        ...,
        ge=0,
        description="Years of employment"
    )


# ---------------------------------------------------------
# 5. Home endpoint
# ---------------------------------------------------------

@app.get("/")
def home():

    return {
        "status": "success",
        "message": "Loan Approval Prediction API is running"
    }


# ---------------------------------------------------------
# 6. Prediction endpoint
# ---------------------------------------------------------

@app.post("/predict")
def predict_loan(application: LoanApplication):

    # Create input DataFrame
    input_data = pd.DataFrame([{
        "income": application.income,
        "credit_score": application.credit_score,
        "loan_amount": application.loan_amount,
        "employment_years": application.employment_years
    }])


    # Apply the same scaler used during training
    input_scaled = scaler.transform(input_data)


    # Make prediction
    prediction = model.predict(input_scaled)[0]


    # Get probability of class 1 (Approved)
    probability = model.predict_proba(input_scaled)[0][1]


    # Convert prediction to readable status
    loan_status = (
        "Approved"
        if prediction == 1
        else "Rejected"
    )


    # Return result
    return {
        "prediction": int(prediction),
        "loan_status": loan_status,
        "approval_probability": float(probability)
    }