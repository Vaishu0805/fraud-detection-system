from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import shap
import os

# Initialize FastAPI app
app = FastAPI()

# ✅ Fix path properly (VERY IMPORTANT)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models/fraud_model.pkl")

# Load model safely
model = joblib.load(MODEL_PATH)

# SHAP explainer
explainer = shap.TreeExplainer(model)

# Input schema
class Transaction(BaseModel):
    features: list


# Home endpoint
@app.get("/")
def home():
    return {"message": "Fraud Detection API"}


# 🔹 Predict endpoint
@app.post("/predict")
def predict_fraud(txn: Transaction):
    data = np.array(txn.features).reshape(1, -1)
    prediction = model.predict(data)

    return {
        "fraud": int(prediction[0])
    }


# 🔥 Explain endpoint
@app.post("/explain")
def explain_fraud(txn: Transaction):
    data = np.array(txn.features).reshape(1, -1)

    prediction = int(model.predict(data)[0])

    shap_values = explainer.shap_values(data)

    return {
        "prediction": prediction,
        "feature_contributions": shap_values[0].tolist()
    }