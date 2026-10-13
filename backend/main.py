from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from pathlib import Path
from backend.database import create_tables, save_prediction, get_predictions
import joblib
import pandas as pd




app = FastAPI()
create_tables()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


BASE_DIR = Path(__file__).resolve().parent.parent

# Find our saved model
MODEL_PATH = BASE_DIR / "models" / "xgboost.pkl"

# Load the trained model
model = joblib.load(MODEL_PATH)

print("XGBoost model loaded successfully!")


@app.get("/")
def home():
    return {"message": "Welcome to FraudLens API"}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "XGBoost",
        "model_loaded": model is not None
    }

from pydantic import BaseModel, Field, ConfigDict

class Transaction(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    Time: float = Field(ge=0)
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(ge=0)

@app.post("/predict")
def predict(transaction: Transaction):
    try: 
         transaction_data = pd.DataFrame([transaction.model_dump()])

         fraud_probability = float(model.predict_proba(transaction_data)[0][1])

         threshold = 0.20

         is_fraud = fraud_probability >= threshold

         save_prediction(transaction.Amount, "Fraud" if is_fraud else "Legitimate", fraud_probability)

         return {
             "fraud_probability": float(fraud_probability),
             "is_fraud": bool(is_fraud),
             "prediction": "Fraud" if is_fraud else "Legitimate"

        
         }
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing the prediction."
        )
    
@app.get("/predictions")
def prediction_history():
    return get_predictions()
    
