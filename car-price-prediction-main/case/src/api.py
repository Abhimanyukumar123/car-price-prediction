from fastapi import FastAPI
import pickle
import pandas as pd
from pathlib import Path

app = FastAPI(title="Used Car Price Prediction API")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "car_price_rf.pkl"

model = None

@app.on_event("startup")
def load_model():
    global model
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully")

@app.post("/predict")
def predict_price(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    return {"predicted_price": round(float(prediction), 2)}