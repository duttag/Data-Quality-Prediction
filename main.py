from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# 1. Load the model and scaler when the app starts
app = FastAPI(title="Data Quality Prediction API")
model = joblib.load('logistic_model.joblib')
scaler = joblib.load('data_scaler.joblib')

# 2. Define the input data format (matching your columns)
class DataInput(BaseModel):
    Creator: int  # Use the encoded integer values
    Source: int
    Duration_sec: float
    Location: int
    Day_of_Week: int
    Hour: int

# 3. Create the prediction endpoint
@app.post("/predict")
def predict_quality(data: DataInput):
    # Convert input into the right format for the model
    features = np.array([[
        data.Creator, data.Source, data.Duration_sec, 
        data.Location, data.Day_of_Week, data.Hour
    ]])
    
    # Scale and predict
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)
    
    return {"prediction": "Good" if prediction[0] == 1 else "Bad"}