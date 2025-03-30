from typing import List

from fastapi import FastAPI

import joblib
import pandas as pd
from pydantic import BaseModel


# FastAPI app instance
app = FastAPI()

# Load the trained model and scaler from disk
model = joblib.load("artifacts/model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")

# Define input data structure
class InputData(BaseModel):
    data: List[float]


@app.post("/predict")
def predict(input_data: InputData):
    # Convert input data to dataframe
    df = pd.DataFrame([input_data.data])

    features_scaled = scaler.transform(df)
    df_scaled = pd.DataFrame(features_scaled, columns=df.columns)

    # Make prediction
    prediction = model.predict(df_scaled)

    # Return the prediction as a response
    return {"prediction": prediction.tolist()}
