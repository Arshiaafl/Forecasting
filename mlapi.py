from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import pickle
import numpy as np

app = FastAPI()

class Weather_data(BaseModel):
    weather_values: List[float]


with open('your_model.pkl', 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        model = u.load()
        print(model)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)



@app.post("/")
async def process_weather_data(weather_input: Weather_data):
    # Access the weather values using weather_input.weather_values
    last_7_days_weather = weather_input.weather_values
    input_data = [weather_input.weather_values]
    input_data = np.array(input_data)
    input_data = input_data.reshape(-1, 1)
    input_data = scaler.transform(input_data)
    input_data = input_data.reshape(1, -1)
    predictions = model.predict(input_data)
    y_pred_reshaped = predictions.reshape(-1, 1)
    y_pred_original = scaler.inverse_transform(y_pred_reshaped)

    return {"predictions": y_pred_original.tolist()}
