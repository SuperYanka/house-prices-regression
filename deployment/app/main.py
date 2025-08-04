from fastapi import FastAPI
import os
import sys
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(BASE_DIR, '..', '..', 'data', 'train.csv')
model_path = os.path.join(BASE_DIR, 'rf_small.pkl')
sys.path.append(os.path.join(BASE_DIR, '..', 'src'))

app = FastAPI()

print("Загружаем модель...")
model = joblib.load(model_path)
print("Модель загружена:", model)

class HouseData(BaseModel):
    OverallQual: int
    GrLivArea: float
    GarageCars: int
    TotalBsmtSF: float

@app.post("/predict")
def predict(data: HouseData):
    input_data = pd.DataFrame([{
    "OverallQual": data.OverallQual,
    "GrLivArea": data.GrLivArea,
    "GarageCars": data.GarageCars,
    "TotalBsmtSF": data.TotalBsmtSF
}])
    prediction = model.predict(input_data)
    sale_price = np.expm1(prediction[0])  # inv logarithm
    return {"Predicted SalePrice": round(sale_price, 2)}