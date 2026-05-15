from fastapi import FastAPI
from pydantic import BaseModel

from src.predict import predict

app = FastAPI()

class InputData(BaseModel):
    feature1: float
    feature2: float

@app.get("/")
def home():
    return {"message": "API funcionando"}

@app.post("/predict")
def predict_endpoint(data: InputData):

    prediction = predict(
        data.feature1,
        data.feature2
    )

    return {
        "prediction": prediction
    }