from fastapi import FastAPI
from src.predict import predict

app = FastAPI()

@app.get("/")
def home():
    return {"message": "SWaT Attack Detection API funcionando"}

@app.post("/predict")
def predict_endpoint(data: dict):

    prediction = predict(data)

    return {
        "prediction": prediction
    }