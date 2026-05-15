from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

from src.predict import predict

app = FastAPI()


class SWaTInput(BaseModel):

    FIT101: float
    LIT101: float
    MV101: int
    P101: int
    P102: int
    AIT201: float
    AIT202: float
    AIT203: float
    FIT201: float
    MV201: int
    P201: int
    P202: int
    P203: int
    P204: int
    P205: int
    P206: int
    DPIT301: float
    FIT301: float
    LIT301: float
    MV301: int
    MV302: int
    MV303: int
    MV304: int
    P301: int
    P302: int
    AIT401: float
    AIT402: float
    FIT401: float
    LIT401: float
    P401: int
    P402: int
    P403: int
    P404: int
    UV401: int
    AIT501: float
    AIT502: float
    AIT503: float
    AIT504: float
    FIT501: float
    FIT502: float
    FIT503: float
    FIT504: float
    P501: int
    P502: int
    PIT501: float
    PIT502: float
    PIT503: float
    FIT601: float

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "FIT101": 2.5,
                "LIT101": 500,
                "MV101": 1,
                "P101": 1,
                "P102": 1,
                "AIT201": 250,
                "AIT202": 8,
                "AIT203": 220,
                "FIT201": 2.1,
                "MV201": 1,
                "P201": 1,
                "P202": 1,
                "P203": 1,
                "P204": 1,
                "P205": 1,
                "P206": 1,
                "DPIT301": 2,
                "FIT301": 2,
                "LIT301": 700,
                "MV301": 1,
                "MV302": 0,
                "MV303": 1,
                "MV304": 0,
                "P301": 1,
                "P302": 1,
                "AIT401": 100,
                "AIT402": 100,
                "FIT401": 1.5,
                "LIT401": 800,
                "P401": 1,
                "P402": 1,
                "P403": 1,
                "P404": 1,
                "UV401": 1,
                "AIT501": 5,
                "AIT502": 5,
                "AIT503": 5,
                "AIT504": 5,
                "FIT501": 1,
                "FIT502": 1,
                "FIT503": 1,
                "FIT504": 1,
                "P501": 1,
                "P502": 1,
                "PIT501": 10,
                "PIT502": 10,
                "PIT503": 10,
                "FIT601": 1
            }
        }
    )


@app.get("/")
def home():
    return {
        "message": "SWaT Attack Detection API funcionando"
    }


@app.post("/predict")
def predict_endpoint(data: SWaTInput):

    prediction = predict(data.model_dump())

    return {
        "prediction": prediction
    }