from fastapi.testclient import TestClient

from src.inference_api import app

client = TestClient(app)

def test_home():

    response = client.get("/")

    assert response.status_code == 200

def test_predict():

    response = client.post(
        "/predict",
        json={
            "feature1": 0.5,
            "feature2": -1.2
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data