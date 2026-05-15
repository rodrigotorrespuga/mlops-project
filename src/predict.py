import joblib
import pandas as pd

model = joblib.load("models/model.pkl")

def predict(feature1, feature2):

    data = pd.DataFrame([{
        "feature1": feature1,
        "feature2": feature2
    }])

    pred = model.predict(data)

    return int(pred[0])