import joblib
import pandas as pd

model = joblib.load("models/model.pkl")

def predict(data_dict):

    df = pd.DataFrame([data_dict])

    # añadir columnas faltantes automáticamente
    df = df.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    prediction = model.predict(df)

    return int(prediction[0])