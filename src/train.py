import pandas as pd
import numpy as np
import joblib
import wandb

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =========================
# W&B
# =========================
import wandb

wandb.login(key="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")
wandb.init(
    project="mlops-project",
    config={
        "model": "RandomForest",
        "test_size": 0.2,
        "random_state": 42
    }
)

config = wandb.config

# =========================
# Datos
# =========================

df = pd.DataFrame({
    'feature1': np.random.randn(200),
    'feature2': np.random.randn(200),
    'target': np.random.randint(0,2,200)
})

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=config.test_size,
    random_state=config.random_state
)

# =========================
# Modelo
# =========================

model = RandomForestClassifier()

model.fit(X_train, y_train)

preds = model.predict(X_test)

acc = accuracy_score(y_test, preds)

print("Accuracy:", acc)

# =========================
# Logging
# =========================

wandb.log({
    "accuracy": acc
})

# =========================
# Guardar modelo
# =========================

joblib.dump(model, "models/model.pkl")

artifact = wandb.Artifact(
    name="random-forest-model",
    type="model"
)

artifact.add_file("models/model.pkl")

wandb.log_artifact(artifact)

print("Modelo guardado.")