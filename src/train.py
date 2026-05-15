import pandas as pd
import joblib
import wandb

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# =========================
# W&B
# =========================

wandb.init(
    project="swat-mlops",
    config={
        "model": "RandomForest",
        "test_size": 0.2,
        "random_state": 42,
        "n_estimators": 100
    }
)

config = wandb.config

# =========================
# LOAD DATA
# =========================

df = pd.read_csv("data/merged.csv")

# limpiar nombres columnas
df.columns = df.columns.str.strip()

# target
df["target"] = df["Normal/Attack"].apply(
    lambda x: 0 if x == "Normal" else 1
)

# eliminar columnas no útiles
X = df.drop(columns=["Timestamp", "Normal/Attack", "target"])

y = df["target"]

# =========================
# SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=config.test_size,
    random_state=config.random_state,
    stratify=y
)

# =========================
# MODEL
# =========================

model = RandomForestClassifier(
    n_estimators=config.n_estimators,
    random_state=config.random_state,
    n_jobs=-1
)

model.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================

preds = model.predict(X_test)

accuracy = accuracy_score(y_test, preds)

print(f"Accuracy: {accuracy}")

print(classification_report(y_test, preds))

# =========================
# LOGGING
# =========================

wandb.log({
    "accuracy": accuracy
})

# =========================
# SAVE MODEL
# =========================

joblib.dump(model, "models/model.pkl")

artifact = wandb.Artifact(
    name="swat-randomforest",
    type="model"
)

artifact.add_file("models/model.pkl")

wandb.log_artifact(artifact)

print("Modelo guardado.")
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