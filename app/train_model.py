"""
Entrena y serializa el modelo de clasificación que servirá la API.

Dataset: Pima Indians Diabetes (768 pacientes, 8 variables clínicas).
Objetivo: predecir si la paciente tiene diabetes (Outcome = 1) o no (0).

El preprocesamiento (imputación de ceros fisiológicamente imposibles)
va DENTRO del Pipeline, de modo que la API recibe los valores crudos
y aplica exactamente el mismo tratamiento que en el entrenamiento.

Genera: app/model/model.joblib y app/model/metadata.json
"""
import json
import os
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(os.path.dirname(BASE_DIR), "data", "diabetes.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
TARGET = "Outcome"
# En estas columnas un 0 no es un valor real (glucosa 0, BMI 0...): se trata como dato faltante.
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
TARGET_NAMES = ["Sin diabetes", "Diabetes"]


def build_pipeline() -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            (
                "imputar_ceros",
                SimpleImputer(missing_values=0, strategy="median"),
                ZERO_AS_MISSING,
            )
        ],
        remainder="passthrough",
    )
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
    )
    return Pipeline([("preprocesamiento", preprocess), ("modelo", model)])


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    X, y = df[FEATURES], df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    print(f"accuracy={acc:.4f}  f1={f1:.4f}  roc_auc={auc:.4f}")

    joblib.dump(pipe, os.path.join(MODEL_DIR, "model.joblib"))

    metadata = {
        "model_type": "RandomForestClassifier (Pipeline con imputación de ceros)",
        "dataset": "Pima Indians Diabetes (768 registros)",
        "feature_names": FEATURES,
        "target_names": TARGET_NAMES,
        "accuracy": round(acc, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Modelo guardado en {MODEL_DIR}/model.joblib")


if __name__ == "__main__":
    main()
