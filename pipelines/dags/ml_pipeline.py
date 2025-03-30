import shutil

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils import timezone

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def _ingest():
    df = pd.read_csv("data/student-enrollments.csv", sep=";")
    df.to_csv("data/raw_data.csv", index=False)


def _preprocess_data():
    df = pd.read_csv("data/raw_data.csv")
    df.columns = df.columns.str.lower()
    df = df.dropna()

    features = df.drop("target", axis=1)
    target = df["target"]

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    joblib.dump(scaler, "models/scaler.pkl")

    df_scaled = pd.DataFrame(features_scaled, columns=features.columns)
    df_scaled["target"] = target
    df_scaled.to_csv("data/features.csv", index=False)


def _train_model():
    df = pd.read_csv("data/features.csv")
    X = df.drop("target", axis=1)
    y = df["target"]
    model = RandomForestClassifier()
    model.fit(X, y)
    joblib.dump(model, "models/model.pkl")


def _evaluate_model():
    df = pd.read_csv("data/features.csv")
    X = df.drop("target", axis=1)
    y = df["target"]
    model = joblib.load("models/model.pkl")
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    with open("models/metrics.txt", "w") as f:
        f.write(f"accuracy: {accuracy}")


def _save_artifacts():
    shutil.copy("models/model.pkl", "artifacts/model.pkl")
    shutil.copy("models/scaler.pkl", "artifacts/scaler.pkl")
    shutil.copy("models/metrics.txt", "artifacts/metrics.txt")


with DAG(
    "ml_pipeline",
    schedule=None,
    start_date=timezone.datetime(2025, 3, 30),
    catchup=False,
):

    ingest = PythonOperator(
        task_id="ingest",
        python_callable=_ingest,
    )

    preprocess_data = PythonOperator(
        task_id="preprocess_data",
        python_callable=_preprocess_data,
    )

    train_model = PythonOperator(
        task_id="train_model",
        python_callable=_train_model,
    )

    evaluate_model = PythonOperator(
        task_id="evaluate_model",
        python_callable=_evaluate_model,
    )

    save_artifacts = PythonOperator(
        task_id="save_artifacts",
        python_callable=_save_artifacts,
    )

    ingest >> preprocess_data >> train_model >> evaluate_model >> save_artifacts
