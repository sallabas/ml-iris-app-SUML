import os
import json
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# ======================
# CONFIG
# ======================
VERSION = "v1.0.0"
EXPERIMENT_NAME = "iris-model-zoo"
REGISTERED_MODEL_NAME = "IrisModel"


def train_model():
    # MLflow experiment
    mlflow.set_experiment(EXPERIMENT_NAME)

    # ======================
    # Load data
    # ======================
    iris = load_iris(as_frame=True)
    df = iris.frame

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ======================
    # Model Zoo
    # ======================
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),
        "LogisticRegression": LogisticRegression(max_iter=200),
        "SVM": SVC(probability=True),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }

    best_f1 = 0.0
    best_model = None
    best_model_name = None
    best_metrics = {}
    best_run_id = None

    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("app", exist_ok=True)

    # ======================
    # Training loop
    # ======================
    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name) as run:

            # Params & tags
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("random_state", 42)
            mlflow.set_tag("version", VERSION)

            # Train
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Metrics
            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average="macro")
            recall = recall_score(y_test, y_pred, average="macro")
            f1 = f1_score(y_test, y_pred, average="macro")

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision_macro", precision)
            mlflow.log_metric("recall_macro", recall)
            mlflow.log_metric("f1_macro", f1)

            if hasattr(model, "predict_proba"):
                roc_auc = roc_auc_score(
                    y_test,
                    model.predict_proba(X_test),
                    multi_class="ovr"
                )
                mlflow.log_metric("roc_auc", roc_auc)

            # ======================
            # Artifacts
            # ======================
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title(f"Confusion Matrix - {model_name}")
            cm_path = f"artifacts/cm_{model_name}.png"
            plt.savefig(cm_path)
            plt.close()
            mlflow.log_artifact(cm_path)

            # Classification Report
            report = classification_report(y_test, y_pred)
            report_path = f"artifacts/report_{model_name}.txt"
            with open(report_path, "w") as f:
                f.write(report)
            mlflow.log_artifact(report_path)

            # ======================
            # Best model tracking
            # ======================
            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                best_model_name = model_name
                best_metrics = {
                    "accuracy": round(acc, 3),
                    "f1_macro": round(f1, 3)
                }
                best_run_id = run.info.run_id

                # Register best model inside its run
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    registered_model_name=REGISTERED_MODEL_NAME
                )

    # ======================
    # Save best model locally
    # ======================
    joblib.dump(best_model, "app/model.joblib")

    model_meta = {
        "best_model": best_model_name,
        "metrics": best_metrics,
        "mlflow_run_id": best_run_id,
        "version": VERSION
    }

    with open("app/model_meta.json", "w") as f:
        json.dump(model_meta, f, indent=4)

    print("===================================")
    print("Training completed successfully")
    print("Best model:", best_model_name)
    print("Best F1-macro:", best_f1)
    print("Saved files:")
    print("- app/model.joblib")
    print("- app/model_meta.json")
    print("Registered model:", REGISTERED_MODEL_NAME)
    print("===================================")


if __name__ == "__main__":
    train_model()
