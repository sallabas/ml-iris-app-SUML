import json
import joblib
import mlflow
import mlflow.sklearn

import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# DATA
def load_iris_data():
    iris = load_iris(as_frame=True)
    return iris.frame


def split_features_target(df: pd.DataFrame):
    X = df.drop(columns="target")
    y = df["target"]
    return X, y


def split_train_test(X, y, train_test_split_params: dict):
    return train_test_split(
        X,
        y,
        test_size=train_test_split_params["test_size"],
        random_state=train_test_split_params["random_state"],
    )


# TRAIN MODELS
def train_knn(X_train, X_test, y_train, y_test, params):
    model = KNeighborsClassifier(n_neighbors=params["n_neighbors"])
    model.fit(X_train, y_train)
    f1 = f1_score(y_test, model.predict(X_test), average="macro")
    return {"model": model, "f1": f1}


def train_logreg(X_train, X_test, y_train, y_test, params):
    model = LogisticRegression(max_iter=params["max_iter"])
    model.fit(X_train, y_train)
    f1 = f1_score(y_test, model.predict(X_test), average="macro")
    return {"model": model, "f1": f1}


def train_rf(X_train, X_test, y_train, y_test, params):
    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        random_state=42,
    )
    model.fit(X_train, y_train)
    f1 = f1_score(y_test, model.predict(X_test), average="macro")
    return {"model": model, "f1": f1}


def train_svm(X_train, X_test, y_train, y_test, params):
    model = SVC(C=params["C"], kernel=params["kernel"], probability=True)
    model.fit(X_train, y_train)
    f1 = f1_score(y_test, model.predict(X_test), average="macro")
    return {"model": model, "f1": f1}


# SELECT + SAVE
def select_best_model(knn, logreg, rf, svm):
    models = {
        "KNN": knn,
        "LogReg": logreg,
        "RandomForest": rf,
        "SVM": svm,
    }
    best_name = max(models, key=lambda k: models[k]["f1"])
    return best_name, models[best_name]


def save_model(best_model):
    joblib.dump(best_model["model"], "app/model.joblib")
    mlflow.sklearn.log_model(
        best_model["model"],
        name="IrisModel",
        registered_model_name="IrisModel",
    )
    return "app/model.joblib"


def save_metadata(best_model_name, best_model):
    meta = {
        "best_model": best_model_name,
        "f1_macro": round(best_model["f1"], 3),
    }
    with open("app/model_meta.json", "w") as f:
        json.dump(meta, f, indent=4)
    return "app/model_meta.json"
