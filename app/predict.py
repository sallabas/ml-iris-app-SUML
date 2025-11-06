# app/predict.py
import joblib
from sklearn.datasets import load_iris
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"

_model = joblib.load(MODEL_PATH)
_class_names = load_iris().target_names

def predict_species(features):
    pred = _model.predict([features])[0]
    return _class_names[pred]
