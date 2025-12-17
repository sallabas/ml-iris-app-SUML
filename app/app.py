# app/app.py
import json
import streamlit as st
from predict import predict_species
from pathlib import Path

# ======================
# Page config
# ======================
st.set_page_config(page_title="Iris Prediction", layout="centered")
st.title("Iris Flower Prediction")
st.caption("Enter measurements and click Predict.")

# ======================
# Input form
# ======================
col1, col2 = st.columns(2)
with col1:
    sepal_length = st.number_input("Sepal Length (cm)", 0.0, 10.0, 5.1, step=0.1)
    petal_length = st.number_input("Petal Length (cm)", 0.0, 10.0, 1.4, step=0.1)
with col2:
    sepal_width = st.number_input("Sepal Width (cm)", 0.0, 10.0, 3.5, step=0.1)
    petal_width = st.number_input("Petal Width (cm)", 0.0, 10.0, 0.2, step=0.1)

if st.button("Predict"):
    species = predict_species(
        [sepal_length, sepal_width, petal_length, petal_width]
    )
    st.success(f"Predicted species: **{species.capitalize()}**")

# ======================
# Footer – MLflow metadata
# ======================
st.markdown("---")

meta_path = Path("app/model_meta.json")

if meta_path.exists():
    with open(meta_path, "r") as f:
        meta = json.load(f)

    version = meta.get("version", "N/A")
    best_model = meta.get("best_model", "N/A")
    run_id = meta.get("mlflow_run_id", "N/A")
    accuracy = meta.get("metrics", {}).get("accuracy", "N/A")

    mlflow_url = f"http://localhost:5000/#/experiments/0/runs/{run_id}"

    st.markdown(
        f"""
        <div style="text-align: center; font-size: 0.9em; color: gray;">
            <b>Version:</b> {version} &nbsp; • &nbsp;
            <b>Best model:</b> {best_model} &nbsp; • &nbsp;
            <b>MLflow run:</b>
            <a href="{mlflow_url}" target="_blank">{run_id[:8]}...</a>
            &nbsp; • &nbsp;
            <b>Accuracy:</b> {accuracy}
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.warning("Model metadata not found. Please run train_model.py first.")
