# app/app.py
import streamlit as st
from predict import predict_species

st.set_page_config(page_title="Iris Prediction", layout="centered")
st.title("CHANGED TITLE")
st.caption("Enter measurements and click Predict.")

col1, col2 = st.columns(2)
with col1:
    sepal_length = st.number_input("Sepal Length (cm)", 0.0, 10.0, 5.1, step=0.1)
    petal_length = st.number_input("Petal Length (cm)", 0.0, 10.0, 1.4, step=0.1)
with col2:
    sepal_width  = st.number_input("Sepal Width (cm)",  0.0, 10.0, 3.5, step=0.1)
    petal_width  = st.number_input("Petal Width (cm)",  0.0, 10.0, 0.2, step=0.1)

if st.button("Predict"):
    species = predict_species([sepal_length, sepal_width, petal_length, petal_width])
    st.success(f"Predicted species: **{species.capitalize()}**")
