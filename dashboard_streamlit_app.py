
import streamlit as st
import pandas as pd
from pathlib import Path
import joblib, json

st.title("🌾 Crop Production Predictor")
MODEL_PATH = Path("models/best_model.joblib")
META_PATH = Path("models/metadata.json")

if not MODEL_PATH.exists() or not META_PATH.exists():
    st.error("Model not found. Run training first.")
    st.stop()

model = joblib.load(MODEL_PATH)
meta = json.loads(Path(META_PATH).read_text())

cat_cols = meta.get("cat_cols", [])
num_cols = meta.get("num_cols", [])

st.subheader("Enter Features")
inputs = {}
for c in cat_cols:
    inputs[c] = st.text_input(c, "Unknown")
for c in num_cols:
    inputs[c] = st.number_input(c, value=0.0)

if st.button("Predict"):
    X = pd.DataFrame([inputs])
    pred = model.predict(X)[0]
    st.success(f"Estimated Production: {pred:,.2f}")
