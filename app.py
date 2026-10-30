import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Student Placement Predictor", page_icon="🎓", layout="centered")

@st.cache_resource
def load_bundle():
    return joblib.load("placement_model.pkl")

st.title("🎓 Student Placement Prediction")
st.write("Enter the student's academic and skill details to view the model's prediction.")

try:
    bundle = load_bundle()
except FileNotFoundError:
    st.error("placement_model.pkl was not found. Add it to the same GitHub repository as app.py.")
    st.stop()

model = bundle["model"]

st.subheader("Student details")
col1, col2 = st.columns(2)
with col1:
    cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
    aptitude = st.slider("Aptitude Score", 0, 100, 65)
    technical = st.slider("Technical Score", 0, 100, 65)
with col2:
    projects = st.number_input("Number of Projects", min_value=0, max_value=20, value=2, step=1)
    internship_choice = st.selectbox("Internship completed?", ["No", "Yes"])
    internship = 1 if internship_choice == "Yes" else 0

if st.button("Predict Placement", type="primary", use_container_width=True):
    input_df = pd.DataFrame([{
        "CGPA": cgpa,
        "AptitudeScore": aptitude,
        "TechnicalScore": technical,
        "Projects": projects,
        "Internship": internship
    }])
    prediction = int(model.predict(input_df)[0])
    labels = bundle.get("class_labels", {0: "Not Placed", 1: "Placed"})
    result = labels[prediction]

    st.subheader("Prediction result")
    if prediction == 1:
        st.success(f"Model prediction: {result}")
    else:
        st.warning(f"Model prediction: {result}")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        classes = list(model.named_steps["classifier"].classes_)
        prob_map = {int(cls): float(prob) for cls, prob in zip(classes, probabilities)}
        st.write("Model probability estimates (not guarantees):")
        st.metric("Placed", f"{prob_map.get(1, 0) * 100:.1f}%")
        st.metric("Not Placed", f"{prob_map.get(0, 0) * 100:.1f}%")

st.divider()
st.caption("Educational demonstration only. Predictions are estimates from a small dataset and should not be used for actual admission, hiring, or placement decisions.")
