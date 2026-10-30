
import streamlit as st
import pandas as pd
import joblib

# 1. Page configuration
st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="🎓",
    layout="wide"
)

# 2. Load trained model
@st.cache_resource
def load_bundle():
    return joblib.load("placement_model.pkl")

bundle = load_bundle()

model = bundle["model"]

feature_columns = bundle["feature_columns"]
numeric_features = bundle["numeric_features"]
categorical_features = bundle["categorical_features"]

# 3. Application title
st.title("🎓 Student Placement Prediction")

st.write(
    "Enter student details to predict the placement status "
    "using Machine Learning."
)

st.info(
    "Educational demonstration only. Predictions are not "
    "guarantees of placement or suitable for employment decisions."
)

# 4. Create student input form
values = {}

with st.form("student_profile"):

    st.subheader("Student Information")

    cols = st.columns(2)

    for i, feature in enumerate(feature_columns):

        with cols[i % 2]:

            label = feature.replace("_", " ").title()

            if feature in categorical_features:

                options = bundle["categories"].get(
                    feature, []
                )

                default = bundle["defaults"][feature]

                index = (
                    options.index(default)
                    if default in options else 0
                )

                values[feature] = st.selectbox(
                    label,
                    options,
                    index=index,
                    key=feature
                )

            else:

                low, high = bundle["numeric_bounds"][feature]

                default = bundle["defaults"][feature]

                values[feature] = st.number_input(
                    label,
                    min_value=float(low),
                    max_value=float(high),
                    value=float(default),
                    key=feature
                )

    submitted = st.form_submit_button(
        "Predict Placement Status"
    )

# 5. Prediction
if submitted:

    input_df = pd.DataFrame(
        [values],
        columns=feature_columns
    )

    prediction = model.predict(input_df)[0]

    probabilities = model.predict_proba(input_df)[0]

    class_index = list(model.classes_).index(prediction)

    confidence = float(probabilities[class_index])

    st.subheader("Prediction Result")

    if str(prediction).lower() == "placed":

        st.success(
            f"Predicted Placement Status: {prediction}"
        )

    else:

        st.warning(
            f"Predicted Placement Status: {prediction}"
        )

    st.metric(
        "Model probability for predicted class",
        f"{confidence:.2%}"
    )

    st.caption(
        "This probability is a model estimate, not a guarantee."
    )
