# app.py

import streamlit as st
import pandas as pd
import joblib


# --- Load the Trained Model and Artifacts ---
@st.cache_data
def load_artifacts():
    """
    Loads the saved model and preprocessing artifacts.
    Using st.cache_data to load only once.
    """
    artifacts = joblib.load("model_artifacts.joblib")
    model = artifacts["model"]
    numeric_imputer = artifacts["numeric_imputer"]
    categorical_imputer = artifacts["categorical_imputer"]
    encoded_columns = artifacts["encoded_columns"]
    return model, numeric_imputer, categorical_imputer, encoded_columns


model, numeric_imputer, categorical_imputer, encoded_columns = load_artifacts()

# --- Page Configuration ---
st.set_page_config(
    page_title="Heart Disease Prediction", page_icon="❤️", layout="centered"
)
st.title("❤️ Heart Disease Prediction")
st.write(
    "Enter the patient's details below to predict the likelihood of heart disease."
)

# --- User Input Form ---
with st.form("prediction_form"):
    st.header("Patient Details")

    # Split columns for a better layout
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Sex", options=["Male", "Female"])
        cp = st.selectbox(
            "Chest Pain Type (cp)",
            options=[
                "typical angina",
                "atypical angina",
                "non-anginal pain",
                "asymptomatic",
            ],
        )
        trestbps = st.number_input(
            "Resting Blood Pressure (trestbps)", min_value=80, max_value=200, value=120
        )
        chol = st.number_input(
            "Serum Cholestoral in mg/dl (chol)", min_value=100, max_value=600, value=200
        )
        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl (fbs)", options=["True", "False"]
        )

    with col2:
        restecg = st.selectbox(
            "Resting Electrocardiographic Results (restecg)",
            options=["normal", "st-t wave abnormality", "left ventricular hypertrophy"],
        )
        thalach = st.number_input(
            "Maximum Heart Rate Achieved (thalch)",
            min_value=60,
            max_value=220,
            value=150,
        )
        exang = st.selectbox("Exercise Induced Angina (exang)", options=["Yes", "No"])
        oldpeak = st.number_input(
            "ST depression induced by exercise relative to rest (oldpeak)",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1,
        )
        slope = st.selectbox(
            "Slope of the peak exercise ST segment (slope)",
            options=["upsloping", "flat", "downsloping"],
        )
        ca = st.selectbox(
            "Number of major vessels colored by flourosopy (ca)",
            options=["0", "1", "2", "3", "4"],
        )
        thal = st.selectbox(
            "Thalassemia (thal)",
            options=["normal", "fixed defect", "reversable defect"],
        )

    # Submit button
    submit_button = st.form_submit_button(label="Predict")

# --- Prediction Logic ---
if submit_button:
    # 1. Create a DataFrame from user inputs
    input_data = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalch": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
    }
    input_df = pd.DataFrame([input_data])

    # 2. Preprocess the input data (Impute and Encode)
    # Separate numerical and categorical columns based on the input_df
    numerical_cols = ["age", "trestbps", "chol", "thalch", "oldpeak"]
    categorical_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

    input_df[numerical_cols] = numeric_imputer.transform(input_df[numerical_cols])
    input_df[categorical_cols] = categorical_imputer.transform(
        input_df[categorical_cols]
    )

    # One-hot encode the categorical features
    input_df_encoded = pd.get_dummies(
        input_df, columns=categorical_cols, drop_first=True
    )

    # 3. Align columns with the training data
    # Ensure the input data has the same columns as the model was trained on
    input_df_aligned = input_df_encoded.reindex(columns=encoded_columns, fill_value=0)

    # 4. Make Prediction
    prediction = model.predict(input_df_aligned)
    prediction_proba = model.predict_proba(input_df_aligned)

    # 5. Display the result
    st.header("Prediction Result")
    if prediction[0] == 1:
        st.error(f"**High Risk of Heart Disease**")
        st.write(
            f"The model predicts a **{prediction_proba[0][1] * 100:.2f}%** probability of heart disease."
        )
    else:
        st.success(f"**Low Risk of Heart Disease**")
        st.write(
            f"The model predicts a **{prediction_proba[0][1] * 100:.2f}%** probability of heart disease."
        )

    st.write("---")
    st.write(
        "**Disclaimer:** This prediction is based on a machine learning model and should not be considered a substitute for professional medical advice."
    )
