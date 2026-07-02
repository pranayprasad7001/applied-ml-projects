import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

# --- Path Configuration ---
# Resolves the absolute path to the directory containing this script
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = BASE_DIR / "models"

# --- Page Configuration ---
st.set_page_config(
    page_title="Bank Customer Churn Predictor", page_icon="🏦", layout="centered"
)


# --- Load Model and Artifacts (Cached for Performance) ---
@st.cache_resource
def load_assets():
    """Loads the model and preprocessing artifacts once and caches them."""
    with open(ARTIFACTS_DIR / "label_encoder.pkl", "rb") as f:
        le_gender = pickle.load(f)
    with open(ARTIFACTS_DIR / "onehot_encoder.pkl", "rb") as f:
        ct_geo = pickle.load(f)
    with open(ARTIFACTS_DIR / "standard_scaler.pkl", "rb") as f:
        sc = pickle.load(f)

    # Some TensorFlow/Keras versions prefer string paths; wrapping in str() is bulletproof
    model = load_model(str(MODELS_DIR / "ann_model.keras"))
    return le_gender, ct_geo, sc, model


le_gender, ct_geo, sc, model = load_assets()


# --- Feature Engineering Function ---
def engineer_features(df):
    """Applies the custom business logic features used during training."""
    df_eng = df.copy()

    # Avoid division by zero in edge cases
    estimated_salary = df_eng["EstimatedSalary"].replace(0, 0.01)
    age = df_eng["Age"].replace(0, 1)

    df_eng["BalanceSalaryRatio"] = df_eng["Balance"] / estimated_salary
    df_eng["TenureByAge"] = df_eng["Tenure"] / age

    bins = [18, 30, 40, 50, 60, 100]
    labels = ["Young_Adult", "Adult", "Middle_Age", "Senior", "Elderly"]
    df_eng["AgeGroup"] = pd.cut(df_eng["Age"], bins=bins, labels=labels, right=False)

    df_eng["CreditScoreGivenAge"] = df_eng["CreditScore"] / age
    df_eng["Products_Active_Interaction"] = (
        df_eng["NumOfProducts"] * df_eng["IsActiveMember"]
    )

    return df_eng


# --- Prediction Pipeline ---
def predict_churn(input_data, threshold=0.45):
    """Processes input and returns a churn prediction."""
    df = pd.DataFrame([input_data])

    # Apply feature engineering
    df = engineer_features(df)

    # Encode Gender
    df["Gender"] = le_gender.transform(df["Gender"].str.capitalize())

    # Encode Geography and AgeGroup
    cols = ct_geo.get_feature_names_out()
    df_encoded = pd.DataFrame(
        ct_geo.transform(df), columns=[col.split("__")[-1] for col in cols]
    )

    # Scale Features
    df_encoded = df_encoded[sc.feature_names_in_]
    df_encoded = pd.DataFrame(sc.transform(df_encoded), columns=sc.feature_names_in_)

    # Get Probability
    probability = float(model.predict(df_encoded, verbose=0)[0][0])

    return probability, bool(probability > threshold)


# --- UI Application ---
st.title("🏦 Bank Customer Churn Predictor")
st.markdown(
    "Enter the customer's details below to determine their likelihood of leaving the bank."
)

# Create a clean layout using columns
col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input(
        "Credit Score", min_value=300, max_value=850, value=600
    )
    geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=18, max_value=100, value=40)
    tenure = st.number_input("Tenure (Years)", min_value=0, max_value=10, value=5)

with col2:
    balance = st.number_input("Account Balance ($)", min_value=0.0, value=50000.0)
    num_products = st.number_input(
        "Number of Products", min_value=1, max_value=4, value=2
    )
    has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
    is_active = st.selectbox("Is Active Member?", ["Yes", "No"])
    salary = st.number_input("Estimated Salary ($)", min_value=0.0, value=60000.0)

# Formatting inputs to match the model's expected format
input_dict = {
    "CreditScore": credit_score,
    "Geography": geography,
    "Gender": gender,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_products,
    "HasCrCard": 1 if has_cr_card == "Yes" else 0,
    "IsActiveMember": 1 if is_active == "Yes" else 0,
    "EstimatedSalary": salary,
}

st.markdown("---")

# --- Run Inference ---
if st.button("Predict Churn Risk", type="primary", use_container_width=True):
    with st.spinner("Analyzing customer profile..."):
        prob, will_churn = predict_churn(input_dict, threshold=0.45)

    st.markdown("### Results")

    if will_churn:
        st.error(
            f"⚠️ High Risk of Churn!\n\n"
            f"Churn Probability: {prob:.1%}\n"
            f"Stay Probability: {(1 - prob):.1%}"
        )
        st.warning(
            "Recommendation: Consider targeted retention strategies, such as fee waivers or promotional interest rates."
        )
    else:
        st.success(
            f"✅ Customer is likely to stay.\n\n"
            f"Stay Probability: {(1 - prob):.1%}\n"
            f"Churn Probability: {prob:.1%}"
        )
        st.info("Recommendation: Maintain standard engagement protocols.")
