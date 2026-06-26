# Import the Libraries
import pickle
import streamlit as st
import pandas as pd
import tensorflow as tf

# Cached loaders
@st.cache_resource
def load_ann_model():
    return tf.keras.models.load_model('models/ann_model.keras')

# Load Cached Encoders and Scalers
@st.cache_resource
def load_artifacts():
    with open("artifacts/label_encoder.pkl", "rb") as file:
        label_encoder = pickle.load(file)
    with open("artifacts/onehot_encoder.pkl", "rb") as file:
        onehot_encoder = pickle.load(file)
    with open("artifacts/standard_scaler.pkl", "rb") as file:
        standard_scaler = pickle.load(file)
    return label_encoder, onehot_encoder, standard_scaler

# Load model + artifacts
model = load_ann_model()
label_encoder, onehot_encoder, standard_scaler = load_artifacts()

# Streamlit App
st.title("Customer Churn Prediction")

# User Inputs
geography = st.selectbox('Geography', onehot_encoder.named_transformers_['encode'].categories_[0])
gender = st.selectbox('Gender', label_encoder.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance', min_value=0.0)
credit_score = st.number_input('Credit Score', min_value=300, max_value=900)
estimated_salary = st.number_input('Estimated Salary', min_value=0.0)
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Predict churn
if st.button("Predict"):

    # Prepare the Input Data
    input_data = pd.DataFrame({
        'Geography': [geography],
        'CreditScore': [credit_score],
        'Gender': [label_encoder.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # OneHotEncoder Geography Column
    input_data = pd.DataFrame(onehot_encoder.transform(input_data), columns=onehot_encoder.get_feature_names_out())

    # Refactor Column Names
    input_data.columns = [col.split("__")[-1] for col in input_data.columns]

    # Scale the Input Data
    input_data.iloc[:, 3:] = standard_scaler.transform(input_data.iloc[:, 3:])

    prediction = model.predict(input_data)
    prediction_proba = prediction[0][0]

    st.metric("Churn Probability", f"{prediction_proba:.2%}")

    if prediction_proba > 0.4:
        st.error('The customer is likely to churn.')
    else:
        st.success('The customer is not likely to churn.')