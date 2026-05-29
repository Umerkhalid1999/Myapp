import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# --- Page Configuration ---
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="centered"
)

# --- Load Model ---
@st.cache_resource
def load_model():
    with open("diabetes_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

# --- Create Scaler Dynamically From Dataset ---
@st.cache_resource
def get_fitted_scaler():
    try:
        # Load the original dataset to match the training distribution perfectly
        df = pd.read_csv("diabetes.csv")
        X = df.drop(columns="Outcome", axis=1)
        
        # Fit a fresh scaler matching your current scikit-learn version
        scaler = StandardScaler()
        scaler.fit(X) 
        return scaler
    except FileNotFoundError:
        st.error("Error: 'diabetes.csv' not found. Please place it in the same directory.")
        return None

# Load assets
model = load_model()
scaler = get_fitted_scaler()

# --- App UI ---
st.title("🩺 Diabetes Predictive System")
st.write("This application utilizes a Machine Learning model to predict whether a patient is likely to be diabetic based on diagnostic measurements.")
st.markdown("---")

st.subheader("Enter Patient Details")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
    glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=100)
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
    
with col2:
    insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI (Body Mass Index)", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.3f")
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=30, step=1)

st.markdown("---")

# --- Prediction Logic ---
if st.button("Predict Health Status", type="primary"):
    if scaler is not None and model is not None:
        # 1. Collect inputs into a raw NumPy array
        raw_features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
        
        # 2. Scale features using our dynamically fitted scaler
        processed_features = scaler.transform(raw_features)
        
        # 3. Predict 
        prediction = model.predict(processed_features)
        
        # 4. Display result
        st.subheader("Result:")
        if prediction[0] == 1:
            st.error("⚠️ **The model predicts the person is likely Diabetic.**")
        else:
            st.success("✅ **The model predicts the person is Not Diabetic.**")