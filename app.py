import streamlit as st
import numpy as np
import joblib

# -------------------------
# Page Setup
# -------------------------
st.set_page_config(
    page_title="Heart Disease Predictor",
    layout="centered"
)

# -------------------------
# Load Model Safely
# -------------------------
try:
    model = joblib.load("heart_disease_model.pkl")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# -------------------------
# Main Title
# -------------------------
st.title("❤️ Heart Disease Prediction System")

st.write("""
This AI tool helps predict the **risk of heart disease**
using machine learning based on medical information.

👉 Fill in the patient details below  
👉 Click the Predict button  
👉 Get prediction result + probability score
""")

st.write("---")

# =========================
# Sidebar Information
# =========================
st.sidebar.title("📘 How to Use This App")

st.sidebar.markdown("""
### 🧠 About This Tool
This system predicts whether a person may have a **risk of heart disease**.

---

### 📊 Input Information

- **Age** → Age in years  
- **Gender** → Male or Female  
- **Chest Pain Type** → Type of chest pain (0–3)  
- **Blood Pressure** → BP level  
- **Cholesterol** → Cholesterol level in blood  
- **Fasting Blood Sugar** → Above 120 or not  
- **EKG Results** → ECG test result  
- **Maximum Heart Rate** → Highest heart rate achieved  
- **Exercise Angina** → Chest pain during exercise  
- **ST Depression** → ECG measurement  
- **Slope** → Slope of heart graph  
- **Blocked Vessels** → Number of blocked vessels  
- **Thallium Test** → Heart scan result  

---

### ⚠️ Important
- This is an AI prediction system  
- This is NOT a medical diagnosis  
- Please consult a doctor for professional advice
""")

# =========================
# User Inputs
# =========================
st.subheader("📝 Enter Patient Details")

Age = st.number_input(
    "Age (Years)",
    min_value=1,
    max_value=120,
    value=25
)

Sex = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

Sex = 0 if Sex == "Female" else 1

ChestPain = st.selectbox(
    "Chest Pain Type",
    [0, 1, 2, 3]
)

BP = st.number_input(
    "Blood Pressure (BP)",
    min_value=0.0,
    value=120.0
)

Cholesterol = st.number_input(
    "Cholesterol Level",
    min_value=0.0,
    value=200.0
)

FBS = st.selectbox(
    "Fasting Blood Sugar > 120",
    [0, 1]
)

EKG = st.selectbox(
    "EKG Results",
    [0, 1, 2]
)

MaxHR = st.number_input(
    "Maximum Heart Rate",
    min_value=0.0,
    value=150.0
)

ExerciseAngina = st.selectbox(
    "Exercise-Induced Angina",
    [0, 1]
)

STDepression = st.number_input(
    "ST Depression",
    min_value=0.0,
    value=1.0
)

Slope = st.selectbox(
    "Slope of ST Segment",
    [0, 1, 2]
)

Vessels = st.selectbox(
    "Number of Major Vessels",
    [0, 1, 2, 3]
)

Thallium = st.selectbox(
    "Thallium Test Result",
    [3, 6, 7]
)

# =========================
# Prediction Button
# =========================
if st.button("🔍 Predict Heart Disease"):

    input_data = np.array([[
        Age,
        Sex,
        ChestPain,
        BP,
        Cholesterol,
        FBS,
        EKG,
        MaxHR,
        ExerciseAngina,
        STDepression,
        Slope,
        Vessels,
        Thallium
    ]])

    # Prediction
    prediction = model.predict(input_data)

    st.write("---")
    st.subheader("📊 Prediction Result")

    if prediction[0] == 1:
        st.error("❤️ High Risk of Heart Disease")
    else:
        st.success("💚 Low Risk of Heart Disease")

    # Probability (if supported)
    if hasattr(model, "predict_proba"):

        probability = model.predict_proba(input_data)[0][1]

        st.write(
            f"### 🔢 Risk Probability: {probability * 100:.2f}%"
        )

# -------------------------
# Footer
# -------------------------
st.write("---")
st.write("Built with ❤️ using Machine Learning + Streamlit")
