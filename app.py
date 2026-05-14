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
- **Chest Pain Type** → Type of chest pain  
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

# Age
Age = st.number_input(
    "Age (Years)",
    min_value=1,
    max_value=120,
    value=25
)

# Gender
Sex = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

Sex = 0 if Sex == "Female" else 1

# Chest Pain Type
ChestPain_option = st.selectbox(
    "Chest Pain Type",
    [
        "Typical Angina",
        "Atypical Angina",
        "Non-Anginal Pain",
        "Asymptomatic"
    ]
)

ChestPain_mapping = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-Anginal Pain": 2,
    "Asymptomatic": 3
}

ChestPain = ChestPain_mapping[ChestPain_option]

# Blood Pressure
BP = st.number_input(
    "Blood Pressure (BP)",
    min_value=0.0,
    value=120.0
)

# Cholesterol
Cholesterol = st.number_input(
    "Cholesterol Level",
    min_value=0.0,
    value=200.0
)

# Fasting Blood Sugar
FBS = st.selectbox(
    "Fasting Blood Sugar Above 120?",
    ["No", "Yes"]
)

FBS = 0 if FBS == "No" else 1

# EKG Results
EKG_option = st.selectbox(
    "EKG Results",
    [
        "Normal",
        "Minor Abnormality",
        "Major Abnormality"
    ]
)

EKG_mapping = {
    "Normal": 0,
    "Minor Abnormality": 1,
    "Major Abnormality": 2
}

EKG = EKG_mapping[EKG_option]

# Maximum Heart Rate
MaxHR = st.number_input(
    "Maximum Heart Rate",
    min_value=0.0,
    value=150.0
)

# Exercise Angina
ExerciseAngina = st.selectbox(
    "Chest Pain During Exercise?",
    ["No", "Yes"]
)

ExerciseAngina = 0 if ExerciseAngina == "No" else 1

# ST Depression
STDepression = st.number_input(
    "ST Depression",
    min_value=0.0,
    value=1.0
)

# Slope
Slope_option = st.selectbox(
    "Slope of ST Segment",
    [
        "Upsloping",
        "Flat",
        "Downsloping"
    ]
)

Slope_mapping = {
    "Upsloping": 0,
    "Flat": 1,
    "Downsloping": 2
}

Slope = Slope_mapping[Slope_option]

# Vessels
Vessels_option = st.selectbox(
    "Number of Major Vessels",
    [
        "0 Vessels",
        "1 Vessel",
        "2 Vessels",
        "3 Vessels"
    ]
)

Vessels_mapping = {
    "0 Vessels": 0,
    "1 Vessel": 1,
    "2 Vessels": 2,
    "3 Vessels": 3
}

Vessels = Vessels_mapping[Vessels_option]

# Thallium
Thallium_option = st.selectbox(
    "Thallium Test Result",
    [
        "Normal",
        "Fixed Defect",
        "Reversible Defect"
    ]
)

Thallium_mapping = {
    "Normal": 3,
    "Fixed Defect": 6,
    "Reversible Defect": 7
}

Thallium = Thallium_mapping[Thallium_option]

# =========================
# Prediction
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

    # Probability
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
