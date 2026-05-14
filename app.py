import streamlit as st
import numpy as np
import joblib

# -------------------------
# Load model
# -------------------------
model = joblib.load("heart_disease_model.pkl")

# -------------------------
# Page setup
# -------------------------
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

st.title("❤️ Heart Disease Prediction System")

st.write("""
This AI tool helps you check the **risk of heart disease** based on simple medical details.

👉 Fill in the form below and click **Predict**  
👉 You will get result + risk probability
""")

st.write("---")

# =========================
# 📘 Sidebar Help Section
# =========================
st.sidebar.title("📘 How to Use This App")

st.sidebar.markdown("""
### 🧠 About This Tool
This system predicts whether a person has **heart disease risk** or not.

---

### 📊 What You Need to Enter:

- **Age** → Your age in years  
- **Gender** → Male or Female  
- **Chest Pain Type** → Type of chest pain (0–3)  
- **Blood Pressure (BP)** → Your BP level  
- **Cholesterol** → Fat level in blood  
- **FBS over 120** → Sugar level (0 = No, 1 = Yes)  
- **EKG Results** → Heart test result  
- **Max Heart Rate** → Highest heart rate  
- **Exercise Angina** → Chest pain during exercise (0/1)  
- **ST Depression** → ECG value  
- **Slope of ST** → Heart curve type  
- **Number of vessels** → Blocked vessels count  
- **Thallium** → Heart scan result  

---

### ⚠️ Important Note:
- This is an AI prediction tool  
- It is NOT a medical diagnosis  
- Always consult a doctor for real treatment
""")

# -------------------------
# Inputs (User Friendly)
# -------------------------
st.subheader("📝 Enter Your Details Below")

Age = st.number_input("Age (in years)", 1, 120)

Sex = st.selectbox("Gender", ["Female", "Male"])
Sex = 0 if Sex == "Female" else 1

ChestPain = st.selectbox("Chest Pain Type (0 = low, 3 = high)", [0, 1, 2, 3])
BP = st.number_input("Blood Pressure (BP)")
Cholesterol = st.number_input("Cholesterol Level")
FBS = st.selectbox("Fasting Blood Sugar > 120?", [0, 1])
EKG = st.selectbox("EKG Results", [0, 1, 2])
MaxHR = st.number_input("Maximum Heart Rate")
ExerciseAngina = st.selectbox("Pain during Exercise?", [0, 1])
STDepression = st.number_input("ST Depression Value")
Slope = st.selectbox("Slope of Heart Curve", [0, 1, 2])
Vessels = st.selectbox("Number of Blocked Vessels", [0, 1, 2, 3])
Thallium = st.selectbox("Thallium Scan Result", [3, 6, 7])

# -------------------------
# Prediction
# -------------------------
if st.button("Predict Heart Disease"):

    input_data = np.array([[
        Age, Sex, ChestPain, BP, Cholesterol,
        FBS, EKG, MaxHR, ExerciseAngina,
        STDepression, Slope, Vessels, Thallium
    ]])

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    st.write("---")
    st.subheader("📊 Prediction Result")

    if prediction[0] == 1:
        st.error("❤️ High Risk: Heart Disease Detected")
    else:
        st.success("💚 Low Risk: No Heart Disease Detected")

    st.write(f"### 🔢 Risk Probability: {probability * 100:.2f}%")

# -------------------------
# Footer
# -------------------------
st.write("---")
st.write("Built with ❤️ using Machine Learning + Streamlit")'''
