# deployment.py  WORKING VERSION
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import pickle

# Load model
# @st.cache_resource
# def load_model():
#     return joblib.load('cvd_risk_rf_model.pkl')

# Load model
with open("logistic_reg_model.pkl", "rb") as f:
    model = pickle.load(f)

# model = load_model()

# Get exact column names the model expects
try:
    feature_names = model.named_steps['preprocessor'].get_feature_names_out()
except:
    feature_names = model[:-1].get_feature_names_out()  # fallback

st.set_page_config(page_title="CVD Risk Predictor", page_icon="Heart")
st.title("10-Year Cardiovascular Disease Risk Predictor")
st.markdown("#### Developers: Akampa Godfrey - Nazziwa Rhoda - Nameeru Bronah")
st.success("Welcome Back to Check on your Health Status👩‍⚕️! Ready for predictions👇")

# USE st.form + st.form_submit_button
with st.form("Patient_Form"):
    st.header("Enter Patient Details")

    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 30, 90, 60)
        gender = st.selectbox("Gender", ["Male", "Female"])
        systolic_bp = st.slider("Systolic BP (mmHg)", 90, 220, 130)
        bmi = st.slider("BMI", 15.0, 50.0, 27.0, 0.1)

    with col2:
        smoker = st.selectbox("Smoker?", ["No", "Yes"])
        diabetes = st.selectbox("Diabetes?", ["No", "Yes"])
        hypertension = st.selectbox("Treated Hypertension?", ["No", "Yes"])
        family_hx = st.selectbox("Family History of CVD?", ["No", "Yes"])

    col3, col4 = st.columns(2)
    with col3:
        afib = st.selectbox("Atrial Fibrillation?", ["No", "Yes"])
        ckd = st.selectbox("Chronic Kidney Disease?", ["No", "Yes"])


    # THIS IS THE REQUIRED SUBMIT BUTTON
    submitted = st.form_submit_button("Predict 10-Year Risk")

if submitted:
    # 1. Prepare input dictionary with binary encoding
    input_dict = {
        'age': age,
        'gender': 1 if gender == "Male" else 0,
        'systolic_blood_pressure': systolic_bp,
        'body_mass_index': bmi,
        'smoker': 1 if smoker == "Yes" else 0,
        'diabetes': 1 if diabetes == "Yes" else 0,
        'hypertension_treated': 1 if hypertension == "Yes" else 0,
        'family_history_of_cardiovascular_disease': 1 if family_hx == "Yes" else 0,
        'atrial_fibrillation': 1 if afib == "Yes" else 0,
        'chronic_kidney_disease': 1 if ckd == "Yes" else 0,
        # Missing features with default 0
        'forced_expiratory_volume_1': 0,
        'chronic_obstructive_pulmonary_disorder': 0,
        'rheumatoid_arthritis': 0,
        'time_to_event_or_censoring': 0
    }

    # 2. Align DataFrame columns to model
    df_input = pd.DataFrame([input_dict])
    df_input = df_input[model.named_steps['preprocessor'].feature_names_in_]

    # 3. Predict probability and class
    risk_prob = model.predict_proba(df_input)[0][1]
    risk_class = model.predict(df_input)[0]

    # 4. Map prediction to clinical interpretation
    if risk_class == 0:
        interpretation = "Low risk-unlikely to experience a cardiovascular event in the next 10 years."
        color = "green"
    else:
        interpretation = "High risk-likely to experience a cardiovascular event in the next 10 years."
        # st.info("This patient qualifies for preventive medication based on international guidelines.")
        color = "red"

    # 5. Display
    st.subheader("Predicted 10-Year CVD Risk")
    st.metric("Risk Probability", f"{risk_prob*100:.2f}%")
    st.balloons()
    # st.info(interpretation)
    st.markdown(f"<span style='color:{color};font-weight:bold'>{interpretation}</span>", unsafe_allow_html=True)


