import streamlit as st
import pandas as pd
import numpy as np
import joblib

# loading the saved model, scaler, and feature names
model = joblib.load('model/churn_model.pkl')
scaler = joblib.load('model/scaler.pkl')
feature_names = joblib.load('model/feature_names.pkl')

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Predictor")
st.markdown("Enter customer details below to predict whether they are likely to churn.")
st.markdown("---")

# input form split into 3 columns for cleaner layout
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Account Info")
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    contract = st.selectbox("Contract Type",
                            ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])

with col2:
    st.subheader("Services")
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No"])
    internet_service = st.selectbox("Internet Service",
                                    ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No"])

with col3:
    st.subheader("Personal Info")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    monthly_charges = st.slider("Monthly Charges ($)", 18, 120, 65)
    total_charges = st.slider("Total Charges ($)", 0, 9000,
                               tenure * monthly_charges)

st.markdown("---")

if st.button("Predict Churn", type="primary"):

    # building input dictionary matching training feature names exactly
    input_data = {
        'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
        'Partner': 1 if partner == "Yes" else 0,
        'Dependents': 1 if dependents == "Yes" else 0,
        'tenure': tenure,
        'PhoneService': 1 if phone_service == "Yes" else 0,
        'MultipleLines': 1 if multiple_lines == "Yes" else 0,
        'OnlineSecurity': 1 if online_security == "Yes" else 0,
        'OnlineBackup': 1 if online_backup == "Yes" else 0,
        'DeviceProtection': 0,
        'TechSupport': 0,
        'StreamingTV': 0,
        'StreamingMovies': 0,
        'PaperlessBilling': 1 if paperless_billing == "Yes" else 0,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'gender_Male': 1 if gender == "Male" else 0,
        'InternetService_Fiber optic': 1 if internet_service == "Fiber optic" else 0,
        'InternetService_No': 1 if internet_service == "No" else 0,
        'Contract_One year': 1 if contract == "One year" else 0,
        'Contract_Two year': 1 if contract == "Two year" else 0,
        'PaymentMethod_Credit card (automatic)': 1 if payment_method == "Credit card (automatic)" else 0,
        'PaymentMethod_Electronic check': 1 if payment_method == "Electronic check" else 0,
        'PaymentMethod_Mailed check': 1 if payment_method == "Mailed check" else 0,
    }

    # aligning with exact training columns — fills missing with 0
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    # scaling and predicting
    input_scaled = scaler.transform(input_df)
    prob = model.predict_proba(input_scaled)[0][1]
    prediction = model.predict(input_scaled)[0]

    # showing result
    st.markdown("## Prediction Result")
    col_res1, col_res2 = st.columns(2)

    with col_res1:
        if prediction == 1:
            st.error("⚠️ High Churn Risk")
        else:
            st.success("✅ Low Churn Risk")
        st.metric("Churn Probability", f"{prob * 100:.1f}%")

    with col_res2:
        # categorizing risk level based on probability
        if prob < 0.3:
            risk = "Low Risk"
            color = "green"
        elif prob < 0.6:
            risk = "Medium Risk"
            color = "orange"
        else:
            risk = "High Risk"
            color = "red"

        st.markdown(f"### Risk Level: :{color}[{risk}]")
        st.progress(float(prob))

    # explaining which factors are contributing to churn risk
    st.markdown("### Key Risk Factors")
    factors = []
    if contract == "Month-to-month":
        factors.append("Month-to-month contract — highest churn risk contract type")
    if tenure < 12:
        factors.append("Low tenure — new customers are at highest risk")
    if monthly_charges > 70:
        factors.append("High monthly charges — increases churn motivation")
    if internet_service == "Fiber optic":
        factors.append("Fiber optic service — associated with higher churn rate")
    if payment_method == "Electronic check":
        factors.append("Electronic check payment — less committed payment method")

    if factors:
        for f in factors:
            st.warning(f"• {f}")
    else:
        st.info("No major risk factors detected for this customer profile.")

st.markdown("---")
st.caption("Model: Logistic Regression | Accuracy: 80.4% | AUC-ROC: 0.836 | Dataset: IBM Telco Customer Churn")