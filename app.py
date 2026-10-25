import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from notebooks.data_processing import feature_engineer

# Set up page styling for a banking app
st.set_page_config(
    page_title="Credit Risk Dashboard",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Credit Risk & Loan Default Engine")
st.markdown("Enter applicant attributes below to evaluate default probabilities and loan approval status.")

#1: Model Loading Path ---
@st.cache_resource
def load_model():
    model_path = Path("models/loan_rf_model.pkl")
    scaler_path = Path("models/scaler.pkl")
    feature_columns_path = Path("models/feature_columns.pkl")
    if not (model_path.exists() and scaler_path.exists()):
        st.error(f"Model, scaler and feature columns file not found at {model_path} and {scaler_path}. Please train and export your model first.")
        return None, None, None

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_columns = joblib.load(feature_columns_path)
    return model, scaler, feature_columns

model, scaler, feature_columns = load_model()

if (model is not None) and (scaler is not None) and (feature_columns is not None):
    # Organize inputs into matching layout columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.header("👤 Applicant Profile")
        person_age = st.number_input("Applicant Age", min_value=18, max_value=100, value=30)
        person_income = st.number_input("Annual Income ($)", min_value=4000, max_value=5000000, value=60000, step=5000)
        person_emp_length = st.number_input("Employment Length (Years)", min_value=0.0, max_value=60.0, value=4.0, step=0.5)
        person_home_ownership = st.selectbox("Home Ownership Status", ["RENT", "MORTGAGE", "OWN", "OTHER"])

    with col2:
        st.header("💰 Loan Request Details")
        loan_intent = st.selectbox("Loan Intent", ["EDUCATION", "MEDICAL", "PERSONAL", "VENTURE", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
        loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
        loan_amnt = st.number_input("Requested Loan Amount ($)", min_value=500, max_value=35000, value=10000, step=1000)
        loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=30.0, value=11.0, step=0.1)

    with col3:
        st.header("📜 Credit History")
        cb_person_default_on_file = st.selectbox("Historical Default on File?", ["N", "Y"])
        cb_person_cred_hist_length = st.number_input("Credit History Length (Years)", min_value=0, max_value=40, value=5)
        
        # RISK THRESHOLD SLIDER - shifting cutoff in real-time
        st.markdown("---")
        st.subheader("⚙️ Risk Boundary")
        risk_threshold = st.slider("Max Acceptable Default Probability Cutoff", min_value=0.30, max_value=0.70, value=0.50, step=0.01)
        st.caption("Lowering this catches more defaulters early (protects bank capital).")

    # Convert 'loan_grade' string to matching ordinal integer mapping
    grade_mapping = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
    loan_grade_ord = grade_mapping[loan_grade]

    # Pre-calculate base loan metric matching raw training setup
    loan_percent_income = round(loan_amnt / person_income, 2)

    interest_burden = loan_int_rate * loan_percent_income
    disposable_income_proxy = person_income - loan_amnt
    loan_to_annual_income_ratio = loan_amnt / person_income

    # 4. Pack into a dictionary matching exact input DataFrame columns 
    input_data = {
        "person_age": [person_age],
        "person_income": [person_income],
        "person_home_ownership": [person_home_ownership],
        "person_emp_length": [person_emp_length],
        "loan_intent": [loan_intent],
        "loan_amnt": [loan_amnt],
        "loan_int_rate": [loan_int_rate],
        "loan_percent_income": [loan_percent_income],
        "cb_person_default_on_file": [cb_person_default_on_file],
        "cb_person_cred_hist_length": [cb_person_cred_hist_length],
        "loan_grade_ord": [loan_grade_ord],
        "interest_burden": [interest_burden],
        "disposable_income_proxy": [disposable_income_proxy],
        "loan_to_annual_income_ratio": [loan_to_annual_income_ratio]
    }
    
    input_df = pd.DataFrame(input_data)
    input_df = feature_engineer(input_df)
    input_df = pd.get_dummies(input_df, columns=['person_home_ownership', 'loan_intent', 'cb_person_default_on_file'], drop_first=True, dtype=int)
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)
    cols = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'disposable_income_proxy', 'interest_burden']
    input_df[cols] = scaler.transform(input_df[cols])

    st.markdown("---")
    if st.button("🚀 Evaluate Decision", use_container_width=True):
        # Predict continuous class default probabilities
        probability = model.predict_proba(input_df) # [[0.90, 0.10]]   90% chance of 0, 10% chance of 1
        predict = model.predict(input_df)
        
        # Apply custom dynamic risk cutoff slider value
        decision = "DENIED" if probability[0][1] >= risk_threshold else "APPROVED"

        # Show Output Metrics
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            a, b, c = st.columns(3)
            with a:
                st.metric(label="0 Calculated NOT Default Probability", value=f"{probability[0][0] * 100:.2f}%")
            with b:
                st.metric(label="1 Calculated Default Probability", value=f"{probability[0][1] * 100:.2f}%")
            with c:
                st.metric(label="Model Prediction", value=f"{predict[0]}")
            if probability[0][1] >= risk_threshold:
                st.warning("⚠️ High Default Risk Concentration Flagged!")
            else:
                st.success("✅ Risk profile stays within bounds.")
                
        with res_col2:
            if decision == "APPROVED":
                st.subheader("Decision Status:")
                st.title("🟢 LOAN APPROVED")
            else:
                st.subheader("Decision Status:")
                st.title("🔴 LOAN DENIED")
        
