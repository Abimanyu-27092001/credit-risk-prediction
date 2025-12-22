import streamlit as st
import pandas as pd
import joblib

st.title("Credit Risk Prediction")

model = joblib.load("artifacts/best_model_lgb.joblib")
preprocessor = joblib.load("artifacts/preprocessor.joblib")

credit_amount = st.number_input("Credit Amount", 1000, 100000)
duration = st.slider("Loan Duration (months)", 6, 72)

# Minimal safe defaults for remaining features
input_data = {
    "Credit_amount": credit_amount,
    "Duration_months": duration,
    "Age_years": 35,
    "Installment_rate": 2,
    "Savings_account_bonds": "A61",
    "Status_checking_account": "A11",
    "Credit_history": "A34",
    "Purpose": "A43",
    "Present_employment_since": "A72",
    "Personal_status_sex": "A93",
    "Other_debtors_guarantors": "A101",
    "Present_residence_since": 2,
    "Property": "A121",
    "Other_installment_plans": "A141",
    "Housing": "A152",
    "Number_existing_credits": 1,
    "Job": "A173",
    "Number_people_liable": 1,
    "Telephone": "A191",
    "Foreign_worker": "A201",
}

input_df = pd.DataFrame([input_data])
X_input = preprocessor.transform(input_df)

if st.button("Predict"):
    prob = model.predict_proba(X_input)[0, 1]
    st.success(f"Default Risk Probability: {prob:.2%}")
