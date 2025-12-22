import streamlit as st
import pandas as pd
import joblib

st.title("Credit Risk Prediction")

# Load artifacts
model = joblib.load("artifacts/best_model_lgb.joblib")
preprocessor = joblib.load("artifacts/preprocessor.joblib")
THRESHOLD = float(open("artifacts/chosen_threshold.txt").read())

# Inputs
credit_amount = st.number_input("Credit Amount", min_value=1000, max_value=100000, value=10000)
duration = st.slider("Loan Duration (months)", min_value=6, max_value=72, value=24)

# Minimal safe defaults
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

# Prediction
if st.button("Predict"):
    prob = model.predict_proba(X_input)[:, 1][0]

    st.markdown("### Prediction Result")

    st.write(f"**Default Risk Probability:** {prob * 100:.2f}%")

    # Decision label
    if prob >= THRESHOLD:
        st.error("⚠️ High Default Risk")
    else:
        st.success("✅ Low Default Risk")

    # Threshold info
    st.caption(f"Decision threshold used: {THRESHOLD:.2f}")

    st.markdown("---")
    st.caption(
        "⚠️ This application is for educational and demonstration purposes only. "
        "It does not constitute a real credit decision system."
    )

