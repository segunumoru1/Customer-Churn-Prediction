import streamlit as st
import pandas as pd
import joblib
import os

# --- Load model ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.abspath(os.path.join(BASE_DIR, "..", "artifacts", "best_xgb_model.joblib"))

# Debug: Show resolved path
# st.write(f"Resolved model path: {model_path}")

if not os.path.exists(model_path):
    st.error("🚨 Model path verification failed.")
    st.error(f"🧭 Checked path: {model_path}")
    st.error("Please make sure:")
    st.error("• The model exists in the /artifacts folder (one level above this file)")
    st.error("• The file name is spelled correctly")
    st.stop()

model = joblib.load(model_path)

# --- Define expected model features ---
expected_columns = [
    "tenure",
    "monthly_charges",
    "support_calls",
    "has_contract",
    "has_contract_Yes",
    "payment_method_Bank Transfer",
    "payment_method_Cash",
    "payment_method_Credit Card",
    "payment_method_Electronic Check",
    "payment_method_Mailed Check",
    "payment_method_PayPal"
]

# --- Streamlit UI ---
st.set_page_config(page_title="Customer Churn Predictor", page_icon="🔍")
st.title("🔍 Customer Churn Prediction App")
st.write("Use the form below to input customer details and estimate churn likelihood.")

with st.form("churn_form"):
    tenure = st.slider("Tenure (months)", 0, 72, value=12)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0, step=1.0)
    support_calls = st.slider("Number of Support Calls", 0, 10, value=1)
    has_contract_input = st.selectbox("Customer Has Contract?", ["Yes", "No"])
    payment_method = st.selectbox(
        "Preferred Payment Method",
        ["Credit Card", "Bank Transfer", "Electronic Check", "Mailed Check"]
    )
    submitted = st.form_submit_button("Predict")

# --- Prediction logic ---
if submitted:
    input_data = {
        "tenure": tenure,
        "monthly_charges": monthly_charges,
        "support_calls": support_calls,
        "has_contract": int(has_contract_input == "Yes"),
        "has_contract_Yes": int(has_contract_input == "Yes"),
        "payment_method_Bank Transfer": int(payment_method == "Bank Transfer"),
        "payment_method_Cash": 0,
        "payment_method_Credit Card": int(payment_method == "Credit Card"),
        "payment_method_Electronic Check": int(payment_method == "Electronic Check"),
        "payment_method_Mailed Check": int(payment_method == "Mailed Check"),
        "payment_method_PayPal": 0
    }

    df = pd.DataFrame([input_data])

    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[expected_columns]

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    st.markdown("---")
    st.subheader("🧠 Prediction Result")

    if prediction == 1:
        st.error("🔮 This customer is likely to churn.")
    else:
        st.success("✅ This customer is likely to stay.")

    formatted_prob = f"{probability:.6f}" if probability < 0.01 else f"{probability:.2%}"
    st.metric(label="Churn Probability", value=formatted_prob)

    st.markdown("---")
    st.caption("Created by Ifeoma Adigwe • Powered by Streamlit + XGBoost")
