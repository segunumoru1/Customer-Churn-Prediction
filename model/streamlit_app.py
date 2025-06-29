import streamlit as st
import pandas as pd
import joblib
import os

# --- Load model ---

# Correct the path formatting for Windows
# model_path = r"C:\Users\IfeomaAugustaAdigwe\Desktop\Customer_Churn_Prediction_and_Model\artifacts\best_xgb_model.joblib"

model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts", "best_xgb_model.joblib"))


# Verify the path exists
if not os.path.exists(model_path):
    st.error("Model path verification failed. Please check:")
    st.error(f"1. Does the file exist at: {model_path}?")
    st.error("2. Is the filename spelled correctly?")
    st.error("3. Are all directories in the path accessible?")
    st.stop()
    
model = joblib.load(model_path)

# --- Define expected columns from training ---
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
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊")
st.title("📊 Customer Churn Prediction App")
st.write("Use the form below to predict the probability of a customer churning.")

with st.form("churn_form"):
    tenure = st.slider("Tenure (months)", 0, 72, value=12)
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0, step=1.0)
    support_calls = st.slider("Support Calls", 0, 10, value=1)
    has_contract_input = st.selectbox("Has Contract?", ["Yes", "No"])
    payment_method = st.selectbox(
        "Payment Method",
        ["Credit Card", "Bank Transfer", "Electronic Check", "Mailed Check"]
    )

    submitted = st.form_submit_button("Predict")

# --- Prediction Logic ---
if submitted:
    # Create input dictionary with dummy variables
    input_data = {
        "tenure": tenure,
        "monthly_charges": monthly_charges,
        "support_calls": support_calls,
        "has_contract": 1 if has_contract_input == "Yes" else 0,
        "has_contract_Yes": 1 if has_contract_input == "Yes" else 0,
        "payment_method_Bank Transfer": 1 if payment_method == "Bank Transfer" else 0,
        "payment_method_Cash": 0,  # Not present in UI
        "payment_method_Credit Card": 1 if payment_method == "Credit Card" else 0,
        "payment_method_Electronic Check": 1 if payment_method == "Electronic Check" else 0,
        "payment_method_Mailed Check": 1 if payment_method == "Mailed Check" else 0,
        "payment_method_PayPal": 0  # Not present in UI
    }

    input_df = pd.DataFrame([input_data])

    # Ensure all expected columns are included and in correct order
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_columns]

    # Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # --- Results ---
    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("🔮 Prediction: Customer is likely to churn")
    else:
        st.success("✅ Prediction: Customer is likely to stay")

    # Format churn probability based on scale
    if probability < 0.01:
        formatted_prob = f"{probability:.6f}"
    else:
        formatted_prob = f"{probability:.2f}"

    st.metric("Churn Probability", formatted_prob)

    st.markdown("---")
    st.caption("App developed by Ifeoma Adigwe • Powered by Streamlit")
