import streamlit as st
import pandas as pd
import joblib
import os
from pathlib import Path

# --- Load model and feature columns ---
def load_model_and_features():
    """Load the trained model and feature columns with error handling"""
    
    # Define paths
    script_dir = Path(__file__).parent
    artifacts_dir = script_dir.parent / "artifacts"
    model_path = artifacts_dir / "best_xgb_model.joblib"
    features_path = artifacts_dir / "feature_columns.joblib"
    
    # Check if files exist
    if not model_path.exists():
        st.error("🚨 **Model not found!**")
        st.error("**Please run the following commands first:**")
        st.code("""
# 1. Navigate to project directory
cd C:\\Users\\SEGUN\\Customer-Churn-Prediction

# 2. Run preprocessing (creates processed data)
python notebooks/preprocessing.py

# 3. Train the model (creates artifacts)
python model/model.py

# 4. Then run Streamlit app
streamlit run model/streamlit_app.py
        """)
        st.stop()
    
    if not features_path.exists():
        st.error("🚨 **Feature columns not found!**")
        st.error("Please retrain the model using the updated model.py script.")
        st.stop()
    
    # Load model and features
    try:
        model = joblib.load(model_path)
        feature_columns = joblib.load(features_path)
        st.success("✅ Model and features loaded successfully!")
        return model, feature_columns
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

# Load model and features
model, feature_columns = load_model_and_features()

# --- Streamlit UI ---
st.set_page_config(page_title="Customer Churn Predictor", page_icon="🔍")
st.title("🔍 Customer Churn Prediction App")
st.write("Use the form below to input customer details and estimate churn likelihood.")

# Create two columns for better layout
col1, col2 = st.columns(2)

with st.form("churn_form"):
    with col1:
        tenure = st.slider("Tenure (months)", 0, 72, value=12)
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0, step=1.0)
        support_calls = st.slider("Number of Support Calls", 0, 10, value=1)
    
    with col2:
        has_contract = st.selectbox("Customer Has Contract?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        payment_method = st.selectbox(
            "Payment Method",
            ["Bank Transfer", "Cash", "Credit Card", "PayPal"]
        )
    
    submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True)

# --- Prediction logic ---
if submitted:
    with st.spinner("Making prediction..."):
        try:
            # Create input data
            input_data = {
                "tenure": tenure,
                "monthly_charges": monthly_charges,
                "support_calls": support_calls,
                "has_contract": has_contract,
                "payment_method": payment_method
            }
            
            # Create DataFrame
            df_input = pd.DataFrame([input_data])
            
            # One-hot encode to match training data
            df_encoded = pd.get_dummies(df_input, columns=['has_contract', 'payment_method'], prefix=['has_contract', 'payment_method'])
            
            # Ensure all feature columns exist (add missing columns with 0)
            for col in feature_columns:
                if col not in df_encoded.columns:
                    df_encoded[col] = 0
            
            # Reorder columns to match training data
            df_final = df_encoded[feature_columns]
            
            # Make prediction
            prediction = model.predict(df_final)[0]
            probability = model.predict_proba(df_final)[0][1]

            st.markdown("---")
            st.subheader("🧠 Prediction Result")

            if prediction == 1:
                st.error("🔮 This customer is **likely to churn**.")
                st.write("💡 **Recommendation:** Consider retention strategies like discounts or improved service.")
            else:
                st.success("✅ This customer is **likely to stay**.")
                st.write("💡 **Great!** Continue providing excellent service to maintain satisfaction.")

            # Display probability with better formatting
            prob_percentage = probability * 100
            st.metric(
                label="Churn Probability", 
                value=f"{prob_percentage:.1f}%",
                delta=f"{'High Risk' if prob_percentage > 50 else 'Low Risk'}"
            )
            
            # Show input summary
            with st.expander("📊 Input Summary"):
                st.write("**Customer Profile:**")
                st.write(f"- Tenure: {tenure} months")
                st.write(f"- Monthly Charges: ${monthly_charges:.2f}")
                st.write(f"- Support Calls: {support_calls}")
                st.write(f"- Has Contract: {'Yes' if has_contract == 1 else 'No'}")
                st.write(f"- Payment Method: {payment_method}")
            
            # Show encoded features (for debugging)
            with st.expander("🔧 Technical Details"):
                st.write("**Encoded Features:**")
                st.dataframe(df_final)

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            st.error("Please check that the model was trained with the correct features.")

st.markdown("---")
st.caption("Created by Ifeoma Adigwe • Powered by Streamlit + XGBoost")
