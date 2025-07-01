import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Churn Predictor", page_icon="📊")
st.title("🔮 Customer Churn Prediction")

# Load model (pipeline)
pipeline = joblib.load('ensemble_pipeline.pkl')

# Input fields — customize as per your features
age = st.number_input("Age", min_value=0)
income = st.number_input("Annual Income")
transaction_amount = st.number_input("Average Transaction Amount")
days_ago = st.number_input("Last Purchase Days Ago")
gender = st.selectbox("Gender", ["Female", "Male", "Other"])
email_opt_in = st.checkbox("Email Opt-In")
promotion_response = st.selectbox("Promotion Response", ["None", "Responded", "Unsubscribed"])
spend = st.number_input("Spend per Year")
return_rate = st.number_input("Return Rate")
engagement = st.number_input("Engagement Score", min_value=0)

# One-hot encode categorical inputs consistent with training
gender_cols = {"Female": [1, 0, 0], "Male": [0, 1, 0], "Other": [0, 0, 1]}
promo_cols = {"None": [0, 0], "Responded": [1, 0], "Unsubscribed": [0, 1]}

if st.button("Predict"):
    # Build input DataFrame with correct order of features
    input_df = pd.DataFrame([[
        age, income, transaction_amount, days_ago,
        *gender_cols[gender],
        int(email_opt_in),
        *promo_cols[promotion_response],
        spend, return_rate, engagement
    ]], columns=[
        "Age", "Annual_Income", "Average_Transaction_Amount", "Last_Purchase_Days_Ago",
        "Gender_Female", "Gender_Male", "Gender_Other",
        "Email_Opt_In_True",
        "Promotion_Response_Responded", "Promotion_Response_Unsubscribed",
        "Spend_per_Year", "Return_Rate", "Engagement_Score"
    ])

    # Predict using the loaded pipeline
    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Customer is likely to churn! Probability: {probability:.2%}")
    else:
        st.success(f"✅ Customer is unlikely to churn. Probability: {probability:.2%}")
