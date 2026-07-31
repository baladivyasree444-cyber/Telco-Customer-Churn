import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_model():
    return joblib.load("telco_customer_churn.pkl")


model = load_model()

st.title("Telco-Customer-Churn-Prediction")
st.markdown("""
Predict whether a telecom customer is likely to **Churn** or **Stay**.
""")
st.sidebar.header("Customer Information")

gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

SeniorCitizen = st.sidebar.selectbox(
    "Senior Citizen",
    [0, 1]
)
Partner = st.sidebar.selectbox(
    "Partner",
    ["Yes", "No"]
)

Dependents = st.sidebar.selectbox(
    "Dependents",
    ["Yes", "No"]
)

tenure = st.sidebar.slider(
    "Tenure (Months)",
    0,
    72,
    24
)

PhoneService = st.sidebar.selectbox(
    "Phone Service",
    ["Yes", "No"]
)

MultipleLines = st.sidebar.selectbox(
    "Multiple Lines",
    ["Yes", "No", "No phone service"]
)

InternetService = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

OnlineSecurity = st.sidebar.selectbox(
    "Online Security",
    ["Yes", "No", "No internet service"]
)

OnlineBackup = st.sidebar.selectbox(
    "Online Backup",
    ["Yes", "No", "No internet service"]
)

DeviceProtection = st.sidebar.selectbox(
    "Device Protection",
    ["Yes", "No", "No internet service"]
)

TechSupport = st.sidebar.selectbox(
    "Tech Support",
    ["Yes", "No", "No internet service"]
)

StreamingTV = st.sidebar.selectbox(
    "Streaming TV",
    ["Yes", "No", "No internet service"]
)

StreamingMovies = st.sidebar.selectbox(
    "Streaming Movies",
    ["Yes", "No", "No internet service"]
)

Contract = st.sidebar.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

PaperlessBilling = st.sidebar.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

PaymentMethod = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

MonthlyCharges = st.sidebar.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=200.0,
    value=70.0
)

TotalCharges = st.sidebar.number_input(
    "Total Charges",
    min_value=0.0,
    value=1500.0
)

input_df = pd.DataFrame({

    "gender":[gender],
    "SeniorCitizen":[SeniorCitizen],
    "Partner":[Partner],
    "Dependents":[Dependents],
    "tenure":[tenure],
    "PhoneService":[PhoneService],
    "MultipleLines":[MultipleLines],
    "InternetService":[InternetService],
    "OnlineSecurity":[OnlineSecurity],
    "OnlineBackup":[OnlineBackup],
    "DeviceProtection":[DeviceProtection],
    "TechSupport":[TechSupport],
    "StreamingTV":[StreamingTV],
    "StreamingMovies":[StreamingMovies],
    "Contract":[Contract],
    "PaperlessBilling":[PaperlessBilling],
    "PaymentMethod":[PaymentMethod],
    "MonthlyCharges":[MonthlyCharges],
    "TotalCharges":[TotalCharges]

})

st.subheader("Customer Information")

st.dataframe(input_df, use_container_width=True)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Churn"):

    prediction = model.predict(input_df)[0]

    try:
        probability = model.predict_proba(input_df)[0]
    except:
        probability = None

    st.markdown("---")

    if prediction == 1:

        st.error("⚠️ Customer is likely to CHURN.")

    else:

        st.success("✅ Customer is likely to STAY.")

    if probability is not None:

        st.subheader("Prediction Probability")

        st.write(f"Stay Probability : **{probability[0]*100:.2f}%**")

        st.write(f"Churn Probability : **{probability[1]*100:.2f}%**")
 
        st.progress(float(probability[1]))
