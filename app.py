import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Page config
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# Title
st.title("📊 Customer Churn Prediction")
st.write("Enter customer details below to predict churn.")

# Load dataset
df = pd.read_csv("data.csv")

# Remove customerID if exists
if 'customerID' in df.columns:
    df = df.drop('customerID', axis=1)

# Encode categorical columns
le_dict = {}

for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

# Split data
X = df.drop('Churn', axis=1)
y = df['Churn']

x_train, x_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# ---------------- UI ---------------- #
customer_id = st.text_input("Customer ID")

gender = st.selectbox("Gender", ["Male", "Female"])

senior = st.selectbox("Senior Citizen", [0, 1])

partner = st.selectbox("Partner", ["Yes", "No"])

dependents = st.selectbox("Dependents", ["Yes", "No"])

tenure = st.slider("Tenure (Months)", 0, 72, 12)

monthlycharges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=50.0
)

totalcharges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=500.0
)

# Convert inputs
gender = 1 if gender == "Male" else 0
partner = 1 if partner == "Yes" else 0
dependents = 1 if dependents == "Yes" else 0

# Prediction button
if st.button("Predict Churn"):

    input_data = pd.DataFrame({
        'gender': [gender],
        'SeniorCitizen': [senior],
        'Partner': [partner],
        'Dependents': [dependents],
        'tenure': [tenure],
        'MonthlyCharges': [monthlycharges],
        'TotalCharges': [totalcharges]
    })

    # Match training columns
    for col in X.columns:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[X.columns]

    prediction = model.predict(input_data)

    if prediction[0] == 1:
       st.error(f"⚠️ Customer {customer_id} is likely to Churn")
else:
       st.success(f"✅ Customer {customer_id} is likely to Stay")

# Footer
st.markdown("---")
st.caption("Built with Streamlit & Machine Learning")