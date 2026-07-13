import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
# ----------------------------------
# PAGE CONFIG
# ----------------------------------

st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

# ----------------------------------
# CUSTOM CSS
# ----------------------------------

st.markdown("""
<style>
.kpi-card {
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    color: white;
    font-weight: bold;
}

.blue {
    background: linear-gradient(135deg, #2193b0, #6dd5ed);
}

.green {
    background: linear-gradient(135deg, #11998e, #38ef7d);
}

.orange {
    background: linear-gradient(135deg, #ff8008, #ffc837);
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------
# SIDEBAR
# ----------------------------------

st.sidebar.title("📂 Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Dashboard",
        "Analytics",
        "Prediction",
        "About Project"
    ]
)

# ----------------------------------
# TITLE
# ----------------------------------
if page == "Dashboard":
    st.title("📊 Customer Churn Prediction Dashboard")
    st.write("Machine Learning Based Customer Churn Analysis & Prediction")

# ----------------------------------
# LOAD MODEL FILES
# ----------------------------------

try:

    with open("models/churn_model.pkl", "rb") as file:
        model = pickle.load(file)

    with open("models/scaler.pkl", "rb") as file:
        scaler = pickle.load(file)

    with open("models/feature_names.pkl", "rb") as file:
        feature_names = pickle.load(file)

    st.success("Model Loaded Successfully ✅")

except Exception as e:
    st.error(f"Model Loading Error : {e}")

# ----------------------------------
# LOAD DATASET
# ----------------------------------

try:
    df = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
except:
    df = None

# ----------------------------------
# KPI SECTION
# ----------------------------------

st.header("Project Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="kpi-card blue">
    <h3>🤖 Model</h3>
    <h2>Logistic Regression</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="kpi-card green">
    <h3>🎯 Accuracy</h3>
    <h2>80.03%</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="kpi-card orange">
    <h3>📈 ROC-AUC</h3>
    <h2>0.835</h2>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------------
# DATASET INFO
# ----------------------------------

st.header("Dataset Information")

st.write("""
- Total Customers : 7043
- Churn Customers : 1869
- Retained Customers : 5174
- Features Used : 30
""")

# ----------------------------------
# DATA PREVIEW
# ----------------------------------

if df is not None:
    st.header("Dataset Preview")
    st.dataframe(df.head())

# ----------------------------------
# ANALYTICS DASHBOARD
# ----------------------------------
if page == "Analytics":
    st.header("📊 Customer Churn Analytics")

    if df is not None:

        col1, col2 = st.columns(2)

        # Churn Distribution

        with col1:

            churn_counts = df["Churn"].value_counts()

            fig = px.pie(
                values=churn_counts.values,
                names=churn_counts.index,
                title="Customer Churn Distribution"
            )

            st.plotly_chart(fig, use_container_width=True)

        # Contract Distribution

        with col2:

            fig = px.bar(
                df["Contract"].value_counts(),
                title="Contract Type Distribution"
            )

            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)

        # Internet Service

        with col3:

            fig = px.bar(
                df["InternetService"].value_counts(),
                title="Internet Service Distribution"
            )

            st.plotly_chart(fig, use_container_width=True)

        # Payment Method

        with col4:

            fig = px.bar(
                df["PaymentMethod"].value_counts(),
                title="Payment Method Distribution"
            )

            st.plotly_chart(fig, use_container_width=True)

# ----------------------------------
# BUSINESS INSIGHTS
# ----------------------------------

st.header("💡 Business Insights")

st.info("""
✅ Customer Retention Rate : 73.4%

⚠️ Churn Rate : 26.6%

📌 Month-to-Month customers show higher churn risk.

📌 Fiber Optic users are more likely to churn.

📌 Customers with longer tenure are less likely to leave.

📌 Electronic Check users show relatively higher churn behavior.
""")
# ----------------------------------
# CUSTOMER INPUT
# ----------------------------------

st.header("🔍 Customer Churn Prediction")

col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure = st.slider(
        "Tenure (Months)",
        0,
        72,
        12
    )

with col2:

    monthly = st.number_input(
        "Monthly Charges",
        min_value=18.0,
        max_value=120.0,
        value=70.0
    )

    contract = st.selectbox(
        "Contract Type",
        ["Month-to-month", "One year", "Two year"]
    )

    internet = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

# ----------------------------------
# PREDICTION
# ----------------------------------
if page == "Prediction":
    if st.button("🚀 Predict Churn Risk"):

        try:

            input_data = pd.DataFrame(
                np.zeros((1, len(feature_names))),
                columns=feature_names
            )

            # Numerical

            input_data["SeniorCitizen"] = senior
            input_data["tenure"] = tenure
            input_data["MonthlyCharges"] = monthly
            input_data["TotalCharges"] = tenure * monthly

            # Gender

            if gender == "Male":
                input_data["gender_Male"] = 1

            # Partner

            if partner == "Yes":
                input_data["Partner_Yes"] = 1

            # Dependents

            if dependents == "Yes":
                input_data["Dependents_Yes"] = 1

            # Internet

            if internet == "Fiber optic":
                input_data["InternetService_Fiber optic"] = 1

            elif internet == "No":
                input_data["InternetService_No"] = 1

            # Contract

            if contract == "One year":
                input_data["Contract_One year"] = 1

            elif contract == "Two year":
                input_data["Contract_Two year"] = 1

            # Payment

            if payment == "Electronic check":
                input_data["PaymentMethod_Electronic check"] = 1

            elif payment == "Mailed check":
                input_data["PaymentMethod_Mailed check"] = 1

            elif payment == "Credit card (automatic)":
                input_data["PaymentMethod_Credit card (automatic)"] = 1

            # Scale

            scaled_input = scaler.transform(input_data)

            # Predict

            prediction = model.predict(scaled_input)[0]
            probability = model.predict_proba(scaled_input)[0][1]

            st.divider()
            st.subheader("Prediction Result")

            if prediction == 1:

                st.error(
                    f"⚠️ Customer Likely To Churn\n\nRisk Probability: {probability:.2%}"
                )

            else:

                st.success(
                    f"✅ Customer Likely To Stay\n\nRisk Probability: {probability:.2%}"
                )

            # Risk Level

            if probability < 0.30:
                st.success("🟢 Low Risk Customer")

            elif probability < 0.70:
                st.warning("🟡 Medium Risk Customer")

            else:
                st.error("🔴 High Risk Customer")

        except Exception as e:
            st.error(f"Prediction Error: {e}")

        st.subheader("📌 Customer Summary")
        st.write(f"""
        - Gender : {gender}
        - Senior Citizen : {senior}
        - Partner : {partner}
        - Dependents : {dependents}
        - Tenure : {tenure} Months
        - Monthly Charges : ₹{monthly}
        - Contract : {contract}
        - Internet Service : {internet}
        - Payment Method : {payment}
        """)

        st.subheader("🔥 Top Churn Factors")

        st.info("""
        1. Short Tenure Customers are more likely to churn.
        2. Fiber Optic Users show higher churn behaviour.
        3. Month-to-Month contracts increase churn risk.
        4. Electronic Check users churn more often.
        5. Higher Monthly Charges increase churn probability.
        """)
        st.header("📊 Feature Importance")

        feature_imp = pd.DataFrame({
            "Feature": [
                "Tenure",
                "Monthly Charges",
                "Contract (Two Year)",
                "Fiber Optic Internet",
                "Total Charges"
            ],
            "Importance": [
                1.35,
                0.85,
                0.60,
                0.73,
                0.64
            ]
        })

        st.bar_chart(
            feature_imp.set_index("Feature")
        )
        st.header("🏆 Model Performance")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Accuracy",
                "80.38%"
            )

        with col2:
            st.metric(
                "ROC-AUC",
                "0.836"
            )

        with col3:
            st.metric(
                "CV Accuracy",
                "80.24%"
            )
        st.subheader("🔥 Confusion Matrix Heatmap")

        cm = np.array([
            [912, 121],
            [160, 214]
        ])

        fig, ax = plt.subplots(figsize=(6,4))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Predicted No", "Predicted Yes"],
            yticklabels=["Actual No", "Actual Yes"],
            ax=ax
        )

        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        st.pyplot(fig)
# ----------------------------------
# ABOUT PROJECT
# ----------------------------------

if page == "About Project":

    st.title("ℹ️ About Project")

    st.subheader("Project Name")

    st.write(
        "Customer Churn Analysis and Prediction Dashboard"
    )

    st.subheader("Problem Statement")

    st.write("""
    Customer churn is one of the major challenges
    faced by telecom companies.

    This project predicts whether a customer
    is likely to leave the company using
    Machine Learning techniques.
    """)

    st.subheader("Objectives")

    st.write("""
    • Analyze customer behavior

    • Identify churn patterns

    • Predict future churn

    • Improve customer retention
    """)

    st.subheader("Technology Stack")

    st.write("""
    • Python

    • Pandas

    • NumPy

    • Scikit-Learn

    • Streamlit

    • Plotly
    """)

    st.subheader("Model Performance")

    st.success("""
    Model : Logistic Regression

    Accuracy : 80.03%

    ROC-AUC : 0.835
    """)