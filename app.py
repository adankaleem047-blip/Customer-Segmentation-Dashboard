import streamlit as st
import pandas as pd
import joblib

# Page settings
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide"
)

# Load model and dataset
model = joblib.load("customer_segmentation_model.pkl")
df = pd.read_csv("Mall_Customers.csv")

# Title
st.title("📊 Customer Segmentation Dashboard")
st.write(
    "Analyze customer behavior using Machine Learning and K-Means Clustering."
)

# Dataset Preview
with st.expander("View Dataset"):
    st.dataframe(df.head())

st.divider()

# User Inputs
st.subheader("Enter Customer Information")

income = st.slider(
    "Annual Income (k$)",
    min_value=0,
    max_value=150,
    value=60
)

score = st.slider(
    "Spending Score (1-100)",
    min_value=1,
    max_value=100,
    value=50
)

# Prediction
if st.button("Analyze Customer"):

    cluster = model.predict([[income, score]])[0]

    cluster_info = {
        0: {
            "name": "Premium Customers",
            "description": "High income and high spending customers.",
            "recommendation": "Offer luxury products and VIP memberships."
        },
        1: {
            "name": "Budget Customers",
            "description": "Low income and low spending customers.",
            "recommendation": "Provide discounts and promotional offers."
        },
        2: {
            "name": "High Income - Low Spending",
            "description": "Customers with purchasing power but low engagement.",
            "recommendation": "Use personalized marketing campaigns."
        },
        3: {
            "name": "Regular Customers",
            "description": "Customers with average income and spending habits.",
            "recommendation": "Offer loyalty programs."
        },
        4: {
            "name": "High Spending Customers",
            "description": "Customers who spend frequently.",
            "recommendation": "Provide rewards and exclusive benefits."
        }
    }

    result = cluster_info[cluster]

    st.divider()

    st.subheader("Analysis Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Annual Income", f"${income}k")

    with col2:
        st.metric("Spending Score", score)

    st.success(f"Customer Segment: {result['name']}")

    st.write(f"**Description:** {result['description']}")

    st.info(f"Recommendation: {result['recommendation']}")