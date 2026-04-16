import streamlit as st
from utils import setup_page

# Initialize the page using your utility function
setup_page("Home")


st.title("🧠 Global Mental Health Crisis Index")
st.markdown("### Welcome to the Data Mining & Analytics Dashboard")
st.markdown("""
This prototype system analyzes the intersection of **mental health, economic inequality, and digital environments** across 92 countries. 
It was engineered to uncover hidden global patterns and predict mental health crisis levels using machine learning.
""")

st.markdown("---")

# 2. Module Feature Cards using Columns
st.markdown("### 🔍 Explore the System Modules")
st.write("Use the sidebar navigation on the left to interact with the following core components:")
st.markdown("<br>", unsafe_allow_html=True) # Adding a tiny bit of vertical spacing

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 📊 Exploratory Analytics")
    st.write("Dive into the raw dataset, visualize wealth inequality disparities, and explore how daily screen time impacts youth depression prevalence.")
    st.info("Navigate to Pages 1, 2, 3 & 4")

with col2:
    st.markdown("#### 🤖 Unsupervised ML")
    st.write("Discover hidden global personas. We utilize a **K-Means Clustering** algorithm to dynamically segment countries based on their care capacity and digital habits.")
    st.info("Navigate to Page 5")

with col3:
    st.markdown("#### 🎯 Predictive Sandboxes")
    st.write("Interact with our **Random Forest** models. Tweak GDP, mental health budgets, and social media usage to predict a custom Crisis Index and Risk Tier in real-time.")
    st.info("Navigate to Pages 6 & 7")

st.markdown("---")

# 3. Project & Academic Context
st.markdown("### 🎓 Project Details")
st.write("**Institution:** University of Technology, Jamaica")
st.write("**Course:** Data Analysis - Group Project")

# You can add your names here!
st.write("**Group Members:** [Tashiek Abdul ], [Cleo Dixon], [Jheonoy]") 

st.caption("Data Sources harmonized from: WHO Mental Health Atlas 2024, GBD Study 2023 (IHME), OECD Health Statistics 2024, and DataReportal 2025.")