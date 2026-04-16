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


# 3. Project & Academic Context
st.markdown("### 🎓 Project Details")


st.caption("Data Sources harmonized from: WHO Mental Health Atlas 2024, GBD Study 2023 (IHME), OECD Health Statistics 2024, and DataReportal 2025.")