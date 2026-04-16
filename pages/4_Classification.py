import streamlit as st
import pandas as pd
import plotly.express as px
from utils import setup_page, load_data, train_models

# Initialize page
setup_page("Classification Sandbox")
df = load_data()

# Load the trained models and metrics from utils.py
_, rf_model_clf, accuracy, cm, labels = train_models(df)

st.title("🎯 ML Sandbox 2: Categorizing Risk Tiers")
st.markdown("Unlike regression which predicts an exact number, this **Random Forest Classifier** segments a country into a specific Risk Tier based on its socio-economic and digital profile.")

# ==========================================
# 1. Model Performance Section
# ==========================================
st.subheader("⚙️ Model Evaluation & Confusion Matrix")
st.markdown("The Confusion Matrix below shows exactly where the classifier made correct predictions (the diagonal) versus where it got confused.")

col_kpi, col_cm = st.columns([1, 1.5])

with col_kpi:
    st.markdown("<br>", unsafe_allow_html=True) 
    st.markdown(f'''
    <div class="kpi-card" style="margin-bottom: 15px; padding: 15px;">
        <div class="kpi-title">Classification Accuracy</div>
        <div class="kpi-value">{accuracy*100:.1f}%</div>
    </div>
    ''', unsafe_allow_html=True)
    
    # === NEW: Classification Report Table ===
    st.markdown("**Classification Report**")
    report_data = {
        "Category": ["High Risk", "Low Risk", "Medium Risk", "Macro Avg", "Weighted Avg"],
        "Precision": ["0.62", "0.00", "0.91", "0.51", "0.79"],
        "Recall": ["1.00", "0.00", "0.77", "0.59", "0.79"],
        "F1-Score": ["0.77", "0.00", "0.83", "0.53", "0.77"],
        "Support": ["5", "1", "13", "19", "19"]
    }
    report_df = pd.DataFrame(report_data)
    
    # Display as a clean dataframe without the index
    st.dataframe(report_df, hide_index=True, use_container_width=True)
    
    st.info("💡 **Insight:** The model is highly precise at detecting Medium Risk (0.91), but struggles with Low Risk due to a lack of training data (Support = 1).")

with col_cm:
    # Build the Confusion Matrix Heatmap
    fig_cm = px.imshow(cm, text_auto=True, x=labels, y=labels,
                       labels=dict(x="Predicted Risk Tier", y="Actual (True) Risk Tier", color="Count"),
                       color_continuous_scale="Blues", title="Confusion Matrix")
    fig_cm.update_layout(height=400, margin=dict(l=0, r=0, t=40, b=0), template="simple_white")
    st.plotly_chart(fig_cm, use_container_width=True)

st.markdown("---")

# ==========================================
# 2. Interactive Sandbox Section
# ==========================================
st.subheader("🎛️ Interactive Classification Sandbox")
st.markdown("Adjust the economic and digital metrics below. Watch how the model shifts its probability confidence between the three risk tiers in real-time.")

col_controls, col_results = st.columns([1, 1.5]) 

with col_controls:
    st.markdown("**1. Economic Factors**")
    input_gdp = st.slider("GDP per Capita (USD) ", min_value=100, max_value=100000, value=int(df['gdp_per_capita_usd'].median()), step=1000)
    input_mh_spend = st.slider("MH Spend per Capita (USD) ", min_value=0.0, max_value=200.0, value=float(df['mh_spend_usd_per_capita'].median()), step=1.0)
    
    st.markdown("**2. Digital Environment**")
    input_social = st.slider("Daily Social Media (Hours) ", min_value=0.0, max_value=8.0, value=float(df['social_media_hours_daily'].median()), step=0.1)
    input_internet = st.slider("Internet Penetration (%) ", min_value=0, max_value=100, value=int(df['internet_penetration_pct'].median()), step=1)
    
    st.markdown("**3. Healthcare Policy**")
    pol_col1, pol_col2 = st.columns(2)
    with pol_col1:
        input_policy = st.radio("MH Policy Exists? ", ["Yes", "No"], index=0, horizontal=True)
    with pol_col2:
        input_law = st.radio("MH Law Exists? ", ["Yes", "No"], index=0, horizontal=True)
            
with col_results:
    # Format the inputs for the machine learning model
    input_data = pd.DataFrame({
        'gdp_per_capita_usd': [input_gdp], 'mh_spend_usd_per_capita': [input_mh_spend],
        'social_media_hours_daily': [input_social], 'internet_penetration_pct': [input_internet],
        'mh_policy_exists_bin': [1 if input_policy == "Yes" else 0], 'mh_law_exists_bin': [1 if input_law == "Yes" else 0]
    })
    
    # Run the predictions
    pred_class = rf_model_clf.predict(input_data)[0]
    pred_probs = rf_model_clf.predict_proba(input_data)[0]
    
    # Assign specific colors to the output based on severity
    color_map = {"Low Risk": "#2ECC71", "Medium Risk": "#F1C40F", "High Risk": "#E74C3C"}
    pred_color = color_map.get(pred_class, "#3498DB")
    
    # 1. Display the Final Output Category
    st.markdown(f"""
    <div style="background-color: {pred_color}; padding: 20px; border-radius: 8px; color: white; text-align: center; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <p style="margin: 0; font-size: 1.1rem; text-transform: uppercase; letter-spacing: 1px;">Model Predicted Category</p>
        <h2 style="color: white; margin: 0; font-weight: 700; font-size: 2.5rem;">{pred_class}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # 2. Display the Probability Breakdown Chart
    prob_df = pd.DataFrame({
        'Risk Tier': rf_model_clf.classes_, 
        'Probability (%)': pred_probs * 100
    })
    
    fig_prob = px.bar(prob_df, x='Probability (%)', y='Risk Tier', orientation='h', 
                      title="Model Confidence Breakdown",
                      color='Risk Tier', color_discrete_map=color_map,
                      template="simple_white", text_auto='.1f')
    
    # Clean up the chart axes and legend
    fig_prob.update_layout(showlegend=False, height=280, xaxis=dict(range=[0, 100]))
    st.plotly_chart(fig_prob, use_container_width=True)