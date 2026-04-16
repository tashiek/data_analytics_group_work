import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from utils import setup_page, load_data

# Initialize page
setup_page("Regression Sandbox")
df = load_data()

st.title("📈 ML Sandbox 1: Predicting the Crisis Index")
st.markdown("Compare how a **Linear Regression** model and a **Random Forest Regressor** estimate the exact numeric Mental Health Crisis Index. Explore their accuracy, and test hypothetical scenarios below.")

# ==========================================
# 1. Train Models & Calculate Metrics
# ==========================================
features = ['gdp_per_capita_usd', 'mh_spend_usd_per_capita', 'social_media_hours_daily', 
            'internet_penetration_pct', 'mh_policy_exists_bin', 'mh_law_exists_bin']
X = df[features]
y = df['mh_crisis_index']

# Split data to get honest evaluation metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

# Train Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# Calculate Metrics
metrics_data = {
    "Metric": ["R² Score (Accuracy)", "Mean Absolute Error (MAE)", "Root Mean Squared Error (RMSE)"],
    "Linear Regression": [
        f"{r2_score(y_test, lr_preds):.3f}",
        f"{mean_absolute_error(y_test, lr_preds):.2f}",
        f"{np.sqrt(mean_squared_error(y_test, lr_preds)):.2f}"
    ],
    "Random Forest": [
        f"{r2_score(y_test, rf_preds):.3f}",
        f"{mean_absolute_error(y_test, rf_preds):.2f}",
        f"{np.sqrt(mean_squared_error(y_test, rf_preds)):.2f}"
    ]
}
metrics_df = pd.DataFrame(metrics_data)

# Retrain on ALL data for the interactive sandbox predictions
lr_model_full = LinearRegression().fit(X, y)
rf_model_full = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)

# ==========================================
# 2. Visualizations: Actual vs Predicted
# ==========================================
st.subheader("📊 Model Performance (Actual vs. Predicted)")
st.markdown("If a model is perfectly accurate, all dots will fall exactly on the red trendline (where Actual = Predicted).")

col_plot1, col_plot2 = st.columns(2)

# Helper function to create the plots
def create_pred_plot(y_true, y_pred, title, color):
    # We use OLS trendline to see the actual line of best fit, 
    # and we can visually compare it to a perfect 1:1 ratio
    temp_df = pd.DataFrame({'Actual': y_true, 'Predicted': y_pred})
    fig = px.scatter(temp_df, x='Actual', y='Predicted', trendline="ols", 
                     title=title, template="simple_white", 
                     color_discrete_sequence=[color])
    
    # Add a Perfect Prediction reference line (y=x)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], 
                             mode='lines', name='Perfect Prediction',
                             line=dict(color='red', dash='dash')))
    return fig

with col_plot1:
    st.plotly_chart(create_pred_plot(y_test, lr_preds, "Linear Regression", "#3498DB"), use_container_width=True)

with col_plot2:
    st.plotly_chart(create_pred_plot(y_test, rf_preds, "Random Forest Regressor", "#2ECC71"), use_container_width=True)

# Metrics Table
st.markdown("#### Evaluation Metrics")
st.dataframe(metrics_df, use_container_width=True, hide_index=True)

st.markdown("---")

# ==========================================
# 3. Interactive Predictive Sandbox
# ==========================================
st.subheader("🎛️ Interactive Predictive Sandbox")
st.markdown("Adjust the variables below to see how each algorithm interprets the data to predict the final Mental Health Crisis Index.")

col_controls, col_gauges = st.columns([1, 1.5]) 

with col_controls:
    st.markdown("**1. Economic Factors**")
    input_gdp = st.slider("GDP per Capita (USD)", min_value=100, max_value=100000, value=int(df['gdp_per_capita_usd'].median()), step=1000)
    input_mh_spend = st.slider("MH Spend per Capita (USD)", min_value=0.0, max_value=200.0, value=float(df['mh_spend_usd_per_capita'].median()), step=1.0)
    
    st.markdown("**2. Digital Environment**")
    input_social = st.slider("Daily Social Media (Hours)", min_value=0.0, max_value=8.0, value=float(df['social_media_hours_daily'].median()), step=0.1)
    input_internet = st.slider("Internet Penetration (%)", min_value=0, max_value=100, value=int(df['internet_penetration_pct'].median()), step=1)
    
    st.markdown("**3. Healthcare Policy**")
    pol_col1, pol_col2 = st.columns(2)
    with pol_col1:
        input_policy = st.radio("MH Policy Exists?", ["Yes", "No"], horizontal=True)
    with pol_col2:
        input_law = st.radio("MH Law Exists?", ["Yes", "No"], horizontal=True)

with col_gauges:
    # Prepare input for prediction
    input_data = pd.DataFrame({
        'gdp_per_capita_usd': [input_gdp], 'mh_spend_usd_per_capita': [input_mh_spend],
        'social_media_hours_daily': [input_social], 'internet_penetration_pct': [input_internet],
        'mh_policy_exists_bin': [1 if input_policy == "Yes" else 0], 'mh_law_exists_bin': [1 if input_law == "Yes" else 0]
    })
    
    # Predict with both models
    pred_lr = lr_model_full.predict(input_data)[0]
    pred_rf = rf_model_full.predict(input_data)[0]
    
    # Helper to build gauge
    def build_gauge(value, title):
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=value, title={'text': title},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#111111"},
                   'steps': [{'range': [0, 39], 'color': "#2ECC71"}, 
                             {'range': [40, 69], 'color': "#F1C40F"}, 
                             {'range': [70, 100], 'color': "#E74C3C"}]}
        ))
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
        return fig

    # Display Gauges side-by-side
    gauge_col1, gauge_col2 = st.columns(2)
    with gauge_col1:
        st.plotly_chart(build_gauge(pred_lr, "Linear Regression<br><span style='font-size:0.8em;color:gray'>Predicted Index</span>"), use_container_width=True)
    with gauge_col2:
        st.plotly_chart(build_gauge(pred_rf, "Random Forest<br><span style='font-size:0.8em;color:gray'>Predicted Index</span>"), use_container_width=True)