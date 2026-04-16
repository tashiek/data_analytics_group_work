# utils.py
import streamlit as st
import pandas as pd
import pycountry
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

def setup_page(title):
    """Sets up the page config and injects custom CSS for every page."""
    st.set_page_config(page_title=title, layout="wide", page_icon="🧠")
    st.markdown("""
    <style>
        .kpi-card { background: #ffffff; color: #111111; padding: 24px; border: 1px solid #eaeaea; border-radius: 4px; text-align: center; transition: border-color 0.2s ease; margin-bottom: 20px;}
        .kpi-card:hover { border-color: #3498DB; }
        .kpi-title { font-size: 0.85rem; font-weight: 500; color: #666666; margin-bottom: 8px; text-transform: uppercase;}
        .kpi-value { font-size: 2.2rem; font-weight: 300; margin: 0; color: #000000; }
    </style>
    """, unsafe_allow_html=True)

def get_flag(iso3):
    try:
        country = pycountry.countries.get(alpha_3=iso3)
        return chr(ord(country.alpha_2[0]) + 127397) + chr(ord(country.alpha_2[1]) + 127397) if country else "🏳️"
    except: return "🏳️"

@st.cache_data
def load_data():
    df = pd.read_csv('Global_Mental_Health_Crisis_Index_2026.csv')
    df['income_group'] = pd.Categorical(df['income_group'], categories=['Low', 'Lower-Middle', 'Upper-Middle', 'High'], ordered=True)
    df['country'] = df.apply(lambda row: f"{get_flag(row['iso3'])} {row['country']}", axis=1)
    df['mh_policy_exists_bin'] = df['mh_policy_exists'].map({'Yes': 1, 'No': 0})
    df['mh_law_exists_bin'] = df['mh_law_exists'].map({'Yes': 1, 'No': 0})
    df['mh_crisis_category'] = df['mh_crisis_index'].apply(lambda x: "Low Risk" if x < 40 else ("Medium Risk" if x < 70 else "High Risk"))
    return df

# --- REPLACE YOUR OLD run_clustering IN utils.py WITH THIS ---

@st.cache_data
def run_clustering(_data, k, features):
    X = _data[features].dropna()
    if len(X) < k: return None, None, None
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_data = _data.copy()
    cluster_data['Cluster'] = kmeans.fit_predict(X_scaled)
    cluster_data['Cluster Label'] = 'Cluster ' + cluster_data['Cluster'].astype(str)
    
    wcss = kmeans.inertia_  # Get the WCSS for this specific K
    summary = cluster_data.groupby('Cluster Label')[features].mean().round(2)
    return cluster_data, summary, wcss

@st.cache_data
def get_elbow_data(_data, features, max_k=10):
    """Calculates WCSS for K=1 up to max_k to draw the Elbow Curve."""
    X = _data[features].dropna()
    if X.empty: return pd.DataFrame()
    X_scaled = StandardScaler().fit_transform(X)
    wcss_list = []
    for i in range(1, min(max_k + 1, len(X))):
        kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        wcss_list.append({'K': i, 'WCSS': kmeans.inertia_})
    return pd.DataFrame(wcss_list)
@st.cache_resource
def train_models(_data):
    features = ['gdp_per_capita_usd', 'mh_spend_usd_per_capita', 'social_media_hours_daily', 'internet_penetration_pct', 'mh_policy_exists_bin', 'mh_law_exists_bin']
    X, y_reg, y_class = _data[features], _data['mh_crisis_index'], _data['mh_crisis_category']
    
    # Regression
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y_reg)
    
    # Classification Metrics
    X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)
    rf_clf_test = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    accuracy = accuracy_score(y_test, rf_clf_test.predict(X_test))
    cm = confusion_matrix(y_test, rf_clf_test.predict(X_test), labels=rf_clf_test.classes_)
    
    # Final Classification Model (Trained on all data for sandbox)
    rf_clf_full = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y_class)
    
    return rf_reg, rf_clf_full, accuracy, cm, rf_clf_test.classes_

def global_filters(df):
    st.sidebar.header("Global Filters")
    incomes = st.sidebar.multiselect("Income Group:", options=df['income_group'].cat.categories, default=list(df['income_group'].cat.categories))
    countries = st.sidebar.multiselect("Select Country:", options=sorted(df['country'].unique().tolist()), default=[])
    f_df = df[df['income_group'].isin(incomes)]
    if countries: f_df = f_df[f_df['country'].isin(countries)]
    if f_df.empty: st.error("No data available for the selected filters."); st.stop()
    return f_df

# Shared colors
primary_colors = ['#3498DB', '#E74C3C', '#2ECC71', '#F1C40F']
