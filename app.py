import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ==========================================
# Page Configuration & Styling
# ==========================================
st.set_page_config(page_title="Global Mental Health Analytics", layout="wide")

# ==========================================
# 1. CACHING: Data Loading
# ==========================================
@st.cache_data
def load_data():
    df = pd.read_csv('Global_Mental_Health_Crisis_Index_2026.csv')
    income_order = ['Low', 'Lower-Middle', 'Upper-Middle', 'High']
    df['income_group'] = pd.Categorical(df['income_group'], categories=income_order, ordered=True)
    return df

df = load_data()

# ==========================================
# 2. CACHING: Machine Learning Model
# ==========================================
@st.cache_data
def run_clustering(_data, k):
    features = ['treatment_gap_pct', 'psychiatrists_per100k', 'social_media_hours_daily']
    X = _data[features].dropna()
    
    # K-Means requires at least as many samples as clusters
    if len(X) < k:
        return None, None
        
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    
    cluster_data = _data.copy()
    cluster_data['Cluster'] = kmeans.fit_predict(X_scaled)
    cluster_data['Cluster Label'] = 'Cluster ' + cluster_data['Cluster'].astype(str)
    
    summary = cluster_data.groupby('Cluster Label')[features].mean().round(2)
    return cluster_data, summary

# ==========================================
# Sidebar Navigation & Filters
# ==========================================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", [
    "1. Data Overview", 
    "2. Inequality Research", 
    "3. Social Media Impact", 
    "4. Correlation Heatmap", 
    "5. ML Clustering & Segmentation"
])

st.sidebar.markdown("---")
st.sidebar.header("Global Filters")

# Income Group Filter
selected_income = st.sidebar.multiselect(
    "Filter by Income Group:", 
    options=df['income_group'].cat.categories,
    default=list(df['income_group'].cat.categories)
)

# New: Country Selection / Comparison Filter
all_countries = sorted(df['country'].unique().tolist())
selected_countries = st.sidebar.multiselect(
    "Select Country (Leave empty for all):", 
    options=all_countries,
    default=[],
    help="Select one country to isolate it, or multiple to compare. Leave blank to view all countries."
)

# Apply both filters
filtered_df = df[df['income_group'].isin(selected_income)]
if selected_countries: # If the list is not empty, filter by selected countries
    filtered_df = filtered_df[filtered_df['country'].isin(selected_countries)]

# ==========================================
# Main App Body
# ==========================================
st.title("🧠 Global Mental Health Crisis Index (2026)")
st.markdown("An interactive dashboard analyzing the intersection of mental health, economic inequality, and digital environments.")

# Safety check if filters remove all data
if filtered_df.empty:
    st.error("No data available for the selected filters. Please adjust your Income Group or Country selections.")
    st.stop()

if page == "1. Data Overview":
    st.header("Dataset Overview")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Countries Viewed", len(filtered_df))
    col2.metric("Avg Treatment Gap", f"{filtered_df['treatment_gap_pct'].mean():.1f}%")
    col3.metric("Avg Social Media Hrs", f"{filtered_df['social_media_hours_daily'].mean():.1f}h")
    
    st.dataframe(filtered_df, use_container_width=True)

elif page == "2. Inequality Research":
    st.header("Inequality Research: Wealth vs. Access to Care")
    
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.box(filtered_df, x='income_group', y='treatment_gap_pct', color='income_group', 
                      title="Treatment Gap % by Income Group", points="all", hover_data=["country"])
        st.plotly_chart(fig1, use_container_width=True)
        
    with col2:
        fig2 = px.histogram(filtered_df, x='income_group', y='psychiatrists_per100k', color='income_group',
                            histfunc='avg', title="Average Psychiatrists per 100k")
        fig2.update_layout(yaxis_title="Avg Psychiatrists per 100k")
        st.plotly_chart(fig2, use_container_width=True)

elif page == "3. Social Media Impact":
    st.header("Social Media & Mental Health")
    
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.scatter(filtered_df, x='social_media_hours_daily', y='youth_mh_crisis_score',
                          color='social_media_mental_health_risk', size='depression_pct',
                          hover_name='country', title='Social Media vs. Youth Crisis Score',
                          color_discrete_sequence=px.colors.qualitative.Bold)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Only draw OLS trendline if we have more than 1 point
        trend = 'ols' if len(filtered_df) > 1 else None
        fig2 = px.scatter(filtered_df, x='social_media_hours_daily', y='depression_pct',
                          color='income_group', trendline=trend, hover_name='country',
                          title='Social Media vs. Depression Prevalence')
        st.plotly_chart(fig2, use_container_width=True)

elif page == "4. Correlation Heatmap":
    st.header("Variable Correlation Heatmap")
    
    if len(filtered_df) > 1:
        key_vars = ['mh_crisis_index', 'youth_mh_crisis_score', 'depression_pct', 'anxiety_pct', 
                    'social_media_hours_daily', 'gdp_per_capita_usd', 'treatment_gap_pct', 'psychiatrists_per100k']
        
        corr = filtered_df[key_vars].corr()
        fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r',
                        title="Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Heatmap requires at least 2 countries to calculate correlations.")

elif page == "5. ML Clustering & Segmentation":
    st.header("Machine Learning: Country Segmentation")
    st.markdown("Using a K-Means algorithm to group countries based on their mental health capacity and digital habits.")
    
    num_clusters = st.slider("Select Number of Clusters (K):", min_value=2, max_value=6, value=3, step=1)
    
    # Run cached clustering
    cluster_df, cluster_summary = run_clustering(filtered_df, num_clusters)
    
    if cluster_df is not None:
        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.scatter(cluster_df, x='treatment_gap_pct', y='psychiatrists_per100k', 
                              color='Cluster Label', hover_name='country', size='social_media_hours_daily',
                              title='Treatment Gap vs. Psychiatrists (Size = Social Media Hrs)')
            st.plotly_chart(fig1, use_container_width=True)
            
        with col2:
            fig2 = px.scatter(cluster_df, x='social_media_hours_daily', y='treatment_gap_pct', 
                              color='Cluster Label', hover_name='country',
                              title='Social Media vs. Treatment Gap')
            st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("---")
        
        # New: Detailed Metric Tables
        col_table1, col_table2 = st.columns([1, 2])
        
        with col_table1:
            st.subheader("Cluster Averages")
            st.dataframe(cluster_summary, use_container_width=True)
            
        with col_table2:
            st.subheader("Country Cluster Assignments")
            # Select the most important metrics to show in the detailed table
            important_cols = [
                'country', 'Cluster Label', 'income_group', 
                'treatment_gap_pct', 'psychiatrists_per100k', 
                'social_media_hours_daily', 'mh_crisis_index'
            ]
            st.dataframe(cluster_df[important_cols].sort_values('Cluster Label').reset_index(drop=True), 
                         use_container_width=True)
    else:
        st.warning(f"Not enough data to run {num_clusters} clusters. You currently have {len(filtered_df)} country/countries selected. Please select more countries or reduce K.")