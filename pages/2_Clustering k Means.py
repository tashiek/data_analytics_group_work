import streamlit as st
import plotly.express as px
from utils import setup_page, load_data, global_filters, run_clustering, get_elbow_data, primary_colors

# Initialize page
setup_page("ML Clustering")
filtered_df = global_filters(load_data())

st.title("🤖 Unsupervised ML: Country Segmentation")
st.markdown("We utilize a **K-Means Clustering** algorithm to segment countries based on their resource capacity and digital risk factors. Use the Elbow Curve below to find the optimal number of clusters.")

# The features we use to actually build the clusters
cluster_features = ['treatment_gap_pct', 'psychiatrists_per100k', 'social_media_hours_daily']

# ==========================================
# ROW 1: Controls & The Elbow Curve
# ==========================================
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### ⚙️ Algorithm Settings")
    num_clusters = st.slider("Select Number of Clusters (K):", min_value=2, max_value=8, value=3, step=1)
    
    # Run the clustering algorithm
    cluster_df, cluster_summary, current_wcss = run_clustering(filtered_df, num_clusters, cluster_features)
    
    if current_wcss is not None:
        st.markdown(f"""
        <div class="kpi-card" style="margin-top: 20px;">
            <div class="kpi-title">Current WCSS </div>
            <div class="kpi-value">{current_wcss:.1f}</div>
        </div>
        """, unsafe_allow_html=True)
        st.info("💡 **Tip:** Look for the 'elbow' in the chart to the right. The ideal K is where the WCSS stops dropping drastically.")

with col2:
    # Generate and plot the Elbow Curve
    elbow_df = get_elbow_data(filtered_df, cluster_features)
    if not elbow_df.empty:
        fig_elbow = px.line(elbow_df, x='K', y='WCSS', markers=True, 
                            title="Elbow Method for Optimal K",
                            template="simple_white", color_discrete_sequence=['#E74C3C'])
        
        # Add a vertical line to show where the user's slider currently is!
        fig_elbow.add_vline(x=num_clusters, line_dash="dash", line_color="#3498DB", 
                            annotation_text=f"Selected K={num_clusters}", annotation_position="top right")
        st.plotly_chart(fig_elbow, use_container_width=True)

st.markdown("---")

# ==========================================
# ROW 2: Dynamic Scatter Plots
# ==========================================
if cluster_df is not None:
    st.subheader("🔭 Interactive Cluster Visualization")
    st.markdown("Change the X and Y axes below to explore how the K-Means algorithm separated the countries across different variables.")
    
    # Variables the user can choose from for the charts
    plot_options = {
        'Treatment Gap (%)': 'treatment_gap_pct',
        'Psychiatrists per 100k': 'psychiatrists_per100k',
        'Daily Social Media (Hours)': 'social_media_hours_daily',
        'Crisis Index': 'mh_crisis_index',
        'GDP per Capita (USD)': 'gdp_per_capita_usd'
    }
    
    plot_col1, plot_col2 = st.columns(2)
    
    with plot_col1:
        # Dynamic selectors for Plot 1
        x_axis_1 = st.selectbox("Plot 1: X-Axis", options=list(plot_options.keys()), index=0)
        y_axis_1 = st.selectbox("Plot 1: Y-Axis", options=list(plot_options.keys()), index=1)
        
        fig1 = px.scatter(cluster_df, x=plot_options[x_axis_1], y=plot_options[y_axis_1], 
                          color='Cluster Label', hover_name='country', size='mh_crisis_index',
                          title=f'{y_axis_1} vs. {x_axis_1}',
                          template="simple_white", color_discrete_sequence=primary_colors)
        st.plotly_chart(fig1, use_container_width=True)
        
    with plot_col2:
        # Dynamic selectors for Plot 2
        x_axis_2 = st.selectbox("Plot 2: X-Axis", options=list(plot_options.keys()), index=2)
        y_axis_2 = st.selectbox("Plot 2: Y-Axis", options=list(plot_options.keys()), index=0)
        
        fig2 = px.scatter(cluster_df, x=plot_options[x_axis_2], y=plot_options[y_axis_2], 
                          color='Cluster Label', hover_name='country',
                          title=f'{y_axis_2} vs. {x_axis_2}',
                          template="simple_white", color_discrete_sequence=primary_colors)
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("---")
    
    # ==========================================
    # ROW 3: Data Tables
    # ==========================================
    col_table1, col_table2 = st.columns([1, 2])
    with col_table1:
        st.markdown("### Cluster Averages")
        st.dataframe(cluster_summary.style.background_gradient(cmap='Blues'), use_container_width=True)
    with col_table2:
        st.markdown("### Country Cluster Assignments")
        important_cols = ['country', 'Cluster Label', 'income_group'] + cluster_features
        display_df = cluster_df[important_cols].sort_values('Cluster Label').reset_index(drop=True)
        st.dataframe(display_df, use_container_width=True)
else:
    st.warning("Not enough data to run clustering. Please select more countries or reduce K.")