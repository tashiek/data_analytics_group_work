import streamlit as st
import plotly.express as px
from utils import setup_page, load_data, global_filters, primary_colors

# Initialize page
setup_page(" Dashboard")
filtered_df = global_filters(load_data())

st.title("📊 Dashboard")
st.markdown("A consolidated view of global mental health KPIs, socioeconomic inequalities, and digital risk factors.")

# ==========================================
# ROW 1: High-Level KPIs
# ==========================================
col1, col2, col3, col4 = st.columns(4)
with col1: 
    st.markdown(f'<div class="kpi-card"><div class="kpi-title">Countries</div><div class="kpi-value">{len(filtered_df)}</div></div>', unsafe_allow_html=True)
with col2: 
    st.markdown(f'<div class="kpi-card"><div class="kpi-title">Avg Crisis Index</div><div class="kpi-value">{filtered_df["mh_crisis_index"].mean():.1f}</div></div>', unsafe_allow_html=True)
with col3: 
    st.markdown(f'<div class="kpi-card"><div class="kpi-title">Avg Treatment Gap</div><div class="kpi-value">{filtered_df["treatment_gap_pct"].mean():.1f}%</div></div>', unsafe_allow_html=True)
with col4: 
    st.markdown(f'<div class="kpi-card"><div class="kpi-title">Avg Social Media</div><div class="kpi-value">{filtered_df["social_media_hours_daily"].mean():.1f}h</div></div>', unsafe_allow_html=True)

st.markdown("---")

# ==========================================
# ROW 2: Inequality & Resource Distribution
# ==========================================
st.subheader("🌍 Resource Distribution & Inequality")
row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    # Box plots show a distribution, so they don't take a single text label.
    # However, points="all" draws a dot for every single country's actual value!
    fig1 = px.box(filtered_df, x='income_group', y='treatment_gap_pct', color='income_group', 
                  title="Treatment Gap % by Income Group", points="all", hover_data=["country"],
                  template="simple_white", color_discrete_sequence=primary_colors)
    st.plotly_chart(fig1, use_container_width=True)

with row2_col2:
    # We added text_auto='.1f' here to show the exact number to 1 decimal place
    fig2 = px.histogram(filtered_df, x='income_group', y='psychiatrists_per100k', color='income_group',
                        histfunc='avg', title="Average Psychiatrists per 100k", text_auto='.1f',
                        template="simple_white", color_discrete_sequence=primary_colors)
    
    fig2.update_layout(yaxis_title="Avg Psychiatrists per 100k", showlegend=False)
    
    # This pushes the text label to sit cleanly just above the top of the bar
    fig2.update_traces(textposition='outside') 
    
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ==========================================
# NEW ROW: Financial Investment & Critical Outcomes
# ==========================================
st.subheader("💰 Financial Investment & Critical Outcomes")
st.markdown("Explore how mental health funding (or the lack thereof) impacts extreme crisis outcomes like suicide rates.")

# 1. Dropdown for selecting the financial metric
fin_x_metric = st.selectbox(
    "Select Financial Metric to Compare on the Scatter Plot (X-Axis):", 
    options=["Mental Health Budget (% of Total Health)", "Mental Health Investment Gap"],
    index=0
)

# Map the dropdown choice to the actual dataset column
x_col = 'mh_budget_pct_health' if "Budget" in fin_x_metric else 'mh_investment_gap'

fin_col1, fin_col2 = st.columns(2)

with fin_col1:
    # 2. Regular Scatter Plot (Removed the 'size' parameter so all dots are equal)
    fig_fin1 = px.scatter(filtered_df, x=x_col, y='suicide_rate_per100k',
                          color='income_group', 
                          hover_name='country', 
                          title=f"Suicide Rate vs. {fin_x_metric}",
                          template="simple_white", color_discrete_sequence=primary_colors)
    
    # Optional: Make the dots slightly larger so they are easier to see on a regular scatter plot
    fig_fin1.update_traces(marker=dict(size=10))
    fig_fin1.update_layout(yaxis_title="Suicide Rate (per 100k)")
    st.plotly_chart(fig_fin1, use_container_width=True)
    
with fin_col2:
    # 3. Box Plot for Investment Gap
    fig_fin2 = px.box(filtered_df, x='income_group', y='mh_investment_gap', color='income_group',
                      title="Mental Health Investment Gap by Income Group",
                      points="all", hover_data=["country"],
                      template="simple_white", color_discrete_sequence=primary_colors)
    fig_fin2.update_layout(yaxis_title="Investment Gap Score", showlegend=False)
    st.plotly_chart(fig_fin2, use_container_width=True)

st.markdown("---")





# ==========================================
# ROW 4: Social Media Impact
# ==========================================
st.subheader("📱 Digital Impact on Mental Health")
row3_col1, row3_col2 = st.columns(2)

with row3_col1:
    fig3a = px.scatter(filtered_df, x='social_media_hours_daily', y='youth_mh_crisis_score',
                      color='social_media_mental_health_risk', size='depression_pct',
                      hover_name='country', title='Daily Social Media (Hours) vs. Youth Crisis Score',
                      template="simple_white", 
                      color_discrete_sequence=['#E74C3C', '#F1C40F', '#2ECC71', '#3498DB'])
    st.plotly_chart(fig3a, use_container_width=True)

with row3_col2:
    fig3b = px.scatter(filtered_df, x='social_media_hours_daily', y='depression_pct',
                      color='income_group', hover_name='country',
                      title='Daily Social Media (Hours) vs. Depression Prevalence',
                      template="simple_white", color_discrete_sequence=primary_colors)
    st.plotly_chart(fig3b, use_container_width=True)

st.markdown("---")

# ==========================================
# ROW 5: Statistical Correlation
# ==========================================
st.subheader("🔥 Statistical Correlation Heatmap")
if len(filtered_df) > 1:
    key_vars = ['mh_crisis_index', 'depression_pct', 'social_media_hours_daily', 'gdp_per_capita_usd', 'treatment_gap_pct', 'psychiatrists_per100k']
    # Rename columns to make the heatmap look cleaner
    clean_names = ['Crisis Index', 'Depression %', 'Social Media Hrs', 'GDP per Capita', 'Treatment Gap', 'Psychiatrists']
    corr_matrix = filtered_df[key_vars].corr()
    corr_matrix.columns = clean_names
    corr_matrix.index = clean_names
    
    # We let it take the full width of the container so the squares are nice and large
    fig4 = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale='RdYlBu_r',
                     template="simple_white")
    st.plotly_chart(fig4, use_container_width=True)
else:
    st.warning("Heatmap requires at least 2 countries.")

st.markdown("---")

# ==========================================
# ROW 6: Collapsible Raw Data Table
# ==========================================
with st.expander("📂 View Raw Dataset & Styled Metrics", expanded=False):
    st.markdown("This table highlights critical zones. Deeper blues indicate higher treatment gaps and crisis indexes.")
    styled_df = filtered_df[['country', 'region', 'income_group', 'treatment_gap_pct', 
                             'social_media_hours_daily', 'mh_crisis_index', 'mh_crisis_category']].style\
        .background_gradient(cmap='Blues', subset=['treatment_gap_pct', 'mh_crisis_index'])\
        .format({"treatment_gap_pct": "{:.1f}%", "social_media_hours_daily": "{:.1f}h"})
    st.dataframe(styled_df, use_container_width=True, height=400)