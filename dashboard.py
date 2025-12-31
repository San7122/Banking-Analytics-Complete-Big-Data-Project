"""
Banking Analytics Dashboard
Interactive visualization of banking data analysis and ML results
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Page configuration
st.set_page_config(
    page_title="Banking Analytics Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: gray;
        margin-bottom: 1rem;
    }
    .filter-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border: 1px solid #e0e0e0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #155a8a;
    }
    div[data-testid="stHorizontalBlock"] > div {
        padding: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    data_path = os.path.join(os.path.dirname(__file__), "data", "bank.csv")
    df = pd.read_csv(data_path)
    return df

# Load the dataset
df = load_data()

# Initialize session state for filters
if 'apply_filters' not in st.session_state:
    st.session_state.apply_filters = False
if 'selected_jobs' not in st.session_state:
    st.session_state.selected_jobs = list(df['job'].unique())
if 'selected_education' not in st.session_state:
    st.session_state.selected_education = list(df['education'].unique())
if 'selected_marital' not in st.session_state:
    st.session_state.selected_marital = list(df['marital'].unique())
if 'age_range' not in st.session_state:
    st.session_state.age_range = (int(df['age'].min()), int(df['age'].max()))
if 'balance_range' not in st.session_state:
    st.session_state.balance_range = (int(df['balance'].min()), int(df['balance'].max()))
if 'subscription_filter' not in st.session_state:
    st.session_state.subscription_filter = "All"

# Main header
st.markdown('<p class="main-header">🏦 Banking Analytics Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Interactive analysis of bank marketing campaign data</p>', unsafe_allow_html=True)

# ============================================================
# TOP FILTER BAR
# ============================================================
st.markdown("### 🎛️ Quick Filters")

# Filter row 1: Quick action buttons
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if st.button("📊 Show All Data", use_container_width=True):
        st.session_state.subscription_filter = "All"
        st.session_state.selected_jobs = list(df['job'].unique())
        st.session_state.selected_education = list(df['education'].unique())
        st.session_state.selected_marital = list(df['marital'].unique())
        st.session_state.age_range = (int(df['age'].min()), int(df['age'].max()))
        st.rerun()

with col2:
    if st.button("✅ Subscribed Only", use_container_width=True):
        st.session_state.subscription_filter = "Yes"
        st.rerun()

with col3:
    if st.button("❌ Not Subscribed", use_container_width=True):
        st.session_state.subscription_filter = "No"
        st.rerun()

with col4:
    if st.button("💰 High Balance (>5000)", use_container_width=True):
        st.session_state.balance_range = (5000, int(df['balance'].max()))
        st.rerun()

with col5:
    if st.button("👴 Senior Clients (>50)", use_container_width=True):
        st.session_state.age_range = (50, int(df['age'].max()))
        st.rerun()

st.markdown("---")

# Filter row 2: Dropdowns and sliders in expander
with st.expander("🔧 Advanced Filters (Click to expand)", expanded=False):

    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:
        st.markdown("**👔 Job Type**")
        job_options = ['All'] + list(df['job'].unique())
        selected_job = st.selectbox(
            "Select Job",
            options=job_options,
            label_visibility="collapsed"
        )
        if selected_job != 'All':
            st.session_state.selected_jobs = [selected_job]
        else:
            st.session_state.selected_jobs = list(df['job'].unique())

    with filter_col2:
        st.markdown("**🎓 Education**")
        edu_options = ['All'] + list(df['education'].unique())
        selected_edu = st.selectbox(
            "Select Education",
            options=edu_options,
            label_visibility="collapsed"
        )
        if selected_edu != 'All':
            st.session_state.selected_education = [selected_edu]
        else:
            st.session_state.selected_education = list(df['education'].unique())

    with filter_col3:
        st.markdown("**💍 Marital Status**")
        marital_options = ['All'] + list(df['marital'].unique())
        selected_marital = st.selectbox(
            "Select Marital Status",
            options=marital_options,
            label_visibility="collapsed"
        )
        if selected_marital != 'All':
            st.session_state.selected_marital = [selected_marital]
        else:
            st.session_state.selected_marital = list(df['marital'].unique())

    st.markdown("---")

    slider_col1, slider_col2 = st.columns(2)

    with slider_col1:
        st.markdown("**📅 Age Range**")
        age_range = st.slider(
            "Age",
            min_value=int(df['age'].min()),
            max_value=int(df['age'].max()),
            value=st.session_state.age_range,
            label_visibility="collapsed"
        )
        st.session_state.age_range = age_range

    with slider_col2:
        st.markdown("**💵 Balance Range**")
        balance_range = st.slider(
            "Balance",
            min_value=int(df['balance'].min()),
            max_value=min(int(df['balance'].max()), 20000),
            value=(max(int(df['balance'].min()), st.session_state.balance_range[0]),
                   min(20000, st.session_state.balance_range[1])),
            label_visibility="collapsed"
        )
        st.session_state.balance_range = balance_range

# Apply filters
filtered_df = df.copy()

# Subscription filter
if st.session_state.subscription_filter == "Yes":
    filtered_df = filtered_df[filtered_df['y'] == 'yes']
elif st.session_state.subscription_filter == "No":
    filtered_df = filtered_df[filtered_df['y'] == 'no']

# Other filters
filtered_df = filtered_df[
    (filtered_df['job'].isin(st.session_state.selected_jobs)) &
    (filtered_df['education'].isin(st.session_state.selected_education)) &
    (filtered_df['marital'].isin(st.session_state.selected_marital)) &
    (filtered_df['age'].between(st.session_state.age_range[0], st.session_state.age_range[1])) &
    (filtered_df['balance'].between(st.session_state.balance_range[0], st.session_state.balance_range[1]))
]

# Show active filter status
st.markdown(f"**Showing: {len(filtered_df):,} of {len(df):,} records** | Subscription: {st.session_state.subscription_filter}")

st.markdown("---")

# ============================================================
# KEY METRICS
# ============================================================
st.markdown("### 📈 Key Metrics")

metric_col1, metric_col2, metric_col3, metric_col4, metric_col5, metric_col6 = st.columns(6)

with metric_col1:
    st.metric("👥 Total Clients", f"{len(filtered_df):,}")

with metric_col2:
    if len(filtered_df) > 0:
        sub_rate = (filtered_df['y'] == 'yes').mean() * 100
    else:
        sub_rate = 0
    st.metric("✅ Subscription Rate", f"{sub_rate:.1f}%")

with metric_col3:
    avg_balance = filtered_df['balance'].mean() if len(filtered_df) > 0 else 0
    st.metric("💰 Avg Balance", f"${avg_balance:,.0f}")

with metric_col4:
    avg_duration = filtered_df['duration'].mean() if len(filtered_df) > 0 else 0
    st.metric("📞 Avg Call Duration", f"{avg_duration:.0f}s")

with metric_col5:
    avg_age = filtered_df['age'].mean() if len(filtered_df) > 0 else 0
    st.metric("📅 Avg Age", f"{avg_age:.1f}")

with metric_col6:
    total_campaigns = filtered_df['campaign'].sum() if len(filtered_df) > 0 else 0
    st.metric("📢 Total Campaigns", f"{total_campaigns:,}")

st.markdown("---")

# ============================================================
# MAIN CONTENT - TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "👥 Customer Analysis",
    "📈 Campaign Insights",
    "🤖 ML Results",
    "📋 Data Explorer"
])

# TAB 1: Overview
with tab1:
    st.header("📊 Data Overview")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        # Job distribution with click interaction
        st.subheader("Distribution by Job Type")
        job_counts = filtered_df['job'].value_counts().reset_index()
        job_counts.columns = ['Job', 'Count']
        fig_job = px.bar(
            job_counts, x='Job', y='Count',
            color='Count',
            color_continuous_scale='Blues'
        )
        fig_job.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_job, use_container_width=True)

    with chart_col2:
        # Subscription donut chart
        st.subheader("Term Deposit Subscription")
        sub_counts = filtered_df['y'].value_counts().reset_index()
        sub_counts.columns = ['Subscribed', 'Count']
        sub_counts['Subscribed'] = sub_counts['Subscribed'].map({'yes': 'Yes ✅', 'no': 'No ❌'})
        fig_sub = px.pie(
            sub_counts, values='Count', names='Subscribed',
            color_discrete_sequence=['#2ecc71', '#e74c3c'],
            hole=0.5
        )
        fig_sub.update_layout(height=400)
        st.plotly_chart(fig_sub, use_container_width=True)

    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        # Education pie
        st.subheader("Education Level")
        edu_counts = filtered_df['education'].value_counts().reset_index()
        edu_counts.columns = ['Education', 'Count']
        fig_edu = px.pie(
            edu_counts, values='Count', names='Education',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_edu.update_layout(height=400)
        st.plotly_chart(fig_edu, use_container_width=True)

    with chart_col4:
        # Marital status
        st.subheader("Marital Status")
        marital_counts = filtered_df['marital'].value_counts().reset_index()
        marital_counts.columns = ['Status', 'Count']
        fig_marital = px.bar(
            marital_counts, x='Status', y='Count',
            color='Status',
            color_discrete_sequence=['#3498db', '#e74c3c', '#2ecc71']
        )
        fig_marital.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_marital, use_container_width=True)

# TAB 2: Customer Analysis
with tab2:
    st.header("👥 Customer Analysis")

    # Interactive job selector
    st.subheader("🔍 Analyze by Job Type")
    selected_job_analysis = st.selectbox(
        "Choose a job type to analyze:",
        options=['All Jobs'] + list(df['job'].unique()),
        key="job_analysis"
    )

    if selected_job_analysis != 'All Jobs':
        analysis_df = filtered_df[filtered_df['job'] == selected_job_analysis]
        st.info(f"Analyzing **{selected_job_analysis}** - {len(analysis_df)} clients")
    else:
        analysis_df = filtered_df

    ana_col1, ana_col2 = st.columns(2)

    with ana_col1:
        # Age distribution
        fig_age = px.histogram(
            analysis_df, x='age', nbins=25,
            title="Age Distribution",
            color_discrete_sequence=['#3498db']
        )
        fig_age.update_layout(height=350)
        st.plotly_chart(fig_age, use_container_width=True)

    with ana_col2:
        # Balance distribution
        fig_balance = px.histogram(
            analysis_df[analysis_df['balance'].between(-2000, 10000)],
            x='balance', nbins=40,
            title="Balance Distribution",
            color_discrete_sequence=['#2ecc71']
        )
        fig_balance.update_layout(height=350)
        st.plotly_chart(fig_balance, use_container_width=True)

    # Balance by job - horizontal bar
    st.subheader("💰 Average Balance by Job Type")
    job_balance = filtered_df.groupby('job')['balance'].mean().sort_values(ascending=True).reset_index()
    job_balance.columns = ['Job', 'Average Balance']
    fig_job_bal = px.bar(
        job_balance, x='Average Balance', y='Job',
        orientation='h',
        color='Average Balance',
        color_continuous_scale='Viridis'
    )
    fig_job_bal.update_layout(height=450)
    st.plotly_chart(fig_job_bal, use_container_width=True)

    # Scatter plot with subscription color
    st.subheader("🔗 Age vs Balance Relationship")
    sample_size = min(1500, len(filtered_df))
    sample_df = filtered_df.sample(sample_size) if len(filtered_df) > sample_size else filtered_df
    fig_scatter = px.scatter(
        sample_df, x='age', y='balance',
        color='y',
        color_discrete_map={'yes': '#2ecc71', 'no': '#e74c3c'},
        opacity=0.6,
        labels={'y': 'Subscribed'}
    )
    fig_scatter.update_layout(height=450)
    st.plotly_chart(fig_scatter, use_container_width=True)

# TAB 3: Campaign Insights
with tab3:
    st.header("📈 Campaign Insights")

    # Month selector
    st.subheader("📅 Select Month to Analyze")
    month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    available_months = [m for m in month_order if m in filtered_df['month'].unique()]

    month_cols = st.columns(len(available_months))
    selected_month = None

    for i, month in enumerate(available_months):
        with month_cols[i]:
            if st.button(month.upper(), key=f"month_{month}", use_container_width=True):
                selected_month = month

    if selected_month:
        month_df = filtered_df[filtered_df['month'] == selected_month]
        st.success(f"Showing data for **{selected_month.upper()}**: {len(month_df)} contacts")
    else:
        month_df = filtered_df
        st.info("Click a month button above to filter, or view all months below")

    camp_col1, camp_col2 = st.columns(2)

    with camp_col1:
        # Monthly contacts
        month_counts = filtered_df.groupby('month').size().reindex(month_order).dropna().reset_index()
        month_counts.columns = ['Month', 'Contacts']
        fig_month = px.bar(
            month_counts, x='Month', y='Contacts',
            title="Contacts by Month",
            color='Contacts',
            color_continuous_scale='Oranges'
        )
        fig_month.update_layout(height=400)
        st.plotly_chart(fig_month, use_container_width=True)

    with camp_col2:
        # Contact method success
        contact_success = filtered_df.groupby('contact').agg(
            total=('y', 'count'),
            subscribed=('y', lambda x: (x == 'yes').sum())
        ).reset_index()
        contact_success['Success Rate %'] = (contact_success['subscribed'] / contact_success['total'] * 100).round(2)
        fig_contact = px.bar(
            contact_success, x='contact', y='Success Rate %',
            title="Success Rate by Contact Method",
            color='Success Rate %',
            color_continuous_scale='Greens',
            text='Success Rate %'
        )
        fig_contact.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_contact.update_layout(height=400)
        st.plotly_chart(fig_contact, use_container_width=True)

    # Duration comparison
    st.subheader("⏱️ Call Duration Impact")

    dur_col1, dur_col2 = st.columns([1, 2])

    with dur_col1:
        duration_sub = filtered_df.groupby('y')['duration'].mean().reset_index()
        duration_sub.columns = ['Subscribed', 'Avg Duration (sec)']
        duration_sub['Subscribed'] = duration_sub['Subscribed'].map({'yes': 'Yes ✅', 'no': 'No ❌'})
        fig_dur = px.bar(
            duration_sub, x='Subscribed', y='Avg Duration (sec)',
            color='Subscribed',
            color_discrete_sequence=['#2ecc71', '#e74c3c'],
            text='Avg Duration (sec)'
        )
        fig_dur.update_traces(texttemplate='%{text:.0f}s', textposition='outside')
        fig_dur.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_dur, use_container_width=True)

    with dur_col2:
        st.markdown("""
        ### 💡 Key Insight

        **Longer calls = Higher conversion!**

        - Subscribers have **~2.5x longer** call duration
        - Average successful call: **~550 seconds** (9 minutes)
        - Average unsuccessful call: **~225 seconds** (4 minutes)

        **Recommendation:** Train agents to engage customers longer!
        """)

    # Previous outcome
    st.subheader("📊 Previous Campaign Outcome Impact")
    poutcome_success = filtered_df.groupby('poutcome').agg(
        total=('y', 'count'),
        subscribed=('y', lambda x: (x == 'yes').sum())
    ).reset_index()
    poutcome_success['Success Rate %'] = (poutcome_success['subscribed'] / poutcome_success['total'] * 100).round(2)
    fig_pout = px.bar(
        poutcome_success, x='poutcome', y='Success Rate %',
        color='Success Rate %',
        color_continuous_scale='RdYlGn',
        text='Success Rate %'
    )
    fig_pout.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig_pout.update_layout(height=400)
    st.plotly_chart(fig_pout, use_container_width=True)

# TAB 4: ML Results
with tab4:
    st.header("🤖 Machine Learning Results")

    st.info("Results from ML model training using Pandas & Scikit-learn")

    # Model selector
    st.subheader("🎯 Select Model to View Details")

    model_cols = st.columns(4)
    models = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting']
    model_scores = [0.8886, 0.6781, 0.8831, 0.8883]

    selected_model = None
    for i, (model, score) in enumerate(zip(models, model_scores)):
        with model_cols[i]:
            if st.button(f"{model}\nAUC: {score:.2f}", key=f"model_{i}", use_container_width=True):
                selected_model = model

    # Model results table
    st.subheader("📊 Model Comparison")

    model_results = pd.DataFrame({
        'Model': models,
        'AUC-ROC': [0.8886, 0.6781, 0.8831, 0.8883],
        'Accuracy': [0.8928, 0.8597, 0.8873, 0.8928],
        'Precision': [0.5636, 0.4000, 0.5192, 0.5538],
        'Recall': [0.2981, 0.4423, 0.2596, 0.3462],
        'F1-Score': [0.3899, 0.4201, 0.3462, 0.4260]
    })

    # Highlight best values
    st.dataframe(
        model_results.style.highlight_max(
            axis=0,
            subset=['AUC-ROC', 'Accuracy', 'Precision', 'Recall', 'F1-Score'],
            color='lightgreen'
        ),
        use_container_width=True,
        hide_index=True
    )

    ml_col1, ml_col2 = st.columns(2)

    with ml_col1:
        # AUC comparison
        fig_auc = px.bar(
            model_results, x='Model', y='AUC-ROC',
            title="AUC-ROC Score Comparison",
            color='AUC-ROC',
            color_continuous_scale='Viridis',
            text='AUC-ROC'
        )
        fig_auc.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig_auc.add_hline(y=0.5, line_dash="dash", line_color="red",
                         annotation_text="Random Baseline (0.5)")
        fig_auc.update_layout(height=400)
        st.plotly_chart(fig_auc, use_container_width=True)

    with ml_col2:
        # Radar chart for selected model or best model
        fig_radar = go.Figure()

        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

        # Add trace for best model (Logistic Regression)
        fig_radar.add_trace(go.Scatterpolar(
            r=[0.8928, 0.5636, 0.2981, 0.3899],
            theta=metrics,
            fill='toself',
            name='Logistic Regression'
        ))

        fig_radar.add_trace(go.Scatterpolar(
            r=[0.8928, 0.5538, 0.3462, 0.4260],
            theta=metrics,
            fill='toself',
            name='Gradient Boosting'
        ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Model Performance Radar",
            height=400
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Feature importance
    st.subheader("🎯 Feature Importance (Random Forest)")

    feature_importance = pd.DataFrame({
        'Feature': ['duration', 'poutcome_success', 'age', 'balance', 'month_oct',
                   'campaign', 'poutcome_unknown', 'contact_unknown', 'housing_yes', 'month_mar'],
        'Importance': [0.330, 0.104, 0.088, 0.079, 0.036, 0.032, 0.028, 0.021, 0.021, 0.017]
    })

    fig_feat = px.bar(
        feature_importance, x='Importance', y='Feature',
        orientation='h',
        color='Importance',
        color_continuous_scale='Blues',
        text='Importance'
    )
    fig_feat.update_traces(texttemplate='%{text:.1%}', textposition='outside')
    fig_feat.update_layout(height=450, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_feat, use_container_width=True)

    # Key takeaways
    st.subheader("💡 Key Takeaways")

    takeaway_cols = st.columns(3)

    with takeaway_cols[0]:
        st.success("""
        **🏆 Best Model**

        Logistic Regression
        - AUC-ROC: **0.89**
        - Simple & interpretable
        - Fast predictions
        """)

    with takeaway_cols[1]:
        st.info("""
        **🎯 Top 3 Predictors**

        1. **Duration** (33%)
        2. **Previous Outcome** (10%)
        3. **Age** (9%)
        """)

    with takeaway_cols[2]:
        st.warning("""
        **📌 Business Actions**

        - Focus on longer calls
        - Target previous successes
        - Best months: Oct, Mar, Dec
        """)

# TAB 5: Data Explorer
with tab5:
    st.header("📋 Data Explorer")

    # Search functionality
    st.subheader("🔍 Search Data")

    search_col1, search_col2, search_col3 = st.columns([2, 1, 1])

    with search_col1:
        search_term = st.text_input("Search by job type:", placeholder="e.g., management, technician...")

    with search_col2:
        min_balance = st.number_input("Min Balance:", value=0)

    with search_col3:
        max_age = st.number_input("Max Age:", value=100)

    # Apply search
    search_df = filtered_df.copy()
    if search_term:
        search_df = search_df[search_df['job'].str.contains(search_term, case=False)]
    search_df = search_df[search_df['balance'] >= min_balance]
    search_df = search_df[search_df['age'] <= max_age]

    st.markdown(f"**Found: {len(search_df)} records**")

    # Display data
    st.subheader("📄 Dataset Preview")
    st.dataframe(search_df.head(100), use_container_width=True, hide_index=True)

    # Statistics
    st.subheader("📊 Statistical Summary")
    st.dataframe(search_df.describe().round(2), use_container_width=True)

    # Column info
    with st.expander("ℹ️ Column Information"):
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.astype(str).values,
            'Non-Null': df.notnull().sum().values,
            'Unique': df.nunique().values
        })
        st.dataframe(col_info, use_container_width=True, hide_index=True)

    # Download
    st.subheader("📥 Download Data")

    download_col1, download_col2 = st.columns(2)

    with download_col1:
        csv = search_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data (CSV)",
            data=csv,
            file_name="banking_filtered_data.csv",
            mime="text/csv",
            use_container_width=True
        )

    with download_col2:
        full_csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Dataset (CSV)",
            data=full_csv,
            file_name="banking_full_data.csv",
            mime="text/csv",
            use_container_width=True
        )

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>🏦 Banking Analytics Dashboard | Built with Streamlit & Plotly</p>",
    unsafe_allow_html=True
)
