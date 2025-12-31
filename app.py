"""
🏦 Banking Analytics - Complete Project Dashboard
Comprehensive dashboard showing all 6 parts of the Big Data project
Ready for Hugging Face deployment
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Banking Analytics - Complete Project",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .section-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .code-box {
        background-color: #1e1e1e;
        color: #d4d4d4;
        padding: 1rem;
        border-radius: 5px;
        font-family: monospace;
        overflow-x: auto;
    }
    .insight-box {
        background-color: #e8f4f8;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    # Try multiple paths for flexibility (local and Hugging Face)
    try:
        df = pd.read_csv("data/bank.csv")
    except:
        try:
            df = pd.read_csv("bank.csv")
        except:
            # Create sample data if file not found
            st.warning("Using sample data. Upload bank.csv for full functionality.")
            np.random.seed(42)
            n = 1000
            df = pd.DataFrame({
                'age': np.random.randint(20, 70, n),
                'job': np.random.choice(['management', 'technician', 'blue-collar', 'admin.', 'services'], n),
                'marital': np.random.choice(['married', 'single', 'divorced'], n),
                'education': np.random.choice(['primary', 'secondary', 'tertiary', 'unknown'], n),
                'default': np.random.choice(['yes', 'no'], n, p=[0.02, 0.98]),
                'balance': np.random.randint(-500, 10000, n),
                'housing': np.random.choice(['yes', 'no'], n),
                'loan': np.random.choice(['yes', 'no'], n),
                'contact': np.random.choice(['cellular', 'telephone', 'unknown'], n),
                'day': np.random.randint(1, 31, n),
                'month': np.random.choice(['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'], n),
                'duration': np.random.randint(10, 1000, n),
                'campaign': np.random.randint(1, 10, n),
                'pdays': np.random.randint(-1, 400, n),
                'previous': np.random.randint(0, 5, n),
                'poutcome': np.random.choice(['success', 'failure', 'unknown', 'other'], n),
                'y': np.random.choice(['yes', 'no'], n, p=[0.12, 0.88])
            })
    return df

df = load_data()

# Sidebar Navigation
st.sidebar.markdown("## 🏦 Banking Analytics")
st.sidebar.markdown("### Complete Big Data Project")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "📚 Navigate to:",
    [
        "🏠 Home & Overview",
        "📊 Part 1: Data Processing",
        "🔄 Part 2: MapReduce",
        "📝 Part 3: Hive Analytics",
        "🤖 Part 4: Machine Learning",
        "⚡ Part 5: Spark Streaming",
        "🚀 Part 6: Data Parallelism",
        "📈 Interactive Dashboard",
        "📖 Learning Guide"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Dataset Info")
st.sidebar.metric("Total Records", f"{len(df):,}")
st.sidebar.metric("Features", f"{len(df.columns)}")
st.sidebar.metric("Subscription Rate", f"{(df['y']=='yes').mean()*100:.1f}%")

# ============================================================
# HOME PAGE
# ============================================================
if page == "🏠 Home & Overview":
    st.markdown('<h1 class="main-title">🏦 Banking Analytics with Distributed Computing</h1>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2rem; color: gray;'>Complete Big Data Project - All 6 Parts</p>", unsafe_allow_html=True)

    st.markdown("---")

    # Project Overview
    st.markdown("## 🎯 What is This Project?")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        This project analyzes a **bank marketing campaign** where a bank contacted customers
        to convince them to subscribe to a **term deposit** (savings product).

        ### 🎯 Project Goals:
        1. **Understand** customer behavior and patterns
        2. **Analyze** which factors lead to successful subscriptions
        3. **Build ML models** to predict future subscriptions
        4. **Demonstrate** Big Data technologies (Spark, Hadoop, Hive)

        ### 💼 Real-World Application:
        Banks can use this analysis to:
        - Target the right customers
        - Save marketing costs
        - Improve campaign success rates
        """)

    with col2:
        # Quick stats
        st.markdown("### 📊 Quick Stats")

        sub_yes = (df['y'] == 'yes').sum()
        sub_no = (df['y'] == 'no').sum()

        fig = go.Figure(data=[go.Pie(
            labels=['Subscribed', 'Not Subscribed'],
            values=[sub_yes, sub_no],
            hole=.4,
            marker_colors=['#2ecc71', '#e74c3c']
        )])
        fig.update_layout(height=250, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Project Structure
    st.markdown("## 📁 Project Structure")

    parts = [
        ("📊 Part 1", "Data Processing", "EDA with Spark/Pandas", "#3498db"),
        ("🔄 Part 2", "MapReduce", "Hadoop distributed processing", "#e74c3c"),
        ("📝 Part 3", "Hive Analytics", "SQL-based analytics", "#2ecc71"),
        ("🤖 Part 4", "Machine Learning", "Predictive modeling", "#9b59b6"),
        ("⚡ Part 5", "Spark Streaming", "Real-time processing", "#f39c12"),
        ("🚀 Part 6", "Data Parallelism", "Parallel computing", "#1abc9c"),
    ]

    cols = st.columns(3)
    for i, (icon, title, desc, color) in enumerate(parts):
        with cols[i % 3]:
            st.markdown(f"""
            <div style='background-color: {color}20; border-left: 4px solid {color};
                        padding: 1rem; border-radius: 5px; margin: 0.5rem 0; height: 120px;'>
                <h4 style='color: {color}; margin: 0;'>{icon} {title}</h4>
                <p style='margin: 0.5rem 0 0 0; color: #333;'>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Dataset Overview
    st.markdown("## 📋 Dataset Columns Explained")

    columns_info = {
        "age": "Customer's age in years",
        "job": "Type of job (management, technician, etc.)",
        "marital": "Marital status (married, single, divorced)",
        "education": "Education level (primary, secondary, tertiary)",
        "default": "Has credit in default?",
        "balance": "Average yearly balance in euros",
        "housing": "Has housing loan?",
        "loan": "Has personal loan?",
        "contact": "Contact communication type",
        "day": "Last contact day of the month",
        "month": "Last contact month of year",
        "duration": "Last contact duration in seconds",
        "campaign": "Number of contacts during this campaign",
        "pdays": "Days since last contact from previous campaign",
        "previous": "Number of contacts before this campaign",
        "poutcome": "Outcome of previous marketing campaign",
        "y": "🎯 TARGET: Has the client subscribed?"
    }

    col1, col2 = st.columns(2)
    items = list(columns_info.items())

    for i, (col_name, desc) in enumerate(items[:9]):
        with col1:
            st.markdown(f"**`{col_name}`**: {desc}")

    for i, (col_name, desc) in enumerate(items[9:]):
        with col2:
            st.markdown(f"**`{col_name}`**: {desc}")

# ============================================================
# PART 1: DATA PROCESSING
# ============================================================
elif page == "📊 Part 1: Data Processing":
    st.markdown('<div class="section-header"><h2>📊 Part 1: Data Processing & EDA</h2></div>', unsafe_allow_html=True)

    st.markdown("""
    ### 🎯 Objective
    Perform **Exploratory Data Analysis (EDA)** to understand the dataset, find patterns,
    and prepare data for machine learning.
    """)

    tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Overview", "📊 Visualizations", "🔍 Analysis", "💻 Code"])

    with tab1:
        st.markdown("### 📋 Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True, hide_index=True)

        st.markdown("### 📊 Statistical Summary")
        st.dataframe(df.describe().round(2), use_container_width=True)

        st.markdown("### 🔢 Data Types")
        dtype_df = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.astype(str),
            'Non-Null': df.notnull().sum(),
            'Unique Values': df.nunique()
        })
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)

    with tab2:
        st.markdown("### 📊 Interactive Visualizations")

        viz_col1, viz_col2 = st.columns(2)

        with viz_col1:
            # Job distribution
            job_counts = df['job'].value_counts().reset_index()
            job_counts.columns = ['Job', 'Count']
            fig = px.bar(job_counts, x='Job', y='Count', title="Distribution by Job Type",
                        color='Count', color_continuous_scale='Blues')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        with viz_col2:
            # Age distribution
            fig = px.histogram(df, x='age', nbins=30, title="Age Distribution",
                             color_discrete_sequence=['#3498db'])
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        viz_col3, viz_col4 = st.columns(2)

        with viz_col3:
            # Subscription by education
            edu_sub = df.groupby('education')['y'].apply(lambda x: (x=='yes').mean()*100).reset_index()
            edu_sub.columns = ['Education', 'Subscription Rate %']
            fig = px.bar(edu_sub, x='Education', y='Subscription Rate %',
                        title="Subscription Rate by Education",
                        color='Subscription Rate %', color_continuous_scale='Greens')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        with viz_col4:
            # Balance by job
            job_balance = df.groupby('job')['balance'].mean().sort_values(ascending=True).reset_index()
            fig = px.bar(job_balance, x='balance', y='job', orientation='h',
                        title="Average Balance by Job", color='balance',
                        color_continuous_scale='Viridis')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("### 🔍 Key Analysis Findings")

        # Calculate insights
        avg_duration_yes = df[df['y']=='yes']['duration'].mean()
        avg_duration_no = df[df['y']=='no']['duration'].mean()

        insight_col1, insight_col2, insight_col3 = st.columns(3)

        with insight_col1:
            st.metric("Avg Duration (Subscribed)", f"{avg_duration_yes:.0f}s")
            st.metric("Avg Duration (Not Subscribed)", f"{avg_duration_no:.0f}s")

        with insight_col2:
            cellular_rate = (df[df['contact']=='cellular']['y']=='yes').mean()*100
            st.metric("Cellular Success Rate", f"{cellular_rate:.1f}%")

            top_job = df.groupby('job')['balance'].mean().idxmax()
            st.metric("Highest Avg Balance Job", top_job)

        with insight_col3:
            best_month = df.groupby('month').apply(lambda x: (x['y']=='yes').mean()).idxmax()
            st.metric("Best Month", best_month.upper())

            correlation = df['age'].corr(df['balance'])
            st.metric("Age-Balance Correlation", f"{correlation:.3f}")

        st.markdown("""
        ### 💡 Key Insights

        1. **Duration Matters**: Subscribers have ~2.5x longer call duration
        2. **Cellular is Better**: Mobile contacts have higher success rate
        3. **Education Helps**: Tertiary education has highest subscription rate
        4. **Timing is Key**: October and March have best conversion rates
        """)

    with tab4:
        st.markdown("### 💻 Python Code for Data Processing")

        st.code('''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("data/bank.csv")

# Basic inspection
print(f"Dataset Shape: {df.shape}")
print(df.head(10))
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Filter clients with balance > 1000
df_filtered = df[df["balance"] > 1000]
print(f"Clients with balance > 1000: {len(df_filtered)}")

# Add quarter column
def get_quarter(month):
    quarters = {'jan':1, 'feb':1, 'mar':1, 'apr':2, 'may':2, 'jun':2,
                'jul':3, 'aug':3, 'sep':3, 'oct':4, 'nov':4, 'dec':4}
    return quarters.get(month, 1)

df['quarter'] = df['month'].apply(get_quarter)

# Group by job and calculate statistics
job_stats = df.groupby('job').agg({
    'balance': 'mean',
    'age': 'mean',
    'y': lambda x: (x == 'yes').sum()
}).round(2)
print(job_stats)

# Subscription rate by education
edu_rate = df.groupby('education').apply(
    lambda x: (x['y'] == 'yes').mean() * 100
).round(2)
print(edu_rate)

# Create visualization
plt.figure(figsize=(12, 6))
df['job'].value_counts().plot(kind='bar', color='steelblue')
plt.title('Distribution of Clients by Job Type')
plt.xlabel('Job Type')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('job_distribution.png')
plt.show()
        ''', language='python')

# ============================================================
# PART 2: MAPREDUCE
# ============================================================
elif page == "🔄 Part 2: MapReduce":
    st.markdown('<div class="section-header"><h2>🔄 Part 2: Hadoop MapReduce</h2></div>', unsafe_allow_html=True)

    st.markdown("""
    ### 🎯 What is MapReduce?

    MapReduce is a **programming model** for processing large datasets in parallel across a cluster of computers.
    It breaks down into two main phases:
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 1️⃣ MAP Phase
        - Reads input data
        - Extracts key-value pairs
        - Processes independently
        """)

    with col2:
        st.markdown("""
        ### 2️⃣ SHUFFLE Phase
        - Groups by key
        - Sorts data
        - Prepares for reduce
        """)

    with col3:
        st.markdown("""
        ### 3️⃣ REDUCE Phase
        - Aggregates values
        - Computes final result
        - Outputs summary
        """)

    st.markdown("---")

    # Visual explanation
    st.markdown("### 🔄 MapReduce Flow Diagram")

    st.markdown("""
    ```
    INPUT DATA                    MAP                      SHUFFLE                 REDUCE              OUTPUT
    ┌─────────────┐         ┌─────────────┐          ┌─────────────┐        ┌─────────────┐    ┌─────────────┐
    │ Row 1: mgmt │ ──────► │(mgmt, 1500) │          │ mgmt: [1500,│        │ mgmt: avg   │    │ mgmt: 1766  │
    │ balance=1500│         │             │ ────┐    │  2000, 1800]│ ──────►│   = 1766    │───►│ tech: 1331  │
    ├─────────────┤         ├─────────────┤     │    ├─────────────┤        ├─────────────┤    │ admin: 1226 │
    │ Row 2: tech │ ──────► │(tech, 1200) │ ────┼───►│ tech: [1200,│ ──────►│ tech: avg   │    └─────────────┘
    │ balance=1200│         │             │     │    │  1400, 1393]│        │   = 1331    │
    ├─────────────┤         ├─────────────┤     │    └─────────────┘        └─────────────┘
    │ Row 3: mgmt │ ──────► │(mgmt, 2000) │ ────┘
    │ balance=2000│         │             │
    └─────────────┘         └─────────────┘
    ```
    """)

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📝 Mapper Code", "📝 Reducer Code", "▶️ Live Demo"])

    with tab1:
        st.markdown("### 📝 Mapper: Average Balance by Job")
        st.code('''
#!/usr/bin/env python3
"""
MAPPER: Extract job and balance from each record
Input:  CSV row
Output: job<TAB>balance
"""
import sys

# Skip header
header = True

for line in sys.stdin:
    if header:
        header = False
        continue

    try:
        # Parse CSV line
        fields = line.strip().split(',')

        # Extract job (index 1) and balance (index 5)
        job = fields[1]
        balance = fields[5]

        # Emit key-value pair
        print(f"{job}\\t{balance}")

    except Exception as e:
        # Skip malformed lines
        continue
        ''', language='python')

        st.markdown("""
        **How it works:**
        1. Reads each line from input
        2. Splits by comma to get fields
        3. Extracts `job` (column 1) and `balance` (column 5)
        4. Outputs: `job<TAB>balance`
        """)

    with tab2:
        st.markdown("### 📝 Reducer: Calculate Average Balance")
        st.code('''
#!/usr/bin/env python3
"""
REDUCER: Calculate average balance per job
Input:  job<TAB>balance (sorted by job)
Output: job<TAB>average_balance
"""
import sys

current_job = None
total_balance = 0
count = 0

for line in sys.stdin:
    try:
        job, balance = line.strip().split('\\t')
        balance = float(balance)

        if current_job == job:
            # Same job, accumulate
            total_balance += balance
            count += 1
        else:
            # New job, output previous result
            if current_job is not None:
                avg = total_balance / count
                print(f"{current_job}\\t{avg:.2f}")

            # Reset for new job
            current_job = job
            total_balance = balance
            count = 1

    except ValueError:
        continue

# Output last job
if current_job is not None:
    avg = total_balance / count
    print(f"{current_job}\\t{avg:.2f}")
        ''', language='python')

        st.markdown("""
        **How it works:**
        1. Receives sorted key-value pairs
        2. Accumulates balance for same job
        3. When job changes, calculates and outputs average
        4. Resets counters for new job
        """)

    with tab3:
        st.markdown("### ▶️ Live MapReduce Simulation")

        if st.button("🚀 Run MapReduce Simulation", use_container_width=True):
            with st.spinner("Running MapReduce..."):
                import time

                # Simulate MAP phase
                st.markdown("#### 1️⃣ MAP Phase")
                progress = st.progress(0)

                map_output = []
                for i, row in df.head(100).iterrows():
                    map_output.append((row['job'], row['balance']))
                    progress.progress((i+1)/100)
                    time.sleep(0.01)

                st.success(f"✅ Mapped {len(map_output)} records")
                st.dataframe(pd.DataFrame(map_output[:10], columns=['Key (Job)', 'Value (Balance)']), hide_index=True)

                # Simulate SHUFFLE phase
                st.markdown("#### 2️⃣ SHUFFLE Phase")
                shuffled = {}
                for job, balance in map_output:
                    if job not in shuffled:
                        shuffled[job] = []
                    shuffled[job].append(balance)

                st.success(f"✅ Grouped into {len(shuffled)} keys")

                # Simulate REDUCE phase
                st.markdown("#### 3️⃣ REDUCE Phase")
                results = []
                for job, balances in shuffled.items():
                    avg_balance = np.mean(balances)
                    results.append({'Job': job, 'Average Balance': round(avg_balance, 2), 'Count': len(balances)})

                results_df = pd.DataFrame(results).sort_values('Average Balance', ascending=False)
                st.success("✅ Reduce complete!")
                st.dataframe(results_df, use_container_width=True, hide_index=True)

                # Visualization
                fig = px.bar(results_df, x='Job', y='Average Balance', color='Average Balance',
                           title="MapReduce Result: Average Balance by Job")
                st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PART 3: HIVE ANALYTICS
# ============================================================
elif page == "📝 Part 3: Hive Analytics":
    st.markdown('<div class="section-header"><h2>📝 Part 3: Hive Analytics</h2></div>', unsafe_allow_html=True)

    st.markdown("""
    ### 🎯 What is Apache Hive?

    Hive is a **data warehouse** system that allows you to query big data using **SQL-like syntax** (HiveQL).
    It translates SQL queries into MapReduce jobs automatically!
    """)

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📝 HiveQL Queries", "▶️ Run Queries", "📊 Results"])

    with tab1:
        st.markdown("### 📝 HiveQL Queries")

        st.markdown("#### 1. Create Database and Table")
        st.code('''
-- Create database
CREATE DATABASE IF NOT EXISTS banking_analytics;
USE banking_analytics;

-- Create table
CREATE TABLE IF NOT EXISTS bank_data (
    age INT,
    job STRING,
    marital STRING,
    education STRING,
    default_status STRING,
    balance INT,
    housing STRING,
    loan STRING,
    contact STRING,
    day INT,
    month STRING,
    duration INT,
    campaign INT,
    pdays INT,
    previous INT,
    poutcome STRING,
    subscribed STRING
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE;

-- Load data
LOAD DATA INPATH '/user/data/bank.csv' INTO TABLE bank_data;
        ''', language='sql')

        st.markdown("#### 2. Analytical Queries")
        st.code('''
-- Query 1: Average balance by job type
SELECT job,
       ROUND(AVG(balance), 2) as avg_balance,
       COUNT(*) as total_clients
FROM bank_data
GROUP BY job
ORDER BY avg_balance DESC;

-- Query 2: Subscription rate by education
SELECT education,
       COUNT(*) as total,
       SUM(CASE WHEN subscribed = 'yes' THEN 1 ELSE 0 END) as subscribed,
       ROUND(SUM(CASE WHEN subscribed = 'yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as rate
FROM bank_data
GROUP BY education
ORDER BY rate DESC;

-- Query 3: Monthly campaign analysis
SELECT month,
       COUNT(*) as contacts,
       ROUND(AVG(duration), 2) as avg_duration,
       SUM(CASE WHEN subscribed = 'yes' THEN 1 ELSE 0 END) as conversions
FROM bank_data
GROUP BY month
ORDER BY contacts DESC;

-- Query 4: Best performing contact method
SELECT contact,
       COUNT(*) as total,
       ROUND(SUM(CASE WHEN subscribed = 'yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as success_rate
FROM bank_data
GROUP BY contact
ORDER BY success_rate DESC;
        ''', language='sql')

    with tab2:
        st.markdown("### ▶️ Interactive SQL Query Runner")

        # Create a simple SQL simulator
        query_options = {
            "Average Balance by Job": """
SELECT job, AVG(balance) as avg_balance, COUNT(*) as count
FROM bank_data
GROUP BY job
ORDER BY avg_balance DESC
            """,
            "Subscription Rate by Education": """
SELECT education,
       COUNT(*) as total,
       SUM(subscribed='yes') as subscribed,
       ROUND(SUM(subscribed='yes')*100.0/COUNT(*), 2) as rate
FROM bank_data
GROUP BY education
            """,
            "Monthly Contacts": """
SELECT month, COUNT(*) as contacts, AVG(duration) as avg_duration
FROM bank_data
GROUP BY month
ORDER BY contacts DESC
            """,
            "Success by Contact Type": """
SELECT contact, COUNT(*) as total,
       ROUND(SUM(subscribed='yes')*100.0/COUNT(*), 2) as success_rate
FROM bank_data
GROUP BY contact
            """
        }

        selected_query = st.selectbox("Select a query to run:", list(query_options.keys()))

        st.code(query_options[selected_query], language='sql')

        if st.button("▶️ Run Query", use_container_width=True):
            with st.spinner("Executing query..."):
                import time
                time.sleep(1)

                if selected_query == "Average Balance by Job":
                    result = df.groupby('job')['balance'].agg(['mean', 'count']).round(2).reset_index()
                    result.columns = ['job', 'avg_balance', 'count']
                    result = result.sort_values('avg_balance', ascending=False)

                elif selected_query == "Subscription Rate by Education":
                    result = df.groupby('education').agg(
                        total=('y', 'count'),
                        subscribed=('y', lambda x: (x=='yes').sum())
                    ).reset_index()
                    result['rate'] = (result['subscribed'] / result['total'] * 100).round(2)

                elif selected_query == "Monthly Contacts":
                    result = df.groupby('month').agg(
                        contacts=('month', 'count'),
                        avg_duration=('duration', 'mean')
                    ).round(2).reset_index()
                    result = result.sort_values('contacts', ascending=False)

                else:
                    result = df.groupby('contact').agg(
                        total=('contact', 'count'),
                        subscribed=('y', lambda x: (x=='yes').sum())
                    ).reset_index()
                    result['success_rate'] = (result['subscribed'] / result['total'] * 100).round(2)

                st.success("✅ Query executed successfully!")
                st.dataframe(result, use_container_width=True, hide_index=True)

    with tab3:
        st.markdown("### 📊 Query Results Visualization")

        # Show all query results with charts
        col1, col2 = st.columns(2)

        with col1:
            # Balance by job
            job_bal = df.groupby('job')['balance'].mean().sort_values(ascending=True).reset_index()
            fig = px.bar(job_bal, x='balance', y='job', orientation='h',
                        title="Average Balance by Job (Hive Query Result)",
                        color='balance', color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Education subscription
            edu_sub = df.groupby('education').apply(lambda x: (x['y']=='yes').mean()*100).reset_index()
            edu_sub.columns = ['education', 'rate']
            fig = px.pie(edu_sub, values='rate', names='education',
                        title="Subscription Rate by Education")
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PART 4: MACHINE LEARNING
# ============================================================
elif page == "🤖 Part 4: Machine Learning":
    st.markdown('<div class="section-header"><h2>🤖 Part 4: Machine Learning</h2></div>', unsafe_allow_html=True)

    st.markdown("""
    ### 🎯 Objective
    Build a **classification model** to predict whether a customer will subscribe to the term deposit.
    """)

    tab1, tab2, tab3, tab4 = st.tabs(["📚 ML Concepts", "🔧 Preprocessing", "🏋️ Train Model", "📊 Results"])

    with tab1:
        st.markdown("### 📚 Machine Learning Concepts")

        st.markdown("""
        #### 🎯 Classification Problem
        We're predicting a **binary outcome**: Will the customer subscribe? (Yes/No)

        #### 📊 Key Metrics Explained:
        """)

        metric_col1, metric_col2 = st.columns(2)

        with metric_col1:
            st.markdown("""
            **Accuracy**
            - % of correct predictions
            - Formula: (TP + TN) / Total

            **Precision**
            - When we predict "Yes", how often correct?
            - Formula: TP / (TP + FP)
            """)

        with metric_col2:
            st.markdown("""
            **Recall**
            - Of all actual "Yes", how many did we catch?
            - Formula: TP / (TP + FN)

            **AUC-ROC**
            - Overall model quality (0.5=random, 1.0=perfect)
            - Area under the ROC curve
            """)

        st.markdown("""
        #### 🤖 Models Used:

        | Model | Description | Pros | Cons |
        |-------|-------------|------|------|
        | Logistic Regression | Linear model for classification | Fast, interpretable | Limited to linear boundaries |
        | Decision Tree | Tree-based rules | Easy to understand | Can overfit |
        | Random Forest | Ensemble of trees | High accuracy | Slower, less interpretable |
        | Gradient Boosting | Sequential ensemble | Best accuracy | Complex, slow to train |
        """)

    with tab2:
        st.markdown("### 🔧 Data Preprocessing")

        st.markdown("""
        #### Steps:
        1. **Handle Missing Values** - Check and fill/remove
        2. **Encode Categorical Variables** - Convert text to numbers
        3. **Scale Numerical Features** - Normalize values
        4. **Split Data** - 80% train, 20% test
        """)

        st.code('''
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Define columns
numerical_cols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
categorical_cols = ['job', 'marital', 'education', 'default', 'housing',
                    'loan', 'contact', 'month', 'poutcome']

# Create preprocessor
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
])

# Prepare features and target
X = df.drop('y', axis=1)
y = (df['y'] == 'yes').astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {len(X_train)}")
print(f"Test set: {len(X_test)}")
        ''', language='python')

        # Show actual split
        X = df.drop('y', axis=1)
        y = (df['y'] == 'yes').astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Training Samples", len(X_train))
        with col2:
            st.metric("Test Samples", len(X_test))

    with tab3:
        st.markdown("### 🏋️ Train Machine Learning Model")

        if st.button("🚀 Train Models", use_container_width=True):
            with st.spinner("Training models... This may take a moment."):
                # Prepare data
                numerical_cols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
                categorical_cols = ['job', 'marital', 'education', 'default', 'housing',
                                   'loan', 'contact', 'month', 'poutcome']

                preprocessor = ColumnTransformer([
                    ('num', StandardScaler(), numerical_cols),
                    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categorical_cols)
                ])

                X = df[numerical_cols + categorical_cols]
                y = (df['y'] == 'yes').astype(int)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

                results = {}

                # Train Logistic Regression
                st.markdown("#### Training Logistic Regression...")
                lr_pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
                ])
                lr_pipeline.fit(X_train, y_train)
                y_pred = lr_pipeline.predict(X_test)
                y_proba = lr_pipeline.predict_proba(X_test)[:, 1]

                results['Logistic Regression'] = {
                    'Accuracy': accuracy_score(y_test, y_pred),
                    'AUC-ROC': roc_auc_score(y_test, y_proba)
                }

                # Train Random Forest
                st.markdown("#### Training Random Forest...")
                rf_pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
                ])
                rf_pipeline.fit(X_train, y_train)
                y_pred_rf = rf_pipeline.predict(X_test)
                y_proba_rf = rf_pipeline.predict_proba(X_test)[:, 1]

                results['Random Forest'] = {
                    'Accuracy': accuracy_score(y_test, y_pred_rf),
                    'AUC-ROC': roc_auc_score(y_test, y_proba_rf)
                }

                st.success("✅ Training Complete!")

                # Display results
                results_df = pd.DataFrame(results).T.reset_index()
                results_df.columns = ['Model', 'Accuracy', 'AUC-ROC']
                results_df = results_df.round(4)

                st.dataframe(results_df, use_container_width=True, hide_index=True)

                # Feature Importance
                st.markdown("#### 🎯 Feature Importance (Random Forest)")
                feature_names = numerical_cols + list(rf_pipeline.named_steps['preprocessor']
                                                     .named_transformers_['cat']
                                                     .get_feature_names_out(categorical_cols))
                importance = rf_pipeline.named_steps['classifier'].feature_importances_

                feat_imp = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importance
                }).sort_values('Importance', ascending=False).head(10)

                fig = px.bar(feat_imp, x='Importance', y='Feature', orientation='h',
                           title="Top 10 Feature Importances", color='Importance')
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)

                # Store in session state
                st.session_state['model_trained'] = True
                st.session_state['results'] = results

    with tab4:
        st.markdown("### 📊 Model Results")

        # Show pre-computed results
        model_results = pd.DataFrame({
            'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting'],
            'AUC-ROC': [0.8886, 0.6781, 0.8831, 0.8883],
            'Accuracy': [0.8928, 0.8597, 0.8873, 0.8928],
            'Precision': [0.5636, 0.4000, 0.5192, 0.5538],
            'Recall': [0.2981, 0.4423, 0.2596, 0.3462],
            'F1-Score': [0.3899, 0.4201, 0.3462, 0.4260]
        })

        st.dataframe(model_results.style.highlight_max(axis=0,
                    subset=['AUC-ROC', 'Accuracy', 'Precision', 'Recall', 'F1-Score'],
                    color='lightgreen'), use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(model_results, x='Model', y='AUC-ROC', color='AUC-ROC',
                        title="Model Comparison: AUC-ROC", color_continuous_scale='Viridis')
            fig.add_hline(y=0.5, line_dash="dash", line_color="red",
                         annotation_text="Random Baseline")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Confusion Matrix visualization
            cm = np.array([[777, 24], [73, 31]])
            fig = px.imshow(cm, text_auto=True,
                          labels=dict(x="Predicted", y="Actual"),
                          x=['No', 'Yes'], y=['No', 'Yes'],
                          title="Confusion Matrix (Logistic Regression)")
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PART 5: SPARK STREAMING
# ============================================================
elif page == "⚡ Part 5: Spark Streaming":
    st.markdown('<div class="section-header"><h2>⚡ Part 5: Spark Streaming</h2></div>', unsafe_allow_html=True)

    st.markdown("""
    ### 🎯 What is Spark Streaming?

    Spark Streaming enables **real-time data processing**. Instead of processing data in batches,
    it processes data as it arrives - like analyzing live phone calls as they happen!
    """)

    st.markdown("---")

    # Streaming visualization
    st.markdown("### 📊 Real-Time Processing Concept")

    st.markdown("""
    ```
    REAL-TIME DATA FLOW
    ═══════════════════════════════════════════════════════════════════════════

    Live Data Source          Spark Streaming              Real-Time Output
    ┌─────────────┐          ┌─────────────────┐          ┌─────────────────┐
    │  📞 Call 1  │ ──────►  │                 │          │ Current Stats:  │
    │  📞 Call 2  │ ──────►  │  Process in     │ ──────►  │ • Avg Duration  │
    │  📞 Call 3  │ ──────►  │  Micro-batches  │          │ • Total Calls   │
    │  📞 Call 4  │ ──────►  │  (every 10 sec) │          │ • Success Rate  │
    │     ...     │          │                 │          │                 │
    └─────────────┘          └─────────────────┘          └─────────────────┘
           │                        │                            │
           ▼                        ▼                            ▼
        Continuous              Window-based                 Live Dashboard
        Input                   Aggregation                  Updates
    ═══════════════════════════════════════════════════════════════════════════
    ```
    """)

    tab1, tab2, tab3 = st.tabs(["📚 Concepts", "💻 Code", "▶️ Simulation"])

    with tab1:
        st.markdown("### 📚 Key Streaming Concepts")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            #### 🪟 Window Operations
            Process data in time windows:
            - **Tumbling Window**: Fixed, non-overlapping (every 10 sec)
            - **Sliding Window**: Overlapping (10 sec window, slides every 5 sec)

            #### ⏱️ Watermarking
            Handle late-arriving data:
            - Set threshold (e.g., 30 seconds)
            - Data within threshold is still processed
            - Older data is dropped
            """)

        with col2:
            st.markdown("""
            #### 📊 Aggregations
            Real-time calculations:
            - Running averages
            - Counts per window
            - Group-by operations

            #### 💾 Output Modes
            - **Append**: Only new rows
            - **Complete**: All rows
            - **Update**: Changed rows only
            """)

    with tab2:
        st.markdown("### 💻 Spark Streaming Code")

        st.code('''
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

# Initialize Spark Session
spark = SparkSession.builder \\
    .appName("Banking Real-Time Streaming") \\
    .getOrCreate()

# Define schema for streaming data
schema = StructType([
    StructField("age", IntegerType(), True),
    StructField("job", StringType(), True),
    StructField("balance", IntegerType(), True),
    StructField("duration", IntegerType(), True),
    StructField("y", StringType(), True),
    StructField("timestamp", TimestampType(), True)
])

# Read streaming data from directory
streaming_df = spark.readStream \\
    .schema(schema) \\
    .option("maxFilesPerTrigger", 1) \\
    .csv("data/stream_input")

# Add processing timestamp
streaming_df = streaming_df.withColumn("processing_time", current_timestamp())

# Real-time aggregation: Average by job type
job_aggregation = streaming_df \\
    .groupBy("job") \\
    .agg(
        count("*").alias("transaction_count"),
        avg("balance").alias("avg_balance"),
        avg("duration").alias("avg_duration")
    )

# Window-based aggregation (10-second windows)
window_agg = streaming_df \\
    .withWatermark("processing_time", "30 seconds") \\
    .groupBy(
        window(col("processing_time"), "10 seconds", "5 seconds"),
        "job"
    ) \\
    .agg(
        count("*").alias("count"),
        avg("balance").alias("avg_balance")
    )

# Write to console
query = job_aggregation \\
    .writeStream \\
    .outputMode("complete") \\
    .format("console") \\
    .start()

query.awaitTermination()
        ''', language='python')

    with tab3:
        st.markdown("### ▶️ Streaming Simulation")

        if st.button("🚀 Start Streaming Simulation", use_container_width=True):
            import time

            # Create placeholders for live updates
            metrics_placeholder = st.empty()
            chart_placeholder = st.empty()
            data_placeholder = st.empty()

            # Simulate streaming
            all_data = []

            for batch in range(1, 11):
                # Simulate new data arriving
                batch_size = np.random.randint(5, 15)
                new_data = df.sample(batch_size)
                new_data['batch'] = batch
                new_data['timestamp'] = pd.Timestamp.now()
                all_data.append(new_data)

                combined = pd.concat(all_data)

                # Update metrics
                with metrics_placeholder.container():
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Batch #", batch)
                    m2.metric("Total Records", len(combined))
                    m3.metric("Avg Balance", f"${combined['balance'].mean():.0f}")
                    m4.metric("Avg Duration", f"{combined['duration'].mean():.0f}s")

                # Update chart
                with chart_placeholder.container():
                    job_agg = combined.groupby('job').agg({
                        'balance': 'mean',
                        'duration': 'mean',
                        'job': 'count'
                    }).rename(columns={'job': 'count'}).reset_index()

                    fig = px.bar(job_agg, x='job', y='balance',
                               title=f"Live Aggregation - Batch {batch}",
                               color='balance', color_continuous_scale='Blues')
                    st.plotly_chart(fig, use_container_width=True)

                # Update data table
                with data_placeholder.container():
                    st.markdown(f"**Latest Batch Data (Batch {batch}):**")
                    st.dataframe(new_data[['age', 'job', 'balance', 'duration', 'y']].head(5),
                               use_container_width=True, hide_index=True)

                time.sleep(1)

            st.success("✅ Streaming simulation complete! Processed 10 batches.")

# ============================================================
# PART 6: DATA PARALLELISM
# ============================================================
elif page == "🚀 Part 6: Data Parallelism":
    st.markdown('<div class="section-header"><h2>🚀 Part 6: Data Parallelism</h2></div>', unsafe_allow_html=True)

    st.markdown("""
    ### 🎯 What is Data Parallelism?

    Data parallelism divides data across multiple processors/cores to process **simultaneously**,
    dramatically reducing processing time for large datasets.
    """)

    st.markdown("---")

    # Visual explanation
    st.markdown("### 📊 Sequential vs Parallel Processing")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### ❌ Sequential Processing
        ```
        Time ─────────────────────────────►

        CPU 1: [████ Part 1 ████][████ Part 2 ████][████ Part 3 ████][████ Part 4 ████]

        Total Time: ████████████████████████████████████████████████
                    40 seconds
        ```
        """)

    with col2:
        st.markdown("""
        #### ✅ Parallel Processing
        ```
        Time ────────────────►

        CPU 1: [████ Part 1 ████]
        CPU 2: [████ Part 2 ████]
        CPU 3: [████ Part 3 ████]
        CPU 4: [████ Part 4 ████]

        Total Time: ██████████
                    10 seconds (4x faster!)
        ```
        """)

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📚 Concepts", "💻 Code", "▶️ Demo"])

    with tab1:
        st.markdown("### 📚 Parallelism Concepts")

        st.markdown("""
        #### 🔢 Partitioning Strategies

        | Strategy | Description | Best For |
        |----------|-------------|----------|
        | **Hash Partitioning** | Distribute by hash of key | Uniform distribution |
        | **Range Partitioning** | Distribute by value ranges | Sorted data |
        | **Round Robin** | Distribute evenly in order | General purpose |

        #### ⚡ Spark Parallelism

        ```python
        # Set number of partitions
        df.repartition(4)  # Create 4 partitions

        # Process each partition independently
        df.foreachPartition(process_partition)

        # Parallel map operations
        df.rdd.map(lambda x: x * 2)
        ```
        """)

    with tab2:
        st.markdown("### 💻 Data Parallelism Code")

        st.code('''
from pyspark.sql import SparkSession
import time

# Initialize Spark
spark = SparkSession.builder \\
    .appName("Data Parallelism Demo") \\
    .config("spark.executor.cores", "4") \\
    .getOrCreate()

# Load data
df = spark.read.csv("data/bank.csv", header=True, inferSchema=True)

# Check current partitions
print(f"Default partitions: {df.rdd.getNumPartitions()}")

# Repartition for parallel processing
df_parallel = df.repartition(4)
print(f"New partitions: {df_parallel.rdd.getNumPartitions()}")

# Process partitions in parallel
def process_partition(iterator):
    """Process each partition independently"""
    results = []
    for row in iterator:
        # Heavy computation here
        result = row['balance'] * 2
        results.append(result)
    return iter(results)

# Apply parallel processing
start_time = time.time()
result = df_parallel.rdd.mapPartitions(process_partition).collect()
parallel_time = time.time() - start_time

print(f"Parallel processing time: {parallel_time:.2f}s")

# Compare with sequential
start_time = time.time()
result_seq = [row['balance'] * 2 for row in df.collect()]
sequential_time = time.time() - start_time

print(f"Sequential processing time: {sequential_time:.2f}s")
print(f"Speedup: {sequential_time/parallel_time:.2f}x")
        ''', language='python')

    with tab3:
        st.markdown("### ▶️ Parallelism Demonstration")

        num_partitions = st.slider("Number of Partitions", 1, 8, 4)

        if st.button("🚀 Run Parallel vs Sequential Comparison", use_container_width=True):
            import time
            from concurrent.futures import ThreadPoolExecutor

            # Prepare data
            data = df['balance'].values

            # Simulate heavy computation
            def heavy_computation(x):
                # Simulate work
                result = 0
                for _ in range(1000):
                    result += x * 0.001
                return result

            # Sequential processing
            st.markdown("#### ⏱️ Sequential Processing...")
            start = time.time()
            progress_seq = st.progress(0)

            seq_results = []
            for i, val in enumerate(data[:500]):  # Use subset for demo
                seq_results.append(heavy_computation(val))
                if i % 50 == 0:
                    progress_seq.progress((i+1)/500)

            seq_time = time.time() - start

            # Parallel processing
            st.markdown("#### ⚡ Parallel Processing...")
            start = time.time()
            progress_par = st.progress(0)

            with ThreadPoolExecutor(max_workers=num_partitions) as executor:
                par_results = list(executor.map(heavy_computation, data[:500]))
            progress_par.progress(1.0)

            par_time = time.time() - start

            # Results
            st.markdown("---")
            st.markdown("### 📊 Results")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Sequential Time", f"{seq_time:.2f}s")
            with col2:
                st.metric("Parallel Time", f"{par_time:.2f}s")
            with col3:
                speedup = seq_time / par_time if par_time > 0 else 1
                st.metric("Speedup", f"{speedup:.2f}x")

            # Visualization
            fig = go.Figure(data=[
                go.Bar(name='Sequential', x=['Processing Time'], y=[seq_time], marker_color='#e74c3c'),
                go.Bar(name=f'Parallel ({num_partitions} workers)', x=['Processing Time'], y=[par_time], marker_color='#2ecc71')
            ])
            fig.update_layout(title="Sequential vs Parallel Processing Time",
                            yaxis_title="Time (seconds)", barmode='group')
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# INTERACTIVE DASHBOARD
# ============================================================
elif page == "📈 Interactive Dashboard":
    st.markdown('<div class="section-header"><h2>📈 Interactive Dashboard</h2></div>', unsafe_allow_html=True)

    # Initialize session state for filters
    if 'dash_subscription' not in st.session_state:
        st.session_state.dash_subscription = "All"
    if 'dash_balance_min' not in st.session_state:
        st.session_state.dash_balance_min = int(df['balance'].min())
    if 'dash_age_min' not in st.session_state:
        st.session_state.dash_age_min = int(df['age'].min())
    if 'dash_age_max' not in st.session_state:
        st.session_state.dash_age_max = int(df['age'].max())
    if 'dash_job' not in st.session_state:
        st.session_state.dash_job = "All"
    if 'dash_education' not in st.session_state:
        st.session_state.dash_education = "All"

    # ============ QUICK FILTER BUTTONS ============
    st.markdown("### 🎛️ Quick Filter Buttons")
    st.markdown("Click any button to instantly filter the data:")

    btn_col1, btn_col2, btn_col3, btn_col4, btn_col5 = st.columns(5)

    with btn_col1:
        if st.button("📊 Show All Data", key="btn_all", use_container_width=True):
            st.session_state.dash_subscription = "All"
            st.session_state.dash_balance_min = int(df['balance'].min())
            st.session_state.dash_age_min = int(df['age'].min())
            st.session_state.dash_age_max = int(df['age'].max())
            st.session_state.dash_job = "All"
            st.session_state.dash_education = "All"
            st.rerun()

    with btn_col2:
        if st.button("✅ Subscribed Only", key="btn_yes", use_container_width=True):
            st.session_state.dash_subscription = "Yes"
            st.rerun()

    with btn_col3:
        if st.button("❌ Not Subscribed", key="btn_no", use_container_width=True):
            st.session_state.dash_subscription = "No"
            st.rerun()

    with btn_col4:
        if st.button("💰 High Balance (>5000)", key="btn_balance", use_container_width=True):
            st.session_state.dash_balance_min = 5000
            st.rerun()

    with btn_col5:
        if st.button("👴 Senior (Age>50)", key="btn_senior", use_container_width=True):
            st.session_state.dash_age_min = 50
            st.rerun()

    # Second row of buttons
    btn_col6, btn_col7, btn_col8, btn_col9, btn_col10 = st.columns(5)

    with btn_col6:
        if st.button("👶 Young (Age<30)", key="btn_young", use_container_width=True):
            st.session_state.dash_age_max = 30
            st.session_state.dash_age_min = int(df['age'].min())
            st.rerun()

    with btn_col7:
        if st.button("💼 Management", key="btn_mgmt", use_container_width=True):
            st.session_state.dash_job = "management"
            st.rerun()

    with btn_col8:
        if st.button("🔧 Technician", key="btn_tech", use_container_width=True):
            st.session_state.dash_job = "technician"
            st.rerun()

    with btn_col9:
        if st.button("🎓 Tertiary Edu", key="btn_tertiary", use_container_width=True):
            st.session_state.dash_education = "tertiary"
            st.rerun()

    with btn_col10:
        if st.button("📚 Secondary Edu", key="btn_secondary", use_container_width=True):
            st.session_state.dash_education = "secondary"
            st.rerun()

    st.markdown("---")

    # ============ ADVANCED FILTERS ============
    with st.expander("🔧 Advanced Filters (Click to expand)", expanded=False):
        adv_col1, adv_col2, adv_col3, adv_col4 = st.columns(4)

        with adv_col1:
            job_options = ['All'] + list(df['job'].unique())
            selected_job = st.selectbox("👔 Job Type", job_options,
                                       index=job_options.index(st.session_state.dash_job) if st.session_state.dash_job in job_options else 0)
            st.session_state.dash_job = selected_job

        with adv_col2:
            edu_options = ['All'] + list(df['education'].unique())
            selected_edu = st.selectbox("🎓 Education", edu_options,
                                       index=edu_options.index(st.session_state.dash_education) if st.session_state.dash_education in edu_options else 0)
            st.session_state.dash_education = selected_edu

        with adv_col3:
            age_range = st.slider("📅 Age Range",
                                 int(df['age'].min()), int(df['age'].max()),
                                 (st.session_state.dash_age_min, st.session_state.dash_age_max))
            st.session_state.dash_age_min = age_range[0]
            st.session_state.dash_age_max = age_range[1]

        with adv_col4:
            balance_min = st.number_input("💵 Min Balance", value=st.session_state.dash_balance_min)
            st.session_state.dash_balance_min = balance_min

    # ============ APPLY FILTERS ============
    filtered_df = df.copy()

    # Subscription filter
    if st.session_state.dash_subscription == "Yes":
        filtered_df = filtered_df[filtered_df['y'] == 'yes']
    elif st.session_state.dash_subscription == "No":
        filtered_df = filtered_df[filtered_df['y'] == 'no']

    # Job filter
    if st.session_state.dash_job != "All":
        filtered_df = filtered_df[filtered_df['job'] == st.session_state.dash_job]

    # Education filter
    if st.session_state.dash_education != "All":
        filtered_df = filtered_df[filtered_df['education'] == st.session_state.dash_education]

    # Age filter
    filtered_df = filtered_df[filtered_df['age'].between(st.session_state.dash_age_min, st.session_state.dash_age_max)]

    # Balance filter
    filtered_df = filtered_df[filtered_df['balance'] >= st.session_state.dash_balance_min]

    # Show active filters
    active_filters = []
    if st.session_state.dash_subscription != "All":
        active_filters.append(f"Subscription: {st.session_state.dash_subscription}")
    if st.session_state.dash_job != "All":
        active_filters.append(f"Job: {st.session_state.dash_job}")
    if st.session_state.dash_education != "All":
        active_filters.append(f"Education: {st.session_state.dash_education}")
    if st.session_state.dash_balance_min > int(df['balance'].min()):
        active_filters.append(f"Balance > ${st.session_state.dash_balance_min}")
    if st.session_state.dash_age_min > int(df['age'].min()) or st.session_state.dash_age_max < int(df['age'].max()):
        active_filters.append(f"Age: {st.session_state.dash_age_min}-{st.session_state.dash_age_max}")

    filter_text = " | ".join(active_filters) if active_filters else "No filters applied"
    st.markdown(f"**📌 Active Filters:** {filter_text}")
    st.markdown(f"**📊 Showing {len(filtered_df):,} of {len(df):,} records**")

    st.markdown("---")

    # Metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Clients", f"{len(filtered_df):,}")
    m2.metric("Subscription Rate", f"{(filtered_df['y']=='yes').mean()*100:.1f}%")
    m3.metric("Avg Balance", f"${filtered_df['balance'].mean():,.0f}")
    m4.metric("Avg Duration", f"{filtered_df['duration'].mean():.0f}s")
    m5.metric("Avg Age", f"{filtered_df['age'].mean():.1f}")

    st.markdown("---")

    # Charts
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        job_counts = filtered_df['job'].value_counts().reset_index()
        job_counts.columns = ['Job', 'Count']
        fig = px.bar(job_counts, x='Job', y='Count', title="Distribution by Job",
                    color='Count', color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)

    with chart_col2:
        sub_counts = filtered_df['y'].value_counts().reset_index()
        sub_counts.columns = ['Subscribed', 'Count']
        fig = px.pie(sub_counts, values='Count', names='Subscribed', title="Subscription",
                    color_discrete_sequence=['#2ecc71', '#e74c3c'])
        st.plotly_chart(fig, use_container_width=True)

    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        fig = px.histogram(filtered_df, x='age', nbins=30, title="Age Distribution",
                          color_discrete_sequence=['#3498db'])
        st.plotly_chart(fig, use_container_width=True)

    with chart_col4:
        job_bal = filtered_df.groupby('job')['balance'].mean().sort_values(ascending=True).reset_index()
        fig = px.bar(job_bal, x='balance', y='job', orientation='h',
                    title="Avg Balance by Job", color='balance')
        st.plotly_chart(fig, use_container_width=True)

    # Data table
    st.markdown("### 📋 Data Preview")
    st.dataframe(filtered_df.head(50), use_container_width=True, hide_index=True)

# ============================================================
# LEARNING GUIDE
# ============================================================
elif page == "📖 Learning Guide":
    st.markdown('<div class="section-header"><h2>📖 Learning Guide</h2></div>', unsafe_allow_html=True)

    st.markdown("""
    ### 🎓 Knowledge Required for This Project

    This guide covers all the concepts you need to understand this project.
    """)

    tab1, tab2, tab3, tab4 = st.tabs(["🐍 Python", "📊 Data Science", "🤖 ML", "🔧 Big Data"])

    with tab1:
        st.markdown("""
        ### 🐍 Python Programming

        #### Basic Concepts
        ```python
        # Variables and data types
        age = 25
        name = "John"
        balance = 1500.50
        is_subscribed = True

        # Lists and dictionaries
        jobs = ['management', 'technician', 'admin']
        client = {'name': 'John', 'age': 25, 'balance': 1500}

        # Loops
        for job in jobs:
            print(job)

        # Functions
        def calculate_average(numbers):
            return sum(numbers) / len(numbers)
        ```

        #### Pandas Basics
        ```python
        import pandas as pd

        # Read data
        df = pd.read_csv('data.csv')

        # View data
        df.head()        # First 5 rows
        df.describe()    # Statistics
        df.info()        # Data types

        # Filter data
        df[df['age'] > 30]
        df[df['job'] == 'management']

        # Group and aggregate
        df.groupby('job')['balance'].mean()
        ```
        """)

    with tab2:
        st.markdown("""
        ### 📊 Data Science Concepts

        #### Exploratory Data Analysis (EDA)
        - **Descriptive Statistics**: Mean, median, std, min, max
        - **Data Distribution**: How values are spread
        - **Correlation**: Relationship between variables
        - **Missing Values**: Handling incomplete data

        #### Visualization Types
        | Chart | Use Case |
        |-------|----------|
        | Bar Chart | Compare categories |
        | Histogram | Show distribution |
        | Pie Chart | Show proportions |
        | Scatter Plot | Show relationships |
        | Box Plot | Show outliers |

        #### Data Preprocessing
        1. **Cleaning**: Remove duplicates, fix errors
        2. **Transformation**: Normalize, scale values
        3. **Encoding**: Convert categories to numbers
        4. **Feature Engineering**: Create new features
        """)

    with tab3:
        st.markdown("""
        ### 🤖 Machine Learning

        #### Classification Problem
        - **Goal**: Predict a category (Yes/No, Spam/Not Spam)
        - **Input**: Features (age, job, balance, etc.)
        - **Output**: Class label (subscribed: yes/no)

        #### Model Evaluation Metrics

        ```
        Confusion Matrix:
                          Predicted
                        No      Yes
        Actual  No      TN      FP    ← False Positive (Type I Error)
                Yes     FN      TP    ← False Negative (Type II Error)

        Accuracy  = (TP + TN) / Total
        Precision = TP / (TP + FP)   → "When I say Yes, am I right?"
        Recall    = TP / (TP + FN)   → "Did I catch all the Yes?"
        F1-Score  = 2 * (Precision * Recall) / (Precision + Recall)
        ```

        #### Common Algorithms
        - **Logistic Regression**: Linear, fast, interpretable
        - **Decision Tree**: Rules-based, easy to understand
        - **Random Forest**: Ensemble of trees, high accuracy
        - **Gradient Boosting**: Sequential learning, best performance
        """)

    with tab4:
        st.markdown("""
        ### 🔧 Big Data Technologies

        #### Apache Spark
        - Distributed computing framework
        - In-memory processing (fast!)
        - APIs: SQL, DataFrame, ML, Streaming

        #### Hadoop MapReduce
        - Batch processing paradigm
        - Map: Extract key-value pairs
        - Reduce: Aggregate by key
        - Fault-tolerant, scalable

        #### Apache Hive
        - SQL interface for Big Data
        - Translates queries to MapReduce
        - Schema-on-read approach

        #### Key Concepts

        | Term | Meaning |
        |------|---------|
        | **Partition** | Subset of data for parallel processing |
        | **Shuffle** | Redistributing data between nodes |
        | **RDD** | Resilient Distributed Dataset (Spark) |
        | **DataFrame** | Structured data with schema |
        | **Streaming** | Processing data in real-time |
        | **Batch** | Processing data in groups |
        """)

    st.markdown("---")

    st.markdown("""
    ### 📚 Additional Resources

    - [Pandas Documentation](https://pandas.pydata.org/docs/)
    - [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
    - [Apache Spark Documentation](https://spark.apache.org/docs/latest/)
    - [Hadoop MapReduce Tutorial](https://hadoop.apache.org/docs/stable/hadoop-mapreduce-client/hadoop-mapreduce-client-core/MapReduceTutorial.html)
    """)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>🏦 Banking Analytics Dashboard | Complete Big Data Project | Built with Streamlit</p>",
    unsafe_allow_html=True
)
