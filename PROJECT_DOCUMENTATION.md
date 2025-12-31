# Banking Data Analytics with Distributed Computing
## Complete End-to-End Project Documentation

---

# Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Project Overview](#2-project-overview)
3. [Dataset Description](#3-dataset-description)
4. [Part 1: Data Processing with Spark/Pandas](#4-part-1-data-processing)
5. [Part 2: Hadoop MapReduce](#5-part-2-hadoop-mapreduce)
6. [Part 3: Hive Analytics](#6-part-3-hive-analytics)
7. [Part 4: Machine Learning](#7-part-4-machine-learning)
8. [Part 5: Spark Streaming](#8-part-5-spark-streaming)
9. [Part 6: Data Parallelism](#9-part-6-data-parallelism)
10. [Key Findings & Insights](#10-key-findings)
11. [Technical Architecture](#11-technical-architecture)
12. [Conclusion](#12-conclusion)

---

# 1. Executive Summary

This project demonstrates a comprehensive big data analytics pipeline for analyzing bank marketing campaign data. Using distributed computing technologies including Apache Spark, Hadoop MapReduce, and Apache Hive, we processed 4,521 customer records to understand factors affecting term deposit subscription rates.

**Key Achievements:**
- Built ML models achieving 89% accuracy in predicting customer subscription
- Identified call duration as the most important predictor (33% importance)
- Demonstrated real-time data processing with Spark Streaming
- Achieved significant speedup through data parallelism

---

# 2. Project Overview

## 2.1 Problem Statement

A Portuguese banking institution conducted direct marketing campaigns (phone calls) to promote term deposits. The goal is to:

1. **Understand** what factors influence a customer's decision to subscribe
2. **Predict** which customers are likely to subscribe in future campaigns
3. **Optimize** marketing efforts to improve success rates

## 2.2 Business Objectives

| Objective | Description |
|-----------|-------------|
| Customer Segmentation | Identify customer profiles most likely to subscribe |
| Campaign Optimization | Determine best times and methods to contact customers |
| Resource Allocation | Focus efforts on high-potential leads |
| Cost Reduction | Minimize wasted marketing efforts |

## 2.3 Technologies Used

| Technology | Purpose | Version |
|------------|---------|---------|
| Python | Primary programming language | 3.8+ |
| Apache Spark | Distributed data processing | 3.x |
| Hadoop MapReduce | Batch processing | 3.x |
| Apache Hive | SQL analytics | 3.x |
| Pandas | Data manipulation | 1.5+ |
| Scikit-learn | Machine learning | 1.3+ |
| Streamlit | Dashboard visualization | 1.28+ |
| Plotly | Interactive charts | 5.15+ |

---

# 3. Dataset Description

## 3.1 Data Source

- **Source**: UCI Machine Learning Repository - Bank Marketing Dataset
- **Records**: 4,521 customer interactions
- **Features**: 17 attributes
- **Target Variable**: `y` (has the client subscribed to a term deposit?)

## 3.2 Feature Description

### Demographic Features

| Feature | Type | Description | Example Values |
|---------|------|-------------|----------------|
| age | Numeric | Customer's age | 18-95 |
| job | Categorical | Type of occupation | management, technician, blue-collar |
| marital | Categorical | Marital status | married, single, divorced |
| education | Categorical | Education level | primary, secondary, tertiary |

### Financial Features

| Feature | Type | Description | Example Values |
|---------|------|-------------|----------------|
| default | Binary | Has credit in default? | yes, no |
| balance | Numeric | Average yearly balance (€) | -8019 to 102127 |
| housing | Binary | Has housing loan? | yes, no |
| loan | Binary | Has personal loan? | yes, no |

### Campaign Features

| Feature | Type | Description | Example Values |
|---------|------|-------------|----------------|
| contact | Categorical | Contact communication type | cellular, telephone, unknown |
| day | Numeric | Last contact day of month | 1-31 |
| month | Categorical | Last contact month | jan-dec |
| duration | Numeric | Last contact duration (seconds) | 0-4918 |
| campaign | Numeric | Contacts during this campaign | 1-63 |
| pdays | Numeric | Days since previous contact | -1 to 871 |
| previous | Numeric | Previous campaign contacts | 0-275 |
| poutcome | Categorical | Previous campaign outcome | success, failure, unknown |

### Target Variable

| Feature | Type | Description | Distribution |
|---------|------|-------------|--------------|
| y | Binary | Subscribed to term deposit? | yes: 11.5%, no: 88.5% |

## 3.3 Data Quality Assessment

```
Missing Values: 0
Duplicate Records: 0
Imbalanced Classes: Yes (88.5% No, 11.5% Yes)
Outliers: Present in balance, duration, campaign
```

---

# 4. Part 1: Data Processing with Spark/Pandas

## 4.1 Objective

Perform comprehensive Exploratory Data Analysis (EDA) to understand data distribution, patterns, and relationships.

## 4.2 Tasks Completed

### Task 1: Data Loading and Inspection
```python
# Load dataset
df = pd.read_csv("data/bank.csv")

# Basic inspection
print(f"Shape: {df.shape}")  # (4521, 17)
print(df.head())
print(df.describe())
```

### Task 2: Data Filtering
```python
# Filter high-balance customers
df_high_balance = df[df['balance'] > 1000]
# Result: 1,481 customers (32.8%)
```

### Task 3: Feature Engineering
```python
# Add quarter column
def get_quarter(month):
    quarters = {'jan':1, 'feb':1, 'mar':1, 'apr':2, 'may':2, 'jun':2,
                'jul':3, 'aug':3, 'sep':3, 'oct':4, 'nov':4, 'dec':4}
    return quarters[month]

df['quarter'] = df['month'].apply(get_quarter)
```

### Task 4: Aggregation Analysis
```python
# Average balance by job
job_stats = df.groupby('job').agg({
    'balance': 'mean',
    'age': 'mean'
}).round(2)
```

**Results:**
| Job | Avg Balance | Avg Age |
|-----|-------------|---------|
| retired | €2,319 | 62 |
| management | €1,767 | 41 |
| self-employed | €1,392 | 41 |

### Task 5: Correlation Analysis
```python
correlation = df['age'].corr(df['balance'])
# Result: 0.084 (weak positive correlation)
```

## 4.3 Key Visualizations

1. **Job Distribution**: Management (21.4%) and Blue-collar (20.9%) dominate
2. **Age Distribution**: Normal distribution, mean age 41
3. **Balance Distribution**: Right-skewed with outliers
4. **Subscription by Education**: Tertiary education has highest rate (14.3%)

---

# 5. Part 2: Hadoop MapReduce

## 5.1 Objective

Demonstrate distributed data processing using the MapReduce programming paradigm.

## 5.2 MapReduce Concept

```
INPUT → MAP → SHUFFLE & SORT → REDUCE → OUTPUT

┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│  Data   │ ──► │ Mapper  │ ──► │ Shuffle │ ──► │ Reducer │ ──► Results
│ Chunks  │     │(Extract)│     │ (Group) │     │(Combine)│
└─────────┘     └─────────┘     └─────────┘     └─────────┘
```

## 5.3 Implementation: Average Balance by Job

### Mapper (mapper_avg_balance.py)
```python
#!/usr/bin/env python3
import sys

header = True
for line in sys.stdin:
    if header:
        header = False
        continue

    fields = line.strip().split(',')
    job = fields[1]
    balance = fields[5]

    print(f"{job}\t{balance}")
```

### Reducer (reducer_avg_balance.py)
```python
#!/usr/bin/env python3
import sys

current_job = None
total_balance = 0
count = 0

for line in sys.stdin:
    job, balance = line.strip().split('\t')
    balance = float(balance)

    if current_job == job:
        total_balance += balance
        count += 1
    else:
        if current_job:
            print(f"{current_job}\t{total_balance/count:.2f}")
        current_job = job
        total_balance = balance
        count = 1

if current_job:
    print(f"{current_job}\t{total_balance/count:.2f}")
```

## 5.4 MapReduce Jobs Implemented

| Job # | Analysis | Key | Value |
|-------|----------|-----|-------|
| 1 | Avg Balance by Job | job | balance |
| 2 | Housing Loan by Education | education | housing_loan |
| 3 | Monthly Contact Distribution | month | count |
| 4 | Call Duration by Outcome | poutcome | duration |
| 5 | Age Group vs Balance | age_group | balance |

## 5.5 Execution

```bash
# Local testing
cat data/bank.csv | python mapper_avg_balance.py | sort | python reducer_avg_balance.py

# Hadoop cluster
hadoop jar hadoop-streaming.jar \
    -input /data/bank.csv \
    -output /output/avg_balance \
    -mapper "python3 mapper_avg_balance.py" \
    -reducer "python3 reducer_avg_balance.py"
```

---

# 6. Part 3: Hive Analytics

## 6.1 Objective

Perform SQL-based analytics on big data using Apache Hive.

## 6.2 Table Creation

```sql
CREATE DATABASE IF NOT EXISTS banking_analytics;
USE banking_analytics;

CREATE TABLE bank_data (
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

LOAD DATA INPATH '/user/data/bank.csv' INTO TABLE bank_data;
```

## 6.3 Analytical Queries

### Query 1: Subscription Rate by Education
```sql
SELECT
    education,
    COUNT(*) as total,
    SUM(CASE WHEN subscribed = 'yes' THEN 1 ELSE 0 END) as subscribed_count,
    ROUND(SUM(CASE WHEN subscribed = 'yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as subscription_rate
FROM bank_data
GROUP BY education
ORDER BY subscription_rate DESC;
```

**Results:**
| Education | Total | Subscribed | Rate |
|-----------|-------|------------|------|
| tertiary | 1,350 | 193 | 14.30% |
| secondary | 2,306 | 245 | 10.62% |
| unknown | 187 | 19 | 10.16% |
| primary | 678 | 64 | 9.44% |

### Query 2: Monthly Performance
```sql
SELECT
    month,
    COUNT(*) as contacts,
    ROUND(AVG(duration), 2) as avg_duration,
    SUM(CASE WHEN subscribed = 'yes' THEN 1 ELSE 0 END) as conversions,
    ROUND(SUM(CASE WHEN subscribed = 'yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as success_rate
FROM bank_data
GROUP BY month
ORDER BY success_rate DESC;
```

### Query 3: Contact Method Effectiveness
```sql
SELECT
    contact,
    COUNT(*) as total_contacts,
    ROUND(SUM(CASE WHEN subscribed = 'yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as success_rate
FROM bank_data
GROUP BY contact
ORDER BY success_rate DESC;
```

**Results:**
| Contact | Total | Success Rate |
|---------|-------|--------------|
| cellular | 2,896 | 14.36% |
| telephone | 301 | 14.62% |
| unknown | 1,324 | 4.61% |

---

# 7. Part 4: Machine Learning

## 7.1 Objective

Build predictive models to classify whether a customer will subscribe to a term deposit.

## 7.2 ML Pipeline

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│     Data     │ ──► │ Preprocessing│ ──► │   Training   │ ──► │  Evaluation  │
│   Loading    │     │   & Feature  │     │    Models    │     │   & Tuning   │
└──────────────┘     │  Engineering │     └──────────────┘     └──────────────┘
                     └──────────────┘
```

## 7.3 Data Preprocessing

### Handling Outliers (IQR Method)
```python
def cap_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return series.clip(lower=lower, upper=upper)
```

### Feature Encoding
```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

numerical_cols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
categorical_cols = ['job', 'marital', 'education', 'default', 'housing',
                    'loan', 'contact', 'month', 'poutcome']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(drop='first'), categorical_cols)
])
```

### Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# Training: 3,616 samples
# Testing: 905 samples
```

## 7.4 Models Trained

### Model 1: Logistic Regression
```python
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=1000, random_state=42)
```

### Model 2: Decision Tree
```python
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
```

### Model 3: Random Forest
```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
```

### Model 4: Gradient Boosting
```python
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
```

## 7.5 Model Evaluation Results

| Model | AUC-ROC | Accuracy | Precision | Recall | F1-Score |
|-------|---------|----------|-----------|--------|----------|
| **Logistic Regression** | **0.889** | **0.893** | 0.564 | 0.298 | 0.390 |
| Decision Tree | 0.678 | 0.860 | 0.400 | 0.442 | 0.420 |
| Random Forest | 0.883 | 0.887 | 0.519 | 0.260 | 0.346 |
| Gradient Boosting | 0.888 | 0.893 | 0.554 | 0.346 | 0.426 |

## 7.6 Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'classifier__n_estimators': [50, 100, 150],
    'classifier__max_depth': [5, 10, 15],
    'classifier__min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='roc_auc')
grid_search.fit(X_train, y_train)

# Best Parameters:
# n_estimators: 150
# max_depth: 10
# min_samples_leaf: 1
# Tuned AUC-ROC: 0.893
```

## 7.7 Feature Importance

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | duration | 33.0% |
| 2 | poutcome_success | 10.4% |
| 3 | age | 8.8% |
| 4 | balance | 7.9% |
| 5 | month_oct | 3.6% |
| 6 | campaign | 3.2% |
| 7 | poutcome_unknown | 2.8% |
| 8 | contact_unknown | 2.1% |
| 9 | housing_yes | 2.1% |
| 10 | month_mar | 1.7% |

## 7.8 Confusion Matrix (Best Model)

```
                 Predicted
                 No      Yes
Actual  No      777      24
        Yes      73      31

True Positives: 31
True Negatives: 777
False Positives: 24
False Negatives: 73
```

---

# 8. Part 5: Spark Streaming

## 8.1 Objective

Process data in real-time using Spark Structured Streaming.

## 8.2 Streaming Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Data      │ ──► │   Spark     │ ──► │   Window    │ ──► │   Output    │
│   Source    │     │  Streaming  │     │   Agg.      │     │   Sink      │
│ (CSV files) │     │   Engine    │     │ (10s, 1min) │     │  (Console)  │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

## 8.3 Implementation

### Schema Definition
```python
schema = StructType([
    StructField("age", IntegerType(), True),
    StructField("job", StringType(), True),
    StructField("balance", IntegerType(), True),
    StructField("duration", IntegerType(), True),
    StructField("y", StringType(), True),
    StructField("timestamp", TimestampType(), True)
])
```

### Streaming Data Source
```python
streaming_df = spark.readStream \
    .schema(schema) \
    .option("maxFilesPerTrigger", 1) \
    .csv("data/stream_input")
```

### Real-Time Aggregation
```python
job_aggregation = streaming_df \
    .groupBy("job") \
    .agg(
        count("*").alias("transaction_count"),
        avg("balance").alias("avg_balance"),
        avg("duration").alias("avg_duration")
    )
```

### Window Operations
```python
# 10-second tumbling window
window_10s = streaming_df \
    .withWatermark("processing_time", "30 seconds") \
    .groupBy(
        window(col("processing_time"), "10 seconds", "5 seconds"),
        "job"
    ) \
    .agg(count("*").alias("count"))
```

## 8.4 Watermarking for Late Data

```python
streaming_with_watermark = streaming_df \
    .withWatermark("processing_time", "30 seconds")
```

**Benefits:**
- Handles network delays up to 30 seconds
- Prevents unbounded memory growth
- Provides exactly-once processing semantics

---

# 9. Part 6: Data Parallelism

## 9.1 Objective

Demonstrate parallel data processing to achieve performance improvements.

## 9.2 Parallelism Concept

```
Sequential:
CPU: [Task 1][Task 2][Task 3][Task 4] → Total: 40s

Parallel (4 workers):
CPU 1: [Task 1]
CPU 2: [Task 2]     → Total: 10s (4x faster)
CPU 3: [Task 3]
CPU 4: [Task 4]
```

## 9.3 Implementation

### Data Partitioning
```python
# Repartition data for parallel processing
df_parallel = df.repartition(4)
print(f"Partitions: {df_parallel.rdd.getNumPartitions()}")
```

### Partition-Level Processing
```python
def process_partition(iterator):
    results = []
    for row in iterator:
        # Heavy computation
        result = complex_calculation(row['balance'])
        results.append(result)
    return iter(results)

result = df_parallel.rdd.mapPartitions(process_partition).collect()
```

## 9.4 Performance Comparison

| Workers | Processing Time | Speedup |
|---------|-----------------|---------|
| 1 (Sequential) | 8.5s | 1.0x |
| 2 | 4.8s | 1.8x |
| 4 | 2.6s | 3.3x |
| 8 | 1.8s | 4.7x |

---

# 10. Key Findings & Insights

## 10.1 Customer Behavior Insights

### Finding 1: Duration is Critical
- **Longer calls = Higher conversion**
- Subscribed customers: avg 553 seconds (9 min)
- Non-subscribed: avg 226 seconds (4 min)
- **Recommendation**: Train agents to engage customers longer

### Finding 2: Previous Success Matters
- Customers with previous successful campaigns: 65% subscription rate
- First-time customers: 11% subscription rate
- **Recommendation**: Prioritize re-targeting previous subscribers

### Finding 3: Contact Method Impact
- Cellular contacts: 14.4% success rate
- Unknown contacts: 4.6% success rate
- **Recommendation**: Focus on mobile contacts

### Finding 4: Timing is Everything
- Best months: October (46%), December (45%), March (43%)
- Worst months: May (6.6%)
- **Recommendation**: Schedule campaigns in Q4 and early Q1

## 10.2 Demographic Insights

### By Education
| Education | Subscription Rate |
|-----------|------------------|
| Tertiary | 14.3% |
| Secondary | 10.6% |
| Primary | 9.4% |

### By Job
| Job | Avg Balance | Subscription Rate |
|-----|-------------|------------------|
| Retired | €2,319 | 25.3% |
| Student | €1,544 | 31.4% |
| Management | €1,767 | 13.6% |

## 10.3 Model Performance Summary

- **Best Model**: Logistic Regression (AUC-ROC: 0.889)
- **Top Predictor**: Call duration (33% importance)
- **Accuracy**: 89.3%
- **Business Impact**: Can correctly identify ~30% of potential subscribers

---

# 11. Technical Architecture

## 11.1 System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                    │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                             │
│  │  HDFS   │  │  Local  │  │ Stream  │                             │
│  │ Storage │  │  CSV    │  │  Input  │                             │
│  └────┬────┘  └────┬────┘  └────┬────┘                             │
└───────┼────────────┼────────────┼───────────────────────────────────┘
        │            │            │
┌───────┼────────────┼────────────┼───────────────────────────────────┐
│       ▼            ▼            ▼       PROCESSING LAYER            │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐               │
│  │ Hadoop  │  │  Spark  │  │  Spark  │  │  Hive   │               │
│  │MapReduce│  │  Batch  │  │Streaming│  │  SQL    │               │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘               │
└───────┼────────────┼────────────┼────────────┼──────────────────────┘
        │            │            │            │
┌───────┼────────────┼────────────┼────────────┼──────────────────────┐
│       ▼            ▼            ▼            ▼  ANALYTICS LAYER     │
│  ┌─────────────────────────────────────────────────┐               │
│  │              Machine Learning                    │               │
│  │  (Logistic Regression, Random Forest, etc.)     │               │
│  └────────────────────────┬────────────────────────┘               │
└───────────────────────────┼─────────────────────────────────────────┘
                            │
┌───────────────────────────┼─────────────────────────────────────────┐
│                           ▼         PRESENTATION LAYER              │
│  ┌─────────────────────────────────────────────────┐               │
│  │           Streamlit Dashboard                    │               │
│  │    (Interactive Visualizations & Reports)        │               │
│  └─────────────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────────────┘
```

## 11.2 Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| Storage | HDFS, Local FS | Data persistence |
| Batch Processing | Spark, MapReduce | Large-scale analytics |
| Stream Processing | Spark Streaming | Real-time analytics |
| Query Engine | Hive, Spark SQL | SQL-based analytics |
| ML Framework | Scikit-learn | Predictive modeling |
| Visualization | Streamlit, Plotly | Interactive dashboards |

---

# 12. Conclusion

## 12.1 Project Achievements

1. **Comprehensive EDA**: Analyzed 4,521 customer records across 17 features
2. **Distributed Processing**: Implemented MapReduce for scalable batch processing
3. **SQL Analytics**: Built HiveQL queries for business insights
4. **ML Models**: Achieved 89% accuracy in predicting subscriptions
5. **Real-time Processing**: Demonstrated Spark Streaming capabilities
6. **Parallel Computing**: Achieved 4.7x speedup with 8 workers
7. **Interactive Dashboard**: Built comprehensive Streamlit application

## 12.2 Business Recommendations

1. **Optimize Call Duration**: Focus on longer, quality conversations
2. **Target Previous Successes**: Prioritize customers with past positive outcomes
3. **Use Cellular Contacts**: Higher success rate than other methods
4. **Seasonal Planning**: Schedule major campaigns in Q4 and early Q1
5. **Education Targeting**: Focus on tertiary-educated customers
6. **Balance Considerations**: Higher-balance customers more likely to subscribe

## 12.3 Future Improvements

1. Implement real-time prediction API
2. Add A/B testing framework for campaigns
3. Integrate with CRM systems
4. Build automated retraining pipeline
5. Expand to multi-channel marketing analysis

---

## Appendix A: File Structure

```
banking_project/
├── data/
│   └── bank.csv
├── 1_Spark_Data_Processing/
│   ├── spark_data_processing.py
│   └── pandas_data_processing.py
├── 2_Hadoop_MapReduce/
│   ├── mappers/
│   └── reducers/
├── 3_Hive_Analytics/
│   └── hive_queries.sql
├── 4_Spark_ML/
│   ├── spark_ml_model.py
│   └── pandas_ml_model.py
├── 5_Spark_Streaming/
│   ├── spark_streaming_app.py
│   └── stream_generator.py
├── 6_Data_Parallelism/
│   └── data_parallelism.py
├── app.py (Dashboard)
├── requirements.txt
└── README.md
```

## Appendix B: How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run Data Processing
python 1_Spark_Data_Processing/pandas_data_processing.py

# Run ML Model
python 4_Spark_ML/pandas_ml_model.py

# Run Dashboard
streamlit run app.py
```

---

**Document Version**: 1.0
**Last Updated**: December 2024
**Author**: Banking Analytics Team
