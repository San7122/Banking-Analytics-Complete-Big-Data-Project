# Banking Data Analytics with Distributed Computing
## Project Report

---

**Course:** Big Data & Distributed Systems
**Project:** Bank Marketing Campaign Analysis
**Dataset:** UCI Bank Marketing Dataset (4,521 records)
**Date:** December 2024

---

## Abstract

This project demonstrates comprehensive big data analytics using distributed computing technologies. We analyzed a Portuguese bank's marketing campaign data to identify factors influencing term deposit subscriptions and built predictive machine learning models achieving 89% accuracy. The project showcases six key areas: Spark data processing, Hadoop MapReduce, Hive SQL analytics, machine learning, Spark Streaming, and data parallelism.

---

## 1. Introduction

### 1.1 Background
Direct marketing campaigns are essential for financial institutions to promote products. Understanding customer behavior and optimizing campaign strategies can significantly improve conversion rates and reduce costs.

### 1.2 Objectives
1. Perform exploratory data analysis on customer data
2. Demonstrate distributed processing with MapReduce
3. Build SQL analytics with Apache Hive
4. Develop predictive models for subscription prediction
5. Implement real-time processing with Spark Streaming
6. Optimize performance through data parallelism

### 1.3 Dataset Description
- **Source:** UCI Machine Learning Repository
- **Records:** 4,521 customer interactions
- **Features:** 17 attributes (demographic, financial, campaign)
- **Target:** Term deposit subscription (yes/no)
- **Class Distribution:** 88.5% No, 11.5% Yes (imbalanced)

---

## 2. Part 1: Data Processing with Spark/Pandas

### 2.1 Data Loading and Inspection
```python
df = pd.read_csv("data/bank.csv")
print(f"Shape: {df.shape}")  # (4521, 17)
```

### 2.2 Key Statistics
| Metric | Value |
|--------|-------|
| Total Records | 4,521 |
| Features | 17 |
| Missing Values | 0 |
| Duplicate Records | 0 |

### 2.3 Feature Analysis
- **Job Distribution:** Management (21.4%), Blue-collar (20.9%), Technician (16.8%)
- **Age Distribution:** Mean 41 years, range 18-95
- **Balance:** Mean €1,423, highly skewed

### 2.4 Key Findings
1. Retired customers have highest average balance (€2,319)
2. Weak positive correlation between age and balance (0.08)
3. Higher-balance customers show higher subscription rates

---

## 3. Part 2: Hadoop MapReduce

### 3.1 MapReduce Paradigm
```
INPUT → MAP → SHUFFLE & SORT → REDUCE → OUTPUT
```

### 3.2 Implemented Jobs

| Job | Description | Key | Value |
|-----|-------------|-----|-------|
| 1 | Average Balance by Job | job | balance |
| 2 | Housing Loan by Education | education | housing |
| 3 | Monthly Contact Distribution | month | count |
| 4 | Call Duration by Outcome | poutcome | duration |
| 5 | Age Group Analysis | age_group | balance |

### 3.3 Sample Mapper Code
```python
#!/usr/bin/env python3
import sys
for line in sys.stdin:
    fields = line.strip().split(',')
    job = fields[1]
    balance = fields[5]
    print(f"{job}\t{balance}")
```

### 3.4 Results
- Management: avg balance €1,767
- Retired: avg balance €2,319
- Blue-collar: avg balance €884

---

## 4. Part 3: Hive Analytics

### 4.1 Table Schema
```sql
CREATE TABLE bank_data (
    age INT, job STRING, marital STRING,
    education STRING, default_status STRING,
    balance INT, housing STRING, loan STRING,
    contact STRING, day INT, month STRING,
    duration INT, campaign INT, pdays INT,
    previous INT, poutcome STRING, subscribed STRING
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ',';
```

### 4.2 Analytical Query Results

**Subscription Rate by Education:**
| Education | Total | Rate |
|-----------|-------|------|
| Tertiary | 1,350 | 14.3% |
| Secondary | 2,306 | 10.6% |
| Primary | 678 | 9.4% |

**Contact Method Effectiveness:**
| Contact | Success Rate |
|---------|--------------|
| Cellular | 14.4% |
| Telephone | 14.6% |
| Unknown | 4.6% |

---

## 5. Part 4: Machine Learning

### 5.1 Preprocessing Pipeline
1. **Outlier Handling:** IQR method for numerical features
2. **Feature Scaling:** StandardScaler for numerical columns
3. **Encoding:** OneHotEncoder for categorical columns
4. **Split:** 80% training, 20% testing (stratified)

### 5.2 Models Trained
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

### 5.3 Model Performance

| Model | AUC-ROC | Accuracy | Precision | Recall |
|-------|---------|----------|-----------|--------|
| **Logistic Regression** | **0.889** | **89.3%** | 0.564 | 0.298 |
| Gradient Boosting | 0.888 | 89.3% | 0.554 | 0.346 |
| Random Forest | 0.883 | 88.7% | 0.519 | 0.260 |
| Decision Tree | 0.678 | 86.0% | 0.400 | 0.442 |

### 5.4 Feature Importance (Top 10)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | **duration** | **33.0%** |
| 2 | poutcome_success | 10.4% |
| 3 | age | 8.8% |
| 4 | balance | 7.9% |
| 5 | month_oct | 3.6% |
| 6 | campaign | 3.2% |
| 7 | poutcome_unknown | 2.8% |
| 8 | contact_unknown | 2.1% |
| 9 | housing_yes | 2.1% |
| 10 | month_mar | 1.7% |

### 5.5 Confusion Matrix (Best Model)
```
                 Predicted
                 No      Yes
Actual  No      777      24
        Yes      73      31
```

---

## 6. Part 5: Spark Streaming

### 6.1 Streaming Architecture
```
Data Source → Spark Streaming → Window Aggregation → Output Sink
```

### 6.2 Implementation Features
- **Schema Definition:** Structured streaming with defined types
- **Real-time Aggregation:** Count and average by job type
- **Window Operations:** 10-second tumbling windows
- **Watermarking:** 30-second tolerance for late data

### 6.3 Sample Code
```python
streaming_df = spark.readStream \
    .schema(schema) \
    .option("maxFilesPerTrigger", 1) \
    .csv("data/stream_input")

job_agg = streaming_df.groupBy("job").agg(
    count("*").alias("count"),
    avg("balance").alias("avg_balance")
)
```

---

## 7. Part 6: Data Parallelism

### 7.1 Parallelism Concept
```
Sequential: [T1][T2][T3][T4] = 40 seconds
Parallel:   [T1][T2][T3][T4] = 10 seconds (4x faster)
```

### 7.2 Performance Results

| Workers | Processing Time | Speedup |
|---------|-----------------|---------|
| 1 | 8.5 seconds | 1.0x |
| 2 | 4.8 seconds | 1.8x |
| 4 | 2.6 seconds | 3.3x |
| 8 | 1.8 seconds | **4.7x** |

### 7.3 Implementation
```python
df_parallel = df.repartition(4)
result = df_parallel.rdd.mapPartitions(process_partition).collect()
```

---

## 8. Key Insights & Business Recommendations

### 8.1 Customer Behavior Insights

1. **Duration is Critical (33% importance)**
   - Subscribed customers: avg 553 seconds (9 min)
   - Non-subscribed: avg 226 seconds (4 min)

2. **Previous Success Matters**
   - Previous success: 65% conversion rate
   - First-time: 11% conversion rate

3. **Contact Method Impact**
   - Cellular: 14.4% success
   - Unknown: 4.6% success

4. **Timing is Everything**
   - Best months: October, December, March
   - Worst month: May (6.6%)

### 8.2 Business Recommendations

| Priority | Action | Expected Impact |
|----------|--------|-----------------|
| High | Extend call duration | +33% conversion |
| High | Re-target previous subscribers | +65% success |
| Medium | Use cellular contacts | +14% success |
| Medium | Focus on Q4 & early Q1 | Better timing |
| Low | Target tertiary education | Higher rate |

---

## 9. Technical Architecture

```
┌─────────────────────────────────────────────────┐
│              DATA LAYER                          │
│  HDFS Storage | Local CSV | Stream Input        │
└───────────────────────┬─────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────┐
│            PROCESSING LAYER                      │
│  Hadoop MapReduce | Spark Batch | Spark Stream  │
│  Hive SQL                                        │
└───────────────────────┬─────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────┐
│            ANALYTICS LAYER                       │
│  Machine Learning | Feature Engineering         │
└───────────────────────┬─────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────┐
│           PRESENTATION LAYER                     │
│  Streamlit Dashboard | Interactive Viz          │
└─────────────────────────────────────────────────┘
```

---

## 10. Conclusion

### 10.1 Project Achievements
1. Comprehensive EDA on 4,521 customer records
2. Implemented 5 MapReduce jobs for batch processing
3. Built 5+ Hive SQL analytics queries
4. Achieved 89% accuracy with ML models
5. Demonstrated real-time streaming capabilities
6. Achieved 4.7x speedup with parallel processing
7. Created interactive Streamlit dashboard
8. Deployed on Hugging Face Spaces

### 10.2 Key Takeaway
**Call duration is the most important factor (33% importance).**
Banks should focus on training agents to engage customers in longer, quality conversations rather than making more short calls.

### 10.3 Future Improvements
1. Real-time prediction API
2. A/B testing framework
3. CRM system integration
4. Automated model retraining
5. Multi-channel marketing analysis

---

## References

1. UCI Machine Learning Repository - Bank Marketing Dataset
2. Apache Spark Documentation
3. Apache Hadoop Documentation
4. Apache Hive Documentation
5. Scikit-learn Documentation
6. Streamlit Documentation

---

## Appendix

### A. Technology Stack
| Component | Technology | Version |
|-----------|------------|---------|
| Language | Python | 3.8+ |
| Big Data | Apache Spark | 3.x |
| Batch Processing | Hadoop | 3.x |
| SQL Engine | Apache Hive | 3.x |
| ML Library | Scikit-learn | 1.3+ |
| Visualization | Plotly | 5.15+ |
| Dashboard | Streamlit | 1.28+ |
| Deployment | Hugging Face | - |

### B. File Structure
```
banking_project/
├── data/bank.csv
├── 1_Spark_Data_Processing/
├── 2_Hadoop_MapReduce/
├── 3_Hive_Analytics/
├── 4_Spark_ML/
├── 5_Spark_Streaming/
├── 6_Data_Parallelism/
├── app.py
├── requirements.txt
└── docs/
    ├── PROJECT_DOCUMENTATION.md
    ├── PROJECT_REPORT.md
    ├── PRESENTATION_SLIDES.md
    └── generate_ppt.py
```

### C. How to Run
```bash
# Install dependencies
pip install -r requirements.txt

# Run Dashboard
streamlit run app.py

# Generate PPT
cd docs
pip install python-pptx
python generate_ppt.py
```

---

**End of Report**
