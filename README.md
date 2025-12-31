---
title: Banking Analytics Dashboard
emoji: 🏦
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.28.0"
app_file: app.py
pinned: false
---

# 🏦 Banking Analytics - Complete Big Data Project

Interactive dashboard showcasing all 6 parts of a Big Data / Distributed Systems project.

## 🎯 What This Project Does

Analyzes a bank marketing campaign to:
- **Understand** customer behavior patterns
- **Predict** who will subscribe to a term deposit
- **Demonstrate** Big Data technologies (Spark, Hadoop, Hive)

## 📊 Project Structure

| Part | Topic | Description |
|------|-------|-------------|
| 1 | Data Processing | EDA with Pandas/Spark |
| 2 | MapReduce | Hadoop distributed processing |
| 3 | Hive Analytics | SQL-based analytics |
| 4 | Machine Learning | Predictive modeling |
| 5 | Spark Streaming | Real-time processing |
| 6 | Data Parallelism | Parallel computing |

## 📊 Dataset

- **Source**: UCI Bank Marketing Dataset
- **Records**: 4,521
- **Features**: 17 columns
- **Target**: Term deposit subscription (yes/no)

## 🤖 Model Performance

| Model | AUC-ROC | Accuracy |
|-------|---------|----------|
| Logistic Regression | 0.89 | 89% |
| Random Forest | 0.88 | 89% |
| Gradient Boosting | 0.89 | 89% |

## 💡 Key Insights

1. **Duration** is the most important predictor (33%)
2. **Previous success** increases conversion by 60%
3. **Cellular** contact has higher success rate
4. **Best months**: October, March, December

## 🛠️ Technologies Used

- Python 3.8+
- Streamlit
- Pandas
- Plotly
- Scikit-learn

---

Made with Streamlit
