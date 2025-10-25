# 🔋 Battery Cell Quality Prediction Project (In-Progress)
 
**Goal:** Predict and explain battery cell quality outcomes (Pass/Fail) using process parameters and electrical test data — emulating the data analytics workflows used in large-scale cell manufacturing.

---

## 🚀 Overview

This project demonstrates how data-driven methods can **quantify product quality**, **detect anomalies**, and **link multiple cell metrics together** using real-world machine learning workflows.

The notebook builds a complete **ML pipeline** for manufacturing analytics:
1. **Data ingestion** (NASA cycling data)
2. **Feature engineering** from time-series and process metrics
3. **Dimensionality reduction** with Principle Component Analysis
4. **Classification** using Random Forests
5. **Explainability & visualization** (feature importance, yield drift, confusion matrix)
6. **Scoring pipeline** for new cells

---

## 🧠 Motivation

In modern cell manufacturing, quality control relies on a variety of correlated process and test parameters.  
This project replicates that environment by predicting whether a cell passes or fails end-of-line quality control using early-stage data.

By developing and interpreting the model, we can:
- Identify key drivers of cell failures (e.g., DCIR, leakage current)
- Detect abnormal process drifts across production batches
- Support **data-informed manufacturing decisions**

---

## 🧰 Tech Stack

| Category | Tools / Libraries |
|-----------|-------------------|
| **Language** | Python |
| **Data Analysis** | pandas, numpy, scipy |
| **Machine Learning** | scikit-learn (PCA, RandomForest, SVM), SHAP |
| **Visualization** | matplotlib |
| **Versioning & Deployment** | Git |

---

## 📊 Data Sources

### 1. Synthetic Data (Only for prototyping purposes)
- 12,000 simulated cells across 12 production batches.
- Includes sample process parameters (coating thickness, drying temperature, calender pressure, slurry viscosity) and EOL metrics (DCIR, capacity, leakage, voltage drift).
- Enables immediate model training without access to proprietary manufacturing data.

### 2. Real Data 
- NASA Battery Aging Dataset sourced from [Kaggle](https://www.kaggle.com/datasets/patrickfleith/nasa-battery-dataset).
- Includes time-series of voltage, current, temperature, and capacity across charge/discharge cycles.
- Label defined by **capacity retention at target cycle** (e.g., 90% retention → Pass).

You can toggle between synthetic and real datasets by changing the `data_source` flag in the notebook.

---

## 🧩 Model Pipeline

```text
Raw Data → Cleaning → Feature Engineering → PCA → RandomForest → Evaluation → Explainability → Visualization
