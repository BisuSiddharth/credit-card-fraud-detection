# 🏗️ System Architecture – Credit Card Fraud Detection

---

## 📌 1. Overview

This system is designed as a **modular machine learning pipeline** to detect fraudulent credit card transactions.

It processes transaction data, applies preprocessing and feature transformations, and uses trained models to classify transactions as **fraudulent or legitimate**.

---

## 🎯 2. Design Goals

- Handle **highly imbalanced datasets**
- Maintain **high recall** for fraud detection
- Ensure **modularity and scalability**
- Enable **easy experimentation with models**
- Support future **API deployment**

---

## 🧩 3. High-Level Architecture

```plaintext
Raw Data
↓
Data Preprocessing
↓
Feature Engineering
↓
Model Training
↓
Model Evaluation
↓
Prediction System
```

---

## ⚙️ 4. Components

### 🔹 4.1 Data Layer
- Source: Kaggle dataset (CSV)
- Handles data loading and validation

---

### 🔹 4.2 Preprocessing Layer (`preprocessing.py`)
- Missing value handling
- Feature scaling (StandardScaler)
- Handling imbalance:
  - SMOTE / undersampling

---

### 🔹 4.3 Feature Engineering
- Transaction-based features
- Time-based transformations
- Feature normalization

---

### 🔹 4.4 Model Layer (`model.py`)
Implements multiple models:
- Logistic Regression (baseline)
- Random Forest (ensemble)
- Isolation Forest (anomaly detection)

---

### 🔹 4.5 Training Pipeline (`train.py`)
- Train-test split
- Model training
- Model saving (`.pkl` files)

---

### 🔹 4.6 Evaluation Layer
Metrics:
- Precision
- Recall
- F1-score
- ROC-AUC

---

### 🔹 4.7 Prediction Layer (`predict.py`)
- Takes new transaction input
- Applies preprocessing
- Outputs:
  - Fraud label
  - Probability score

---

## 🧠 5. Key Design Decisions

### Why not Accuracy?
Due to class imbalance, accuracy is misleading.

Example:
If 99% transactions are non-fraud → model predicting all “non-fraud” gives 99% accuracy but is useless.

---

### Why Random Forest?
- Handles non-linear relationships
- Works well with imbalanced data
- Robust performance

---

### Why Isolation Forest?
- Effective for anomaly detection
- Identifies rare fraud patterns

---

## 🔄 6. Data Flow

1. Load dataset
2. Clean and preprocess data
3. Scale features
4. Train models
5. Evaluate performance
6. Save best model
7. Use model for predictions

---

## 🚀 7. Future Architecture

- Add FastAPI layer for real-time predictions
- Integrate streaming data (Kafka)
- Deploy using Docker + Cloud

---

## 📌 8. Summary

The system is:
- Modular
- Scalable
- Extendable
- Ready for real-world integration

---