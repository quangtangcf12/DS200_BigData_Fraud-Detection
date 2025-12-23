# Anomaly Detection in Financial Transactions

## Overview
This project focuses on **detecting anomalous (fraudulent) financial transactions** using machine learning and graph-based deep learning techniques. The main challenge addressed is **extreme class imbalance** in large-scale transaction data.

The project evaluates both traditional machine learning models and **Graph Neural Networks (GNNs)** to improve fraud detection performance.

---

## Objectives
- Analyze large-scale financial transaction data
- Handle extreme class imbalance in fraud detection
- Compare traditional ML, unsupervised learning, and graph-based models
- Evaluate model performance using robust metrics

---

## Dataset
- Real-world financial transaction dataset
- **8.9+ million transactions**
- Fraud rate: approximately **0.15%**
- Highly imbalanced binary classification problem

---

## Data Preprocessing
- Data cleaning and normalization
- Feature engineering
- Handling missing values
- Train–test split with class balance consideration
- Apache Spark used for scalable data processing

---

## Models Implemented

### Traditional Machine Learning
- Logistic Regression
- Random Forest

### Unsupervised Learning
- Autoencoder

### Graph-Based Deep Learning
- Graph Convolutional Network (GCN)
- Graph Attention Network (GAT)
- Graph Isomorphism Network (GIN)

---

## Evaluation Metrics
- AUC-ROC
- Precision
- Recall
- F1-score (Macro)
- Confusion Matrix

---

## Results
| Model | F1-macro | AUC |
|------|----------|-----|
| Logistic Regression | Low | - |
| Random Forest | Moderate | - |
| Autoencoder | Moderate | - |
| GCN | Improved | - |
| GAT | Improved | - |
| **GIN** | **0.52** | **0.75** |

The **GIN model** achieved the best overall performance, demonstrating the effectiveness of graph-based approaches for fraud detection under severe class imbalance.

---

## Key Findings
- Traditional ML models struggle with extreme imbalance
- Graph Neural Networks significantly improve detection performance
- Modeling transaction relationships as graphs provides richer information

---

## Technologies Used
- Python
- Apache Spark
- Scikit-learn
- PyTorch
- PyTorch Geometric
- Pandas, NumPy
