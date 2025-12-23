# Credit Risk Prediction with Explainable Machine Learning

An end-to-end, production-style machine learning project for **credit default prediction** using the **UCI German Credit dataset**.  
The project focuses not only on predictive performance, but also on **model interpretability, transparency, and business usability**, which are critical in financial risk and regulatory environments.

---

## 🚀 Project Overview

This project builds a **binary classification system** to predict whether a loan applicant will **default on credit**.

Unlike simple ML demos, this project is designed to reflect **real-world ML engineering practices**, including:
- clean modular code
- reproducible experiments
- calibrated probabilities
- threshold optimization
- explainable AI (XAI)

It is well suited for **ML Engineer and ML Internship roles**.

---

## 🧠 Key Highlights

- End-to-end ML pipeline (data → preprocessing → model → evaluation → explainability)
- Robust preprocessing using `ColumnTransformer`
- Class imbalance handling using **SMOTE**
- **LightGBM** model with hyperparameter tuning (cross-validation)
- Probability calibration using **Platt scaling**
- Threshold optimization based on **F1-score**
- **Global explainability** using SHAP
- **Local explainability** using SHAP force plots
- Instance-level explanations using **LIME**
- Business-oriented executive summary and insights
- Clean, modular, production-ready code structure

---

## 📂 Repository Structure

```text
credit-risk-prediction/
│
├── artifacts/                      # Trained models & thresholds
│   ├── best_model_lgb.joblib
│   ├── calibrated_model.joblib
│   └── chosen_threshold.txt
│
├── figures/                        # Evaluation & explainability plots
│   ├── roc.png
│   ├── pr.png
│   ├── confusion_matrix.png
│   ├── shap_summary.png
│   └── force_plot_*.png
│
├── reports/                        # Structured explainability outputs
│   ├── global_shap_all.csv
│   ├── local_shap_reports.json
│   ├── local_lime_reports.json
│   └── report.md
│
├── src/                            # Modular ML pipeline code
│   ├── data.py                     # Data loading & cleaning
│   ├── preprocessing.py            # Feature engineering & pipelines
│   ├── train.py                    # Model training & tuning
│   ├── evaluate.py                 # Evaluation utilities
│   └── explain.py                  # SHAP utilities
│
├── credit_risk_prediction.ipynb    # Orchestration notebook
├── requirements.txt
└── README.md
```
---

## 📊 Model Performance

**Final Test Performance (Calibrated):**
- ROC-AUC: ~0.78  
- F1-score (optimized threshold): ~0.59  
- Threshold: Optimized on validation set

---

## 🔍 Explainability

- Global feature importance using SHAP
- Local explanations using SHAP force plots
- Instance-level explanations using LIME
- Structured JSON outputs for auditability

---

## 🔍 Interactive Demo

A minimal Streamlit application is included to demonstrate
threshold-based credit risk decisioning using the trained model.

> Note: The demo is for educational purposes only and does not
represent a real credit decision system.

---

## 🧪 Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- LightGBM
- SHAP, LIME
- Imbalanced-learn (SMOTE)
- Streamlit

---

## 🎯 Author

**Abimanyu M**  
Machine Learning Engineer
