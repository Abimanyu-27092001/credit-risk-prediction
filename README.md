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

##📊 Dataset
Source: UCI Machine Learning Repository
Dataset: German Credit Data
Samples: 1000
Target: Credit default (binary)

---

##📈 Model Performance
Final Test Performance (calibrated):
    ROC-AUC: ~0.78
    F1-score (optimized threshold): ~0.59
    Threshold: Optimized on validation set
These results are realistic and competitive for classical credit risk datasets.

---

##🔍 Explainability & Interpretability
Global Explainability (SHAP)
    Identifies the most influential features driving default risk
    Top drivers include:
        Checking account status
        Savings balance
        Loan duration
        Installment rate
        Credit history
        Credit amount
Local Explainability (SHAP & LIME)
    Detailed explanations for individual customers
    Both high-risk and low-risk profiles analyzed
    Outputs include:
        SHAP force plots (visual)
        Structured JSON reports
        LIME feature contribution lists
    This enables transparent, auditable, and defensible decisions.

---

##🧩 Business Insights
    Applicants with weak checking/savings balances and longer loan durations show higher default risk
    Probability calibration improves decision reliability
    Threshold tuning allows flexible business strategies:
        Higher recall → conservative lending
        Higher precision → aggressive lending
    Borderline cases can be routed for manual review

---

##⚙️ How to Run the Project
    1️⃣ Install dependencies
```text
        pip install -r requirements.txt

    2️⃣ Run the notebook
```text
        credit_risk_prediction.ipynb

---

##🧪 Technologies Used
    Python
    Pandas, NumPy
    Scikit-learn
    LightGBM
    Imbalanced-learn (SMOTE)
    SHAP
    LIME
    Matplotlib

---

##🎯 Use Cases
    Credit risk assessment
    Loan underwriting decision support
    Explainable AI (XAI) demonstrations
    Model auditability and compliance
    ML Engineer / ML Intern portfolio project

---

##📌 Author
    Abimanyu
    Machine Learning Engineer (Aspiring)
