"""
Entry point to run the full credit risk pipeline.
"""

import os
import joblib
import numpy as np

from sklearn.model_selection import train_test_split

from src.data import load_german_credit_data, clean_german_credit_data
from src.preprocessing import get_feature_columns, build_preprocessor
from src.train import apply_smote, train_lightgbm
from src.evaluate import evaluate_binary_classifier
from src.explain import compute_tree_shap_values, compute_global_shap_importance

# Paths
ARTIFACTS_DIR = "artifacts"
REPORTS_DIR = "reports"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

RND = 42


def main():
    # Load & clean data
    df = load_german_credit_data()
    df = clean_german_credit_data(df)

    X = df.drop(columns=["default"])
    y = df["default"]

    num_cols, cat_cols = get_feature_columns(df, target_col="default")

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y,
        random_state=RND,
    )

    # Preprocessing
    preprocessor = build_preprocessor(num_cols, cat_cols)
    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p = preprocessor.transform(X_test)

    # ✅ Feature names AFTER preprocessing
    feature_names = list(num_cols) + list(
        preprocessor.named_transformers_["cat"]
        .named_steps["ohe"]
        .get_feature_names_out(cat_cols)
    )

    # Handle class imbalance
    X_res, y_res = apply_smote(X_train_p, y_train, random_state=RND)

    # Train model
    model, params = train_lightgbm(X_res, y_res, random_state=RND)
    joblib.dump(model, f"{ARTIFACTS_DIR}/best_model_lgb.joblib")

    # Evaluate
    metrics, _ = evaluate_binary_classifier(model, X_test_p, y_test)
    print("Test metrics:", metrics)

    # SHAP (global)
    shap_vals = compute_tree_shap_values(model, X_train_p)
    shap_df = compute_global_shap_importance(shap_vals, feature_names)
    shap_df.to_csv(f"{REPORTS_DIR}/global_shap_all.csv", index=False)

    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()

