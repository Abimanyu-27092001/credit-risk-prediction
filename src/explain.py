import numpy as np
import pandas as pd
import shap


def compute_tree_shap_values(model, X):
    """
    Computes SHAP values for a tree-based binary classifier.
    Returns SHAP array for the positive class.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Binary classification: take positive class
    if isinstance(shap_values, list):
        shap_arr = shap_values[1]
    else:
        shap_arr = shap_values

    return shap_arr


def compute_global_shap_importance(shap_arr, feature_names):
    """
    Computes mean absolute SHAP importance for each feature.
    Returns a DataFrame sorted by importance.
    """
    if shap_arr.shape[1] != len(feature_names):
        raise ValueError(
            "Mismatch between SHAP values and feature names length."
        )

    mean_abs_shap = np.mean(np.abs(shap_arr), axis=0)

    shap_df = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": mean_abs_shap,
        }
    ).sort_values("mean_abs_shap", ascending=False)

    return shap_df

