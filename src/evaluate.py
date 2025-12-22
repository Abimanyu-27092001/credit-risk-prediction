from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score


def evaluate_binary_classifier(model, X, y, threshold=0.5):
    """
    Evaluate a binary classification model at a given probability threshold.
    Returns metrics dict and predicted probabilities.
    """
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)

    metrics = {
        "auc": roc_auc_score(y, probs),
        "f1": f1_score(y, preds),
        "precision": precision_score(y, preds),
        "recall": recall_score(y, preds),
    }

    return metrics, probs


