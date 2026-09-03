from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    accuracy_score,
    precision_score,
    recall_score,
)


def evaluate_binary_classifier(y_true, y_pred_proba, threshold=0.5):

    y_pred = (y_pred_proba >= threshold).astype(int)

    return {
        "roc_auc": roc_auc_score(y_true, y_pred_proba),
        "log_loss": log_loss(y_true, y_pred_proba),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
    }