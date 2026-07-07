import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

def calculate_evaluation_metrics(y_true, y_pred, y_prob, durations=None):
    """Calculate evaluation metrics."""

    metrics = {
        "completion_rate": np.mean(y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "path_efficiency": np.mean(durations) if durations else 0.0
    }

    try:
        metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["auc_roc"] = 0.5

    return metrics


def display_metrics_summary(fold, metrics):
    """Display evaluation results."""

    print(f"\n--- Fold {fold} Results ---")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
