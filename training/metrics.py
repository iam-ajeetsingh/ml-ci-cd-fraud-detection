import numpy as np
from sklearn.metrics import precision_recall_curve


def precision_at_min_recall(y_true: np.ndarray, y_proba: np.ndarray, min_recall: float) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    mask = recall >= min_recall
    if not np.any(mask):
        return 0.0
    return float(np.max(precision[mask]))
