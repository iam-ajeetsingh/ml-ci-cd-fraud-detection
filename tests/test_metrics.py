import numpy as np
from training.metrics import precision_at_min_recall


def test_precision_at_min_recall_returns_zero_if_unreachable():
    # y_true has no positives → recall can't reach > 0
    y_true = np.array([0, 0, 0, 0])
    y_proba = np.array([0.1, 0.2, 0.3, 0.4])
    assert precision_at_min_recall(y_true, y_proba, min_recall=0.8) == 0.0


def test_precision_at_min_recall_in_range():
    y_true = np.array([0, 1, 0, 1, 0, 0, 1, 0])
    y_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.05, 0.3, 0.7, 0.4])
    p = precision_at_min_recall(y_true, y_proba, min_recall=0.66)
    assert 0.0 <= p <= 1.0
