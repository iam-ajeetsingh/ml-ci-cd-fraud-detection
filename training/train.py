import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.model_selection import train_test_split


RANDOM_SEED = 42
TARGET_COL = "Class"
RAW_CSV_PATH = os.getenv("RAW_CSV_PATH", "data/raw/creditcard.csv")

ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "artifacts")
CANDIDATE_MODEL_PATH = os.path.join(ARTIFACT_DIR, "candidate_model.joblib")
CANDIDATE_METRICS_PATH = os.path.join(ARTIFACT_DIR, "candidate_metrics.json")


def precision_at_min_recall(y_true: np.ndarray, y_proba: np.ndarray, min_recall: float) -> float:
    """
    Returns the best precision achievable with recall >= min_recall by sweeping thresholds.
    If no threshold reaches min_recall, returns 0.0.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve returns precision/recall for thresholds + an extra point.
    mask = recall >= min_recall
    if not np.any(mask):
        return 0.0
    return float(np.max(precision[mask]))


def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    df = pd.read_csv(RAW_CSV_PATH)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COL}' not found in CSV.")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    # 70/15/15 stratified split
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=RANDOM_SEED
    )

    # Baseline model: Logistic Regression (fast + stable)
    model = LogisticRegression(
        max_iter=2000,
        solver="liblinear",
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    model.fit(X_train, y_train)

    # Probabilities for metrics + thresholding
    val_proba = model.predict_proba(X_val)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]

    min_recall = float(os.getenv("MIN_RECALL_GUARDRAIL", "0.80"))

    metrics = {
        "dataset": "creditcardfraud",
        "target": TARGET_COL,
        "model_type": "LogisticRegression(class_weight=balanced)",
        "split": {"train": 0.70, "val": 0.15, "test": 0.15},
        "guardrail": {"min_recall": min_recall},
        "val": {
            "roc_auc": float(roc_auc_score(y_val, val_proba)),
            "precision_at_min_recall": precision_at_min_recall(y_val.to_numpy(), val_proba, min_recall),
            "fraud_rate": float(y_val.mean()),
        },
        "test": {
            "roc_auc": float(roc_auc_score(y_test, test_proba)),
            "precision_at_min_recall": precision_at_min_recall(y_test.to_numpy(), test_proba, min_recall),
            "fraud_rate": float(y_test.mean()),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    # Save model + metrics
    joblib.dump(model, CANDIDATE_MODEL_PATH)
    with open(CANDIDATE_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("✅ Training complete")
    print(f"Saved model:   {CANDIDATE_MODEL_PATH}")
    print(f"Saved metrics: {CANDIDATE_METRICS_PATH}")
    print("\nCandidate VAL metrics:")
    print(json.dumps(metrics["val"], indent=2))


if __name__ == "__main__":
    main()
