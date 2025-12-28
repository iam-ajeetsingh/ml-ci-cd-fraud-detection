import json
import os
from datetime import datetime, timezone


ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "artifacts")
CANDIDATE_METRICS_PATH = os.path.join(ARTIFACT_DIR, "candidate_metrics.json")

REGISTRY_PATH = os.getenv("REGISTRY_PATH", "model_registry/metadata.json")
EVAL_RESULT_PATH = os.path.join(ARTIFACT_DIR, "evaluation_result.json")

# Promotion policy (Phase 0 contract)
MIN_AUC_IMPROVEMENT = float(os.getenv("MIN_AUC_IMPROVEMENT", "0.015"))  # 1.5% absolute AUC
MIN_RECALL_GUARDRAIL = float(os.getenv("MIN_RECALL_GUARDRAIL", "0.80"))

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def main():
    candidate = load_json(CANDIDATE_METRICS_PATH)
    if not candidate:
        raise FileNotFoundError(f"Candidate metrics not found at: {CANDIDATE_METRICS_PATH}")

    registry = load_json(REGISTRY_PATH) or {"current_model": None, "metrics": {}, "created_at": None}

    cand_auc = candidate["val"]["roc_auc"]
    cand_p_at_r = candidate["val"]["precision_at_min_recall"]

    prod_metrics = registry.get("metrics", {}) or {}
    prod_auc = prod_metrics.get("val_roc_auc", None)
    prod_p_at_r = prod_metrics.get("val_precision_at_min_recall", None)

    # First model: auto-approve if guardrail meets minimum sanity
    if registry.get("current_model") is None or prod_auc is None:
        passed = (cand_p_at_r > 0.0)  # sanity: can reach recall threshold at all
        reason = "No current model in registry; bootstrapping baseline." if passed else "Candidate failed guardrail sanity."
        result = {
            "passed": passed,
            "reason": reason,
            "policy": {
                "min_auc_improvement": MIN_AUC_IMPROVEMENT,
                "min_recall_guardrail": MIN_RECALL_GUARDRAIL,
            },
            "candidate_val": {"roc_auc": cand_auc, "precision_at_min_recall": cand_p_at_r},
            "production_val": None,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }
        save_json(EVAL_RESULT_PATH, result)
        print(json.dumps(result, indent=2))
        # Exit codes: 0 pass, 1 fail (useful for CI)
        raise SystemExit(0 if passed else 1)

    # Normal promotion: must improve AUC by >= threshold AND not regress guardrail
    auc_improvement = cand_auc - float(prod_auc)
    guardrail_ok = (prod_p_at_r is None) or (cand_p_at_r >= float(prod_p_at_r))

    passed = (auc_improvement >= MIN_AUC_IMPROVEMENT) and guardrail_ok

    result = {
        "passed": passed,
        "policy": {
            "min_auc_improvement": MIN_AUC_IMPROVEMENT,
            "min_recall_guardrail": MIN_RECALL_GUARDRAIL,
            "guardrail_non_regression": True,
        },
        "candidate_val": {"roc_auc": cand_auc, "precision_at_min_recall": cand_p_at_r},
        "production_val": {"roc_auc": float(prod_auc), "precision_at_min_recall": float(prod_p_at_r)},
        "delta": {"roc_auc": auc_improvement, "precision_at_min_recall": (cand_p_at_r - float(prod_p_at_r))},
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }

    save_json(EVAL_RESULT_PATH, result)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
