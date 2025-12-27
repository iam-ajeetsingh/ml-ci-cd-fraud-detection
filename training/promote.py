import json
import os
import shutil
from datetime import datetime, timezone


ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "artifacts")
EVAL_RESULT_PATH = os.path.join(ARTIFACT_DIR, "evaluation_result.json")
CANDIDATE_MODEL_PATH = os.path.join(ARTIFACT_DIR, "candidate_model.joblib")
CANDIDATE_METRICS_PATH = os.path.join(ARTIFACT_DIR, "candidate_metrics.json")

PROD_DIR = os.path.join(ARTIFACT_DIR, "production")
REGISTRY_PATH = os.getenv("REGISTRY_PATH", "model_registry/metadata.json")
HISTORY_DIR = os.path.join("model_registry", "history")


def load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def bump_patch(version: str) -> str:
    # version like "v0.1.3" -> "v0.1.4"
    if not version or not version.startswith("v"):
        return "v0.1.0"
    parts = version[1:].split(".")
    if len(parts) != 3:
        return "v0.1.0"
    major, minor, patch = parts
    try:
        patch_i = int(patch) + 1
        return f"v{major}.{minor}.{patch_i}"
    except ValueError:
        return "v0.1.0"


def main():
    os.makedirs(PROD_DIR, exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)

    eval_result = load_json(EVAL_RESULT_PATH)
    if not eval_result:
        raise FileNotFoundError(f"Missing evaluation result: {EVAL_RESULT_PATH}")

    if not eval_result.get("passed", False):
        print("❌ Promotion skipped: evaluation did not pass.")
        print(json.dumps(eval_result, indent=2))
        raise SystemExit(1)

    registry = load_json(REGISTRY_PATH) or {"current_model": None, "metrics": {}, "created_at": None}
    current_version = registry.get("current_model")

    new_version = bump_patch(current_version) if current_version else "v0.1.0"

    # Load candidate metrics (we store val metrics into registry)
    cand_metrics = load_json(CANDIDATE_METRICS_PATH)
    if not cand_metrics:
        raise FileNotFoundError(f"Missing candidate metrics: {CANDIDATE_METRICS_PATH}")

    cand_val = cand_metrics["val"]

    # Save old registry to history (for audit + rollback)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    history_path = os.path.join(HISTORY_DIR, f"registry_{timestamp}.json")
    save_json(history_path, registry)

    # Copy candidate model into production with versioned filename
    prod_model_path = os.path.join(PROD_DIR, f"model_{new_version}.joblib")
    if not os.path.exists(CANDIDATE_MODEL_PATH):
        raise FileNotFoundError(f"Missing candidate model: {CANDIDATE_MODEL_PATH}")

    shutil.copy2(CANDIDATE_MODEL_PATH, prod_model_path)

    # Update registry
    new_registry = {
        "current_model": new_version,
        "artifact_path": prod_model_path.replace("\\", "/"),
        "metrics": {
            "val_roc_auc": float(cand_val["roc_auc"]),
            "val_precision_at_min_recall": float(cand_val["precision_at_min_recall"]),
            "guardrail_min_recall": float(cand_metrics["guardrail"]["min_recall"]),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "promotion_reason": eval_result.get("reason", "Promotion gate passed."),
        "evaluation_summary": {
            "passed": True,
            "delta": eval_result.get("delta"),
            "policy": eval_result.get("policy"),
            "evaluated_at": eval_result.get("evaluated_at"),
        },
    }

    save_json(REGISTRY_PATH, new_registry)

    print("✅ PROMOTED candidate to production!")
    print(f"New version: {new_version}")
    print(f"Production model saved to: {prod_model_path}")
    print(f"Registry updated: {REGISTRY_PATH}")
    print(f"Registry history saved: {history_path}")


if __name__ == "__main__":
    main()
