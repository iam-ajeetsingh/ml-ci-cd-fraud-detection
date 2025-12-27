import json
import os
from functools import lru_cache

import joblib


REGISTRY_PATH = os.getenv("REGISTRY_PATH", "model_registry/metadata.json")


class RegistryError(Exception):
    pass


def _load_registry() -> dict:
    if not os.path.exists(REGISTRY_PATH):
        raise RegistryError(f"Registry file not found at {REGISTRY_PATH}")

    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        registry = json.load(f)

    if not registry.get("current_model") or not registry.get("artifact_path"):
        raise RegistryError("Registry missing current_model or artifact_path. Did you run promote.py?")

    return registry


@lru_cache(maxsize=1)
def load_production_model():
    """
    Loads the current production model defined in model_registry/metadata.json.
    Cached so it loads once per process.
    """
    registry = _load_registry()
    artifact_path = registry["artifact_path"]

    if not os.path.exists(artifact_path):
        raise RegistryError(f"Model artifact not found at {artifact_path}")

    model = joblib.load(artifact_path)
    return model, registry


def get_production_info() -> dict:
    """
    Returns the registry info for health checks (model version, metrics, etc.).
    """
    _, registry = load_production_model()
    return {
        "current_model": registry.get("current_model"),
        "artifact_path": registry.get("artifact_path"),
        "metrics": registry.get("metrics", {}),
        "created_at": registry.get("created_at"),
    }
