import json
import os


def test_registry_file_exists_and_is_valid_json():
    path = "model_registry/metadata.json"
    assert os.path.exists(path), "model_registry/metadata.json missing"

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "current_model" in data
    assert "metrics" in data
