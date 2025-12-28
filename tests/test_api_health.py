from fastapi.testclient import TestClient

import inference.main as main_module


def test_health_endpoint_ok(monkeypatch):
    def fake_get_production_info():
        return {
            "current_model": "v0.0.0-test",
            "artifact_path": "artifacts/production/model_v0.0.0-test.joblib",
            "metrics": {"val_roc_auc": 0.9},
            "created_at": "2025-01-01T00:00:00Z",
        }

    monkeypatch.setattr(main_module, "get_production_info", fake_get_production_info)

    client = TestClient(main_module.app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "current_model" in body
    assert "metrics" in body
