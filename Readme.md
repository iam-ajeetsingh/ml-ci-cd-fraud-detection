# 🚀 End-to-End ML CI/CD Pipeline for Fraud Detection

This project demonstrates a **production-oriented Machine Learning CI/CD pipeline** for a **credit card fraud detection** use case.  
The focus is **delivery, governance, and automation**, not model novelty.

The system supports:
- Automated training and evaluation
- Metric-based promotion gates
- Model versioning and registry
- Production inference via FastAPI
- Safe rollback using registry-driven loading

---

## 🧠 Problem Statement

Credit card fraud detection is a **highly imbalanced binary classification** problem where:
- False negatives are costly (missed fraud)
- False positives degrade customer trust
- Models must be evaluated, promoted, and deployed **safely**

This project simulates how such a system would be built and operated in a real organization.

---

## 📊 Dataset

**Credit Card Fraud Detection Dataset (European cardholders)**

- ~284,000 transactions  
- ~0.17% fraud rate  
- Numerical, privacy-preserving features  
- Single CSV file: `creditcard.csv`  

**Target column**
- `Class` → `1 = Fraud`, `0 = Legitimate`

Dataset location in repo:
data/raw/creditcard.csv

yaml
Copy code

---

## 🎯 Phase 0 — Contract Definition (Design First)

Before writing any code, we locked the system contract.

### Use Case
- Binary classification: Fraud vs Non-Fraud

### Model
- **Logistic Regression**
- Chosen for:
  - Stability
  - Fast training (CI-friendly)
  - Interpretable baseline
  - Reliable probability outputs

### Metrics

**Primary Metric**
- ROC-AUC (threshold-independent ranking quality)

**Guardrail Metric**
- Precision at Recall ≥ 0.80  
  (catch most fraud while limiting false positives)

### Promotion Rule

A model is promoted **only if**:
- ROC-AUC improves by **≥ 1.5%**, AND
- Guardrail metric does **not regress**

This rule is enforced automatically.

---

## 🧱 Repository Structure

ml-ci-cd-fraud/
│
├── data/
│ ├── raw/
│ └── processed/
│
├── training/
│ ├── train.py
│ ├── evaluate.py
│ └── promote.py
│
├── inference/
│ ├── main.py
│ ├── model_loader.py
│ └── requirements.txt
│
├── artifacts/
│ ├── candidate_model.joblib
│ ├── candidate_metrics.json
│ ├── evaluation_result.json
│ └── production/
│  └── model_vX.Y.Z.joblib
│
├── model_registry/
│ ├── metadata.json
│ └── history/
│ └── registry_<timestamp>.json
│
├── requirements.txt
└── README.md


yaml

---

## ⚙️ Phase 1 — Training, Evaluation, Promotion

### 1️⃣ Training (`training/train.py`)

- Loads CSV
- Performs **stratified 70/15/15 split**
- Trains Logistic Regression with `class_weight="balanced"`
- Computes:
  - ROC-AUC
  - Precision @ Recall ≥ 0.80
- Saves:
  - Candidate model artifact
  - Candidate metrics JSON

---

### 2️⃣ Evaluation Gate (`training/evaluate.py`)

- Compares candidate model vs current production model
- Enforces promotion policy
- Outputs:
  - PASS / FAIL
  - Metric deltas
- Uses exit codes (CI-ready)

---

### 3️⃣ Promotion (`training/promote.py`)

If evaluation **passes**:
- Auto-increments model version (`v0.1.0`, `v0.1.1`, …)
- Moves model to `artifacts/production/`
- Updates registry (`model_registry/metadata.json`)
- Archives previous registry state for rollback/audit

---

## 📦 Model Registry

The registry is the **single source of truth**.

Example:
```json
{
  "current_model": "v0.1.0",
  "artifact_path": "artifacts/production/model_v0.1.0.joblib",
  "metrics": {
    "val_roc_auc": 0.87,
    "val_precision_at_min_recall": 0.82
  },
  "created_at": "2025-01-10T18:42:00Z"
}




Benefits:

Deterministic deployments
Simple rollback
Auditability



🚦 Phase 1.8 — Inference Service

FastAPI Inference API

The inference service:

Loads current production model from the registry
Caches model at startup
Exposes health + prediction endpoints


Endpoints

Health
GET /health

Returns:
Model version
Artifact path
Metrics
Status

Prediction

POST /predict

Payload:

json
Copy code
{
  "features": [30 numeric values]
}

Response:


Copy code
{
  "model_version": "v0.1.0",
  "fraud_probability": 0.91,
  "fraud_label": 1,
  "threshold": 0.5
}



🔁 Rollback Strategy

Rollback is registry-driven:

Update metadata.json to point to an older model
Redeploy service
No code changes required

This mirrors real production rollback workflows.



🛠️ Local Run (So Far)


# Train
python training/train.py

# Evaluate
python training/evaluate.py

# Promote (if passed)
python training/promote.py


# Start inference API
uvicorn inference.main:app --reload



🧭 What This Project Demonstrates

ML system design under real constraints
Automated evaluation and promotion gates
Model versioning and audit trails
Separation of training and inference
Production-ready thinking beyond notebooks



🔮 Next Phases (Planned)

Dockerization of inference service
GitHub Actions CI/CD
Blue-green deployment (Cloud Run)
Monitoring & drift detection
Threshold tuning via configuration
Feature schema validation



📌 Key Philosophy

Models are disposable. Pipelines are the product.
This project treats ML models as versioned artifacts governed by policy, not as static files.



## 🐳 Phase 1.9 — Dockerized Inference Service

This project packages the FastAPI inference service into a Docker image that includes:
- `model_registry/metadata.json` (source of truth)
- Production model artifact(s) from `artifacts/production/`

### Build
```bash
docker build -t fraud-ml-api:0.1 .

### Run

docker run --rm -p 8080:8080 fraud-ml-api:0.1




---

## What we do next (Phase 2.0)

Now that you have a container, the next step is **real CI/CD**:

- GitHub Actions pipeline:
  - run tests
  - train → evaluate → promote
  - build docker image
  - (later) push to Artifact Registry / Docker Hub
  - deploy to Cloud Run
- Blue/Green rollout strategy




