# API Reference — Scoring Service

Base URL: `http://localhost:8000` (see [scoring/README.md](../scoring/README.md)).

## `GET /health`

Returns service liveness and the currently loaded model version.

```json
{
  "status": "ok",
  "model_version": "v1",
  "models_ready": true
}
```

## `POST /score`

Scores a single feature vector and appends the result to
[`data/fraud_scores.xlsx`](../data/fraud_scores.xlsx).

### Request body

```json
{
  "transaction_id": "b3f1c2...",
  "terminal_id": "TRM00042",
  "card_bin": "539983",
  "velocity_1h": 4,
  "geo_jump_km": 12.3,
  "bin_spend_rate": 18500.0,
  "terminal_reversal_count": 0,
  "amount_ngn": 25000.0
}
```

### Response body

```json
{
  "transaction_id": "b3f1c2...",
  "fraud_probability": 0.12,
  "alert_tier": "low",
  "xgboost_score": 0.10,
  "xgboost_decision": false,
  "xgboost_decision_threshold": 0.337,
  "isolation_forest_score": 0.18,
  "model_version": "v1",
  "scored_at": "2026-08-08T10:15:30+00:00"
}
```

`alert_tier` follows the thresholds documented in
[alert_logic.md](alert_logic.md) and is computed from the blended
`fraud_probability`.

`xgboost_score` is the standalone XGBoost model's positive-class
probability (`predict_proba`), not a binary prediction. `xgboost_decision`
is that probability compared against `xgboost_decision_threshold`, the
threshold selected on a held-out validation split at training time and
packaged with the registered model version (see
[models/README.md](../models/README.md)). This is a **separate concept**
from `alert_tier`: `xgboost_decision_threshold` is a model-validation
threshold applied only to the standalone XGBoost probability, while
`alert_tier`'s thresholds are an operational policy applied to the full
blended `fraud_probability`. Changing one does not move the other.
