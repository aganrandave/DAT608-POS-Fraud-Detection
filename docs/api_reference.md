# API Reference — Scoring Service

Base URL: `http://localhost:8000` (see [scoring/README.md](../scoring/README.md)).

## `GET /health`

Returns service liveness and the currently loaded model version.

```json
{
  "status": "ok",
  "model_version": "v1"
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
  "terminal_reversal_count": 0
}
```

### Response body

```json
{
  "transaction_id": "b3f1c2...",
  "fraud_probability": 0.12,
  "alert_tier": "low",
  "xgboost_score": 0.10,
  "isolation_forest_score": 0.18,
  "model_version": "v1",
  "scored_at": "2026-08-08T10:15:30+00:00"
}
```

`alert_tier` follows the thresholds documented in
[alert_logic.md](alert_logic.md).
