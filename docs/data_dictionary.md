# Data Dictionary

All persistent data lives in Git-tracked `.xlsx` workbooks under
[`data/`](../data/), read and written via the shared `openpyxl` +
`filelock` pattern described in the root README.

## `data/transactions_raw.xlsx`

| Column | Type | Description |
|---|---|---|
| transaction_id | string (UUID) | Unique transaction identifier |
| terminal_id | string | Format `TRM00001`-`TRM00500` |
| merchant_id | string | Format `MRC0001`-`MRC0200` |
| card_bin | string | 6-digit card bureau identification number |
| amount_ngn | float | Transaction amount in Nigerian Naira |
| state | string | Nigerian state, e.g. `Lagos` |
| lga | string | Local Government Area |
| latitude | float | Terminal latitude |
| longitude | float | Terminal longitude |
| timestamp | string (ISO 8601) | Transaction time, UTC |
| is_fraud | boolean | Ground-truth fraud label |
| fraud_type | string | One of `cloned_card`, `agent_collusion`, `fake_reversal`, or empty |

## `data/features.xlsx`

| Column | Type | Description |
|---|---|---|
| transaction_id | string (UUID) | Foreign key to `transactions_raw.xlsx` |
| terminal_id | string | Terminal identifier |
| card_bin | string | Card BIN |
| velocity_1h | float | Transaction count on this terminal in the trailing 1 hour |
| geo_jump_km | float | Haversine distance (km) from this terminal's previous transaction |
| bin_spend_rate | float | Rolling average spend for this card BIN over the trailing 1 hour |
| terminal_reversal_count | float | Count of `fake_reversal`-flagged transactions on this terminal in the trailing 24 hours |
| timestamp | string (ISO 8601) | Feature computation time |

## `data/fraud_scores.xlsx`

| Column | Type | Description |
|---|---|---|
| transaction_id | string (UUID) | Foreign key to `transactions_raw.xlsx` |
| fraud_probability | float | Blended model output, 0-1 |
| alert_tier | string | One of `low`, `medium`, `high`, `critical` |
| xgboost_score | float | Raw XGBoost predicted probability |
| isolation_forest_score | float | Normalized Isolation Forest anomaly score, 0-1 |
| model_version | string | Registered model version tag |
| scored_at | string (ISO 8601) | Scoring time, UTC |

## `data/alerts.xlsx`

| Column | Type | Description |
|---|---|---|
| alert_id | string (UUID) | Unique alert identifier |
| transaction_id | string (UUID) | Foreign key to `transactions_raw.xlsx` |
| terminal_id | string | Terminal identifier |
| merchant_id | string | Merchant identifier |
| card_bin | string | Card BIN |
| fraud_probability | float | Fraud probability at alert time |
| alert_tier | string | `medium`, `high`, or `critical` (see [alert_logic.md](alert_logic.md)) |
| state | string | Nigerian state |
| created_at | string (ISO 8601) | Alert creation time, UTC |

## `data/reference/terminals.xlsx`

| Column | Type | Description |
|---|---|---|
| terminal_id | string | Format `TRM00001`-`TRM00500` |
| terminal_name | string | Human-readable terminal name |
| merchant_id | string | Owning merchant |
| state | string | Nigerian state |
| lga | string | Local Government Area |
| latitude | float | Terminal latitude |
| longitude | float | Terminal longitude |
| operator | string | POS operator / acquirer |
| is_active | boolean | Whether the terminal is currently active |

## `data/reference/merchants.xlsx`

| Column | Type | Description |
|---|---|---|
| merchant_id | string | Format `MRC0001`-`MRC0200` |
| merchant_name | string | Business name |
| category | string | Merchant category |
| state | string | Nigerian state |
| lga | string | Local Government Area |
| registration_date | string (ISO 8601 date) | Merchant registration date |
| is_flagged | boolean | Whether the merchant is under review |
