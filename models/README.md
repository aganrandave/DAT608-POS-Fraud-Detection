# models
This documents the Model training, evaluation, and MLflow registry code for the fraud-scoring
ensemble.

- `excel_reader.py` — loads `data/features.xlsx`, joined against `data/transactions_raw.xlsx` for labels.
- `eda.py` — exploratory data analysis (row/fraud counts, fraud-type distribution, class balance, distributions, correlation, missing values); see [`eda_summary.md`](eda_summary.md) for the current findings, computed against the real ~30k-row committed dataset.
- `train_xgboost.py` — trains the supervised `XGBClassifier` with `scale_pos_weight`, picks a decision threshold on a held-out validation split, 5-fold cross-validation, and full metric/feature-importance logging to MLflow; registers it as `pos-fraud-xgboost` with the threshold packaged alongside it (see [Model contract](#model-contract) below), promotes to Production.
- `train_isolation_forest.py` — trains the unsupervised `IsolationForest` on non-fraud rows only, scores the full dataset against `is_fraud` for a correlation/precision@k check, fits reproducible score normalization bounds, registers it as `pos-fraud-isolation-forest` with those bounds packaged alongside it, promotes to Production.
- `evaluate.py` — standalone-XGBoost precision/recall/F1/ROC-AUC/PR-AUC on a held-out split (`evaluate_xgboost`), plus the same metrics for the actual deployed XGBoost + Isolation Forest blended scoring contract (`evaluate_blended_scoring_contract`) — see the module docstring for why these are reported separately and must not be conflated.
- `feature_importance.py` — extracts and logs XGBoost feature importances.
- `mlflow_registry.py` — shared tracking URI, experiment name, model-name/stage constants, sibling-metadata-artifact helpers, and register/promote/load helpers.

## Model contract

`scoring/scorer.py` does not call `.predict()` on either model — see
[docs/api_reference.md](../docs/api_reference.md) for why, and for the
distinction between `xgboost_decision_threshold` (a model-validation
threshold) and `alert_tier`'s operational thresholds (kept deliberately
separate concepts). Each registered model version carries a small JSON
metadata artifact alongside it, in the same MLflow artifact path as the
model itself, loaded via `models:/{name}/{stage}/{filename}`:

| Model | Sibling artifact | Contents |
|---|---|---|
| `pos-fraud-xgboost` | `decision_threshold.json` | The threshold selected on a held-out validation split by `select_threshold()` |
| `pos-fraud-isolation-forest` | `score_bounds.json` | `score_min`/`score_max` of `score_samples()` over the training-time dataset, used to normalize new scores into 0-1 |

`mlflow_registry.load_xgboost_model_and_threshold()` and
`load_isolation_forest_model_and_bounds()` load each model via its native
flavor (`mlflow.xgboost`/`mlflow.sklearn`, not `mlflow.pyfunc`) together
with this metadata, and raise `ModelMetadataError` — a distinct, clearly-named
failure, not a silent default — if a model is registered but its artifact
is missing or unreadable.

## Run locally

```bash
pip install -r models/requirements.txt
python models/train_xgboost.py
python models/train_isolation_forest.py
```

By default this expects an MLflow tracking server reachable at
`MLFLOW_TRACKING_URI` (defaults to `http://localhost:5000`, started via
`docker-compose up mlflow`). For local development without Docker, the
Model Registry works fine against a local SQLite-backed store instead —
the file store alone does *not* support the registry API:

```bash
export MLFLOW_TRACKING_URI="sqlite:///mlruns.db"
python models/train_xgboost.py
python models/train_isolation_forest.py
```

`mlruns.db` and `mlruns/` are gitignored — they're local run artifacts,
not something to commit.

## Retraining and promoting a new version

1. Update the training script (hyperparameters, feature list, etc.) and re-run it. `register_model()` in `mlflow_registry.py` registers a new model version **and promotes it straight to Production by default** (`archive_existing_versions=True`, so the prior Production version is automatically archived, not deleted).
2. If you want to register without immediately promoting (e.g. to compare against the current Production model first), call `register_model(..., promote=False)` and promote manually later via `MlflowClient().transition_model_version_stage(name=..., version=..., stage="Production")`.
3. Model names and the target stage are constants in `mlflow_registry.py` (`MODEL_NAME_XGBOOST`, `MODEL_NAME_ISOLATION_FOREST`, `PRODUCTION_STAGE`) — update those, not string literals, if either registered model is ever renamed.
4. `scoring/scorer.py` always loads whatever is currently in the `Production` stage via `mlflow_registry.py`'s loaders, so promoting a new version takes effect on the next scoring service restart with no code change needed — **provided** the new version's sibling metadata artifact (decision threshold / score bounds) was logged in the same run; see [Model contract](#model-contract) above.

## Model results

Reproduced by running `train_xgboost.py` and `train_isolation_forest.py`
locally against `MLFLOW_TRACKING_URI=sqlite:///mlruns.db` (see "Run
locally" above) on **commit `54ac207`**, the same ~30,008-row committed
dataset documented in [`eda_summary.md`](eda_summary.md) (1,146 fraud rows,
3.819% prevalence). Every number below came directly off that run's MLflow
metrics — none of it is copied from a docstring or an older run.

### XGBoost (standalone)

| Metric | Default 0.5 threshold | Tuned threshold (0.944, selected on held-out validation split) |
|---|---|---|
| Precision | 0.559 | **0.854** |
| Recall | 0.865 | **0.716** |
| F1 | 0.679 | **0.779** |
| ROC-AUC | — | 0.952 |
| Average precision (PR-AUC) | — | 0.825 |

5-fold cross-validation (real data only, default threshold): precision
0.516, recall 0.860, F1 0.645, ROC-AUC 0.955 (mean across folds; all 5
folds had both classes present).

These are **standalone XGBoost metrics** — the model scored in isolation,
at the stated threshold, on its own held-out test split. They describe
`xgboost_score`/`xgboost_decision` as reported by `/score`, not the final
`fraud_probability`/`alert_tier` the deployed service actually returns.

### XGBoost feature importance (real, from the fitted model — not correlation)

| Feature | Importance |
|---|---|
| velocity_1h | 0.319 |
| geo_jump_km | 0.297 |
| terminal_reversal_count | 0.279 |
| amount_ngn | 0.080 |
| amount_vs_bin_avg_ratio | 0.013 |
| bin_spend_rate | 0.012 |

Worth contrasting with [`eda_summary.md`](eda_summary.md)'s correlation
table: `geo_jump_km` correlates with `is_fraud` more weakly (r=0.230,
ranked 3rd of 6) than `velocity_1h` (r=0.522) or `terminal_reversal_count`
(r=0.316, ranked 2nd), yet XGBoost's actual feature importance ranks it
*above* `terminal_reversal_count` — the model found more signal in it,
interacting with other features, than its standalone correlation
suggested. Conversely `bin_spend_rate` (r=0.182) gets almost no weight
(0.012) despite a non-trivial standalone correlation. This is exactly why
correlation and feature importance are reported as two separate things,
not interchangeable summaries of "what matters."

### Isolation Forest (unsupervised)

| Metric | Value |
|---|---|
| Anomaly-score / `is_fraud` correlation | −0.44 |
| Precision@1146 (top-scoring anomalies, k = true fraud count) | 0.42 |

Trained on the 28,862 non-fraud rows only, evaluated by scoring the full
30,008-row dataset. See `train_isolation_forest.py`'s docstring for why
these numbers are slightly weaker than an earlier 4-feature version
(correlation −0.50 → −0.44) — adding `amount_ngn`/`amount_vs_bin_avg_ratio`
helped XGBoost but diluted Isolation Forest's distance-based signal.

### Real-only vs. CTGAN-augmented

| | Real-only (default) | `USE_SYNTHETIC_AUGMENTATION=true` |
|---|---|---|
| Held-out F1 (XGBoost, tuned threshold) | **0.78** | 0.73 (historical — see note) |
| Held-out precision / recall | 0.85 / 0.72 | 0.83 / 0.66 (historical) |

The augmented-run numbers are **historical**, reported in
`train_xgboost.py`'s docstring from an earlier comparison run rather than
reproduced fresh here — re-running with `USE_SYNTHETIC_AUGMENTATION=true`
requires `validate_synthetic_features.py`'s gate to have approved a
synthetic set first. Both runs were scored on the identical real-only test
split (synthetic rows are added only to the train/val pool — see
`train_xgboost.train()`'s docstring), so the comparison is fair: synthetic
augmentation measurably hurt this model, which is why it stays off by
default.

### Blended scoring contract (deployed `/score` behavior)

**Leakage audit.** An earlier version of this evaluation reused the
PRODUCTION Isolation Forest model (and its score_min/score_max bounds)
directly — a real methodological bug: that model is fit on every non-fraud
row with no train/test split at all (see `train_isolation_forest.py`), so
the "held-out" test rows had already been seen during its own fitting and
had already contributed to its normalization bounds. `evaluate.py` now
fits a fresh, **evaluation-only** Isolation Forest on the train partition
alone (`_fit_evaluation_only_isolation_forest`), with the production
model's exact hyperparameters, and asserts the train/test index sets are
disjoint before scoring anything (`tests/unit/test_evaluate.py` covers
this with both the real committed data and a synthetic adversarial case).
The corrected numbers below differ only slightly from the leaky version in
this instance (precision 0.237 → 0.240) — the leakage was real and worth
fixing on principle regardless of how much any one run's numbers moved by.

Measured via `evaluate.py`'s `evaluate_blended_scoring_contract()` and
`tier_operating_points()`, on the same held-out split as the standalone
numbers above, at the deployed 0.7 XGBoost / 0.3 Isolation Forest blend
weight (`scoring/scorer.py`'s `XGBOOST_WEIGHT`, a fixed constant, not
tuned on any split, so it introduces no leakage risk of its own):

| Operating point | Precision | Recall | F1 | FPR | Alerted | % of test set |
|---|---|---|---|---|---|---|
| `medium` and above (blended ≥ 0.30) | 0.240 | 0.900 | 0.379 | 11.31% | 859 | 14.31% |
| `high` and above (blended ≥ 0.60) | 0.585 | 0.856 | 0.695 | 2.41% | 335 | 5.58% |
| `critical` (blended ≥ 0.85) | 0.893 | 0.511 | 0.650 | 0.24% | 131 | 2.18% |
| XGBoost standalone (tuned threshold 0.944) | 0.854 | 0.716 | 0.779 | 0.49% | 192 | 3.20% |

Confusion counts (test set: 6,002 rows, 229 fraud) — `medium`: TP 206 / FP
653 / FN 23 / TN 5,120. `high`: TP 196 / FP 139 / FN 33 / TN 5,634.
`critical`: TP 117 / FP 14 / FN 112 / TN 5,759. XGBoost standalone: TP 164
/ FP 28 / FN 65 / TN 5,745.

**The 0.30/0.60/0.85 tier thresholds are round-number demonstration
defaults, not empirically selected.** Nothing in this repository's
history documents them being derived from an investigator-capacity model,
a false-positive cost estimate, or any other business rationale — they
were carried over unchanged from the original ticket. The resulting
precision/recall tradeoff at the deployed `medium` threshold (0.240
precision, catching 90% of fraud at the cost of 653 false alerts out of
859 total) is therefore a **measured consequence** of an unvalidated
threshold choice, not a tradeoff chosen "by design" for a stated business
reason. ROC-AUC barely moves between standalone and blended (0.952 →
0.951), confirming this is a threshold/policy effect on top of
essentially unchanged ranking quality, not the Isolation Forest component
degrading the model's underlying discrimination. These thresholds were
not re-selected on this test set to produce this table — they are
evaluated as-is, exactly as `alerts/alert_tiers.py` already defines them.

Whether any of these three operating points is the right one for a real
deployment is an operational policy question this evaluation does not
answer — see [Scope, data, and limitations](../README.md#scope-data-and-limitations).

### Conservative validation: terminal-grouped holdout

The row-random split above stratifies by `is_fraud` but splits rows
independently — the generated data contains terminal-local velocity
bursts and repeated card/terminal/BIN patterns (see
`producer/fraud_scenarios.py`), so correlated records can end up on both
sides of a row-random split, which can overstate how well the model
generalizes to genuinely new activity. `evaluate.py`'s
`evaluate_xgboost_terminal_grouped()` instead splits by unique
`terminal_id` (`sklearn.model_selection.GroupShuffleSplit`, asserted
disjoint in `tests/unit/test_evaluate.py`), trains a fresh XGBoost model
and evaluation-only Isolation Forest on only the train-partition
terminals, and evaluates on terminals never seen in training at all — not
just different rows.

Result (test set: 5,958 rows across held-out terminals, 217 fraud;
XGBoost standalone at its **default 0.5 threshold**, not a newly-tuned
one — a new threshold was deliberately not selected on this split, to
avoid repeating the exact leakage/selection-on-test-data risk this
evaluation exists to guard against):

| | XGBoost standalone (default 0.5) | Blended `is_alertable` |
|---|---|---|
| Precision | 0.505 | 0.279 |
| Recall | 0.876 | 0.917 |
| F1 | 0.641 | 0.427 |
| ROC-AUC | **0.962** | **0.966** |

**No evidence of correlation-driven overstatement.** ROC-AUC under this
stricter, terminal-held-out split (0.962 / 0.966) is slightly *higher*
than the row-random split's (0.952 / 0.951), not lower — if row-random
splitting were letting correlated terminal-local records leak an
artificial advantage, this conservative split would be expected to show
materially *worse* ranking quality, and it doesn't. Precision/recall
aren't directly comparable between the two tables (different thresholds,
different split sizes), but ROC-AUC is threshold-independent and is the
right number to compare here.

Report both: the row-random numbers above describe performance on unseen
*transactions*; the terminal-grouped numbers describe performance on
unseen *terminals*, a stricter and arguably more realistic bar for this
system's actual deployment unit (fraud alerts are triaged per terminal).
