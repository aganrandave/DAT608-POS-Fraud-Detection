# models
This documents the Model training, evaluation, and MLflow registry code for the fraud-scoring
ensemble.

- `excel_reader.py` — loads `data/features.xlsx`, joined against `data/transactions_raw.xlsx` for labels.
- `eda.py` — exploratory data analysis (class balance, distributions, correlation, missing values); see [`eda_summary.md`](eda_summary.md) for the current findings and its N=8 sample-size caveat.
- `train_xgboost.py` — trains the supervised `XGBClassifier` with `scale_pos_weight`, 5-fold cross-validation, and full metric/feature-importance logging to MLflow, registers it as `pos-fraud-xgboost`, promotes to Production.
- `train_isolation_forest.py` — trains the unsupervised `IsolationForest` on non-fraud rows only, scores the full dataset against `is_fraud` for a correlation/precision@k check, registers it as `pos-fraud-isolation-forest`, promotes to Production.
- `evaluate.py` — precision/recall/F1/ROC-AUC/PR-AUC on a held-out split.
- `feature_importance.py` — extracts and logs XGBoost feature importances.
- `mlflow_registry.py` — shared tracking URI, experiment name, model-name/stage constants, and register/promote/load helpers.

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
4. `scoring/scorer.py` always loads whatever is currently in the `Production` stage via `load_latest_model()`, so promoting a new version takes effect on the next scoring service restart with no code change needed.
