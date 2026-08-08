# models

Model training, evaluation, and MLflow registry code for the fraud-scoring
ensemble.

- `excel_reader.py` — loads `data/features.xlsx`, joined against `data/transactions_raw.xlsx` for labels.
- `train_xgboost.py` — trains the supervised `XGBClassifier`, logs params/metrics/model to MLflow, registers it as `pos-fraud-xgboost`.
- `train_isolation_forest.py` — trains the unsupervised `IsolationForest` anomaly detector, registers it as `pos-fraud-isolation-forest`.
- `evaluate.py` — precision/recall/F1/ROC-AUC/PR-AUC on a held-out split.
- `feature_importance.py` — extracts and logs XGBoost feature importances.
- `mlflow_registry.py` — shared tracking URI, experiment name, and register/load helpers.

## Run locally

```bash
pip install -r models/requirements.txt
python models/train_xgboost.py
python models/train_isolation_forest.py
```

Requires an MLflow tracking server reachable at `MLFLOW_TRACKING_URI`
(defaults to `http://localhost:5000`, started via `docker-compose up mlflow`).
