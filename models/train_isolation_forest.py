"""Trains the unsupervised Isolation Forest anomaly detector and logs it to MLflow.

Training data as of this run: 30,008 real rows (bulk-generated via
producer/bulk_generate.py + pipeline/batch_feature_engineering.py,
SCRUM-20/23), ~1.9% fraud. Isolation Forest is fit on non-fraud rows only.
A CTGAN-based synthetic augmentation of this data was evaluated
(models/synthetic_augmentation.py + validate_synthetic_features.py) but
did NOT pass the five-level validation gate (TSTR/TRTR AUC gap 0.11,
0/4 features KS-aligned) - USE_SYNTHETIC_AUGMENTATION has no effect until
a synthetic set actually passes that gate.
"""
import os

import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import IsolationForest  # noqa: E402

from excel_reader import load_augmented_training_frame  # noqa: E402
from mlflow_registry import EXPERIMENT_NAME, MODEL_NAME_ISOLATION_FOREST, register_model  # noqa: E402

FEATURE_COLUMNS = [
    "velocity_1h",
    "geo_jump_km",
    "bin_spend_rate",
    "terminal_reversal_count",
]

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "eda_output")


def load_dataset() -> pd.DataFrame:
    """Real data by default. Set USE_SYNTHETIC_AUGMENTATION=true to add
    CTGAN-synthesized rows on top - only takes effect if
    validate_synthetic_features.py's gate actually approved them; see
    excel_reader.load_augmented_training_frame()."""
    rows = load_augmented_training_frame()
    df = pd.DataFrame(rows)
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].astype(float)
    df["is_fraud"] = df["is_fraud"].astype(bool)
    return df


def precision_at_k(is_fraud: pd.Series, anomaly_score: np.ndarray, k: int) -> float:
    """Fraction of the k most-anomalous rows (lowest score_samples) that are
    actually fraud. k defaults to the true fraud count so this is directly
    comparable to recall@k."""
    order = np.argsort(anomaly_score)  # ascending: most anomalous first
    top_k_idx = order[:k]
    return float(is_fraud.iloc[top_k_idx].mean()) if k > 0 else float("nan")


def save_anomaly_score_plot(df: pd.DataFrame, scores: np.ndarray, path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["crimson" if f else "steelblue" for f in df["is_fraud"]]
    ax.scatter(range(len(scores)), scores, c=colors)
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Row index")
    ax.set_ylabel("Isolation Forest score_samples (lower = more anomalous)")
    ax.set_title("Anomaly score vs. fraud label (red = fraud)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def train() -> IsolationForest:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_dataset()
    non_fraud_df = df[~df["is_fraud"]]

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="isolation_forest"):
        model = IsolationForest(
            n_estimators=200,
            contamination=0.02,
            random_state=42,
        )
        model.fit(non_fraud_df[FEATURE_COLUMNS])

        mlflow.log_params(model.get_params())
        mlflow.log_param("n_rows_trained_on", len(non_fraud_df))
        mlflow.log_param("n_rows_total_dataset", len(df))
        mlflow.sklearn.log_model(model, artifact_path="isolation_forest_model")

        # Score the FULL dataset (including fraud rows) to evaluate against labels.
        scores = model.score_samples(df[FEATURE_COLUMNS])
        mlflow.log_metric("anomaly_score_mean", float(np.mean(scores)))
        mlflow.log_metric("anomaly_score_std", float(np.std(scores)))
        mlflow.log_metric("anomaly_score_min", float(np.min(scores)))
        mlflow.log_metric("anomaly_score_max", float(np.max(scores)))

        n_fraud = int(df["is_fraud"].sum())
        precision_top_k = precision_at_k(df["is_fraud"], scores, k=n_fraud)
        mlflow.log_metric("precision_at_k_top_scoring_anomalies", precision_top_k)
        mlflow.log_param("k_for_precision_at_k", n_fraud)

        correlation = float(np.corrcoef(scores, df["is_fraud"].astype(int))[0, 1])
        mlflow.log_metric("anomaly_score_fraud_correlation", correlation)

        plot_path = os.path.join(OUTPUT_DIR, "isolation_forest_anomaly_scores.png")
        save_anomaly_score_plot(df, scores, plot_path)
        mlflow.log_artifact(plot_path)

        run_id = mlflow.active_run().info.run_id
        register_model(
            run_id, artifact_path="isolation_forest_model", registered_name=MODEL_NAME_ISOLATION_FOREST
        )

        print(f"Trained on {len(non_fraud_df)} non-fraud rows out of {len(df)} total")
        print(f"Anomaly score vs is_fraud correlation: {correlation:.2f}")
        print(f"Precision@{n_fraud} (top-scoring anomalies): {precision_top_k:.2f}")
        print(f"Plot saved to {plot_path}")

    return model


if __name__ == "__main__":
    train()
