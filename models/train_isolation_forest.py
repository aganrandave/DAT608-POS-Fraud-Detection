"""Trains the unsupervised Isolation Forest anomaly detector and logs it to MLflow.

Training data as of this run: 30,008 real rows (bulk-generated via
producer/bulk_generate.py + pipeline/batch_feature_engineering.py,
SCRUM-20/23), ~3.8% fraud, after fixing real weaknesses in the transaction
generator (terminal-location pinning, realistic timestamp spread,
velocity/geo fraud bursts - see transaction_generator.py/fraud_scenarios.py
docstrings). Isolation Forest is fit on non-fraud rows only.

Honesty note: adding amount_ngn/amount_vs_bin_avg_ratio (which meaningfully
improved the supervised XGBoost model - see train_xgboost.py) made this
unsupervised model slightly WORSE: anomaly-score/is_fraud correlation
-0.50->-0.44, precision@1146 0.50->0.42. Extra feature dimensions aren't
free for Isolation Forest the way they are for a supervised model - added
variance in normal transactions' amount can dilute the relative isolation
of fraud points rather than sharpen it. Left as-is rather than reverting,
since the feature set is shared with train_xgboost.py via
excel_reader.FEATURE_COLUMNS and the supervised model is the primary
classifier; worth revisiting if Isolation Forest becomes the priority.

A CTGAN-based synthetic augmentation of this data was evaluated
(models/synthetic_augmentation.py + validate_synthetic_features.py). ML
utility and privacy now pass cleanly, but the gate still does NOT approve
it: CTGAN struggles to reproduce the sparse/bursty velocity_1h and
terminal_reversal_count distributions specifically. USE_SYNTHETIC_AUGMENTATION
has no effect until a synthetic set actually passes that gate.
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

from excel_reader import FEATURE_COLUMNS, load_augmented_training_frame  # noqa: E402
from mlflow_registry import (  # noqa: E402
    EXPERIMENT_NAME,
    ISOLATION_FOREST_BOUNDS_ARTIFACT,
    MODEL_NAME_ISOLATION_FOREST,
    log_sibling_artifact,
    register_model,
)

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
        score_min = float(np.min(scores))
        score_max = float(np.max(scores))
        mlflow.log_metric("anomaly_score_mean", float(np.mean(scores)))
        mlflow.log_metric("anomaly_score_std", float(np.std(scores)))
        mlflow.log_metric("anomaly_score_min", score_min)
        mlflow.log_metric("anomaly_score_max", score_max)

        # Package these same min/max bounds WITH the model version so
        # scoring/scorer.py can reproducibly map a new score_samples() value
        # into a 0-1 "fraudiness" score at serving time: fitted once here,
        # not an arbitrary transform invented at scoring time. score_samples
        # is lower for more anomalous rows (see save_anomaly_score_plot's
        # axis label below), so a new score is normalized as
        # (score_max - new_score) / (score_max - score_min), clipped to
        # [0, 1] since a live transaction can fall outside this training
        # range in either direction.
        log_sibling_artifact(
            {
                "score_min": score_min,
                "score_max": score_max,
                "fitted_on": "score_samples() over the full training-time dataset (fraud + non-fraud)",
                "convention": "lower score_samples = more anomalous",
            },
            ISOLATION_FOREST_BOUNDS_ARTIFACT,
            artifact_path="isolation_forest_model",
        )

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
