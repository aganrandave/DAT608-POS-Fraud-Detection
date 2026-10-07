"""Exploratory data analysis on the feature dataset (SCRUM-29).

Reads data/features.xlsx joined with is_fraud (and fraud_type) from
data/transactions_raw.xlsx via excel_reader.load_training_frame(), and
reports row counts, class balance, fraud-type distribution, feature
distributions, a correlation matrix, and missing value counts. Writes
eda_summary.md and a correlation heatmap to models/eda_output/.

Every number this script prints or writes is computed directly against the
CURRENT committed data/features.xlsx + data/transactions_raw.xlsx - nothing
is copied from a comment or a prior run's output. Re-run this script (and
regenerate eda_summary.md from its output) whenever the committed training
data changes meaningfully, so the markdown summary never drifts from what
the data actually contains.
"""
import os
import subprocess

import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from excel_reader import (  # noqa: E402
    FEATURE_COLUMNS,
    TRANSACTIONS_XLSX,
    _read_rows,
    load_training_frame,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "eda_output")


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=os.path.dirname(__file__), text=True
        ).strip()
    except Exception:  # noqa: BLE001 - EDA metadata, not worth failing the run over
        return "unknown"


def load_dataframe() -> pd.DataFrame:
    rows = load_training_frame()
    df = pd.DataFrame(rows)
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].astype(float)
    df["is_fraud"] = df["is_fraud"].astype(bool)
    return df


def load_fraud_type_series() -> pd.Series:
    """fraud_type isn't carried through load_training_frame() (features.xlsx
    has no such column - it's a property of the raw transaction, not a
    derived feature), so read transactions_raw.xlsx directly for it."""
    txn_rows = _read_rows(TRANSACTIONS_XLSX)
    return pd.Series([row.get("fraud_type") or None for row in txn_rows], name="fraud_type")


def row_counts(df: pd.DataFrame) -> dict:
    n_fraud = int(df["is_fraud"].sum())
    n_total = len(df)
    return {
        "n_rows": n_total,
        "n_fraud": n_fraud,
        "n_non_fraud": n_total - n_fraud,
        "fraud_rate_pct": round(100 * n_fraud / n_total, 3) if n_total else float("nan"),
    }


def fraud_type_distribution(fraud_types: pd.Series) -> pd.Series:
    return fraud_types.dropna().value_counts()


def missing_value_counts(df: pd.DataFrame) -> pd.Series:
    return df[FEATURE_COLUMNS + ["is_fraud"]].isna().sum()


def feature_distributions(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURE_COLUMNS].describe()


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    corr_df = df[FEATURE_COLUMNS + ["is_fraud"]].copy()
    corr_df["is_fraud"] = corr_df["is_fraud"].astype(int)
    return corr_df.corr()


def save_correlation_heatmap(corr: pd.DataFrame, path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(im, ax=ax)
    ax.set_title("Feature correlation matrix (including is_fraud)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run() -> dict:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_dataframe()
    fraud_types = load_fraud_type_series()

    counts = row_counts(df)
    fraud_types_dist = fraud_type_distribution(fraud_types)
    missing = missing_value_counts(df)
    dist = feature_distributions(df)
    corr = correlation_matrix(df)

    save_correlation_heatmap(corr, os.path.join(OUTPUT_DIR, "correlation_heatmap.png"))

    return {
        **counts,
        "fraud_type_distribution": fraud_types_dist,
        "missing_value_counts": missing,
        "feature_distributions": dist,
        "correlation_matrix": corr,
        "git_commit": git_commit(),
    }


if __name__ == "__main__":
    results = run()
    print(f"Snapshot commit: {results['git_commit']}")
    print(f"Rows: {results['n_rows']} (fraud={results['n_fraud']}, "
          f"non_fraud={results['n_non_fraud']}, rate={results['fraud_rate_pct']}%)")
    print("\nFraud type distribution:")
    print(results["fraud_type_distribution"])
    print("\nMissing value counts:")
    print(results["missing_value_counts"])
    print("\nFeature distributions:")
    print(results["feature_distributions"])
    print("\nCorrelation matrix:")
    print(results["correlation_matrix"])
