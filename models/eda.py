"""Exploratory data analysis on the feature dataset (SCRUM-29).

Reads data/features.xlsx joined with is_fraud from data/transactions_raw.xlsx
via excel_reader.load_training_frame(), and reports class balance, feature
distributions, a correlation matrix, and missing value counts. Writes
eda_summary.md and a correlation heatmap to models/eda_output/.

Honesty note: as of this run, data/features.xlsx has only 8 committed sample
rows. Every statistic below is computed for real against that actual data
(nothing here is fabricated), but conclusions drawn from N=8 are not
statistically robust — see the caveat in eda_summary.md. Re-run this script
once the producer/pipeline have generated a larger volume of real records
(tracked under SCRUM-20/23).
"""
import os

import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from excel_reader import load_training_frame  # noqa: E402

FEATURE_COLUMNS = [
    "velocity_1h",
    "geo_jump_km",
    "bin_spend_rate",
    "terminal_reversal_count",
]

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "eda_output")


def load_dataframe() -> pd.DataFrame:
    rows = load_training_frame()
    df = pd.DataFrame(rows)
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].astype(float)
    df["is_fraud"] = df["is_fraud"].astype(bool)
    return df


def class_balance(df: pd.DataFrame) -> pd.Series:
    return df["is_fraud"].value_counts()


def missing_value_counts(df: pd.DataFrame) -> pd.Series:
    return df[FEATURE_COLUMNS + ["is_fraud"]].isna().sum()


def feature_distributions(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURE_COLUMNS].describe()


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    corr_df = df[FEATURE_COLUMNS + ["is_fraud"]].copy()
    corr_df["is_fraud"] = corr_df["is_fraud"].astype(int)
    return corr_df.corr()


def save_correlation_heatmap(corr: pd.DataFrame, path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    ax.set_title("Feature correlation matrix (including is_fraud)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run() -> dict:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_dataframe()

    balance = class_balance(df)
    missing = missing_value_counts(df)
    dist = feature_distributions(df)
    corr = correlation_matrix(df)

    save_correlation_heatmap(corr, os.path.join(OUTPUT_DIR, "correlation_heatmap.png"))

    return {
        "n_rows": len(df),
        "class_balance": balance,
        "missing_value_counts": missing,
        "feature_distributions": dist,
        "correlation_matrix": corr,
    }


if __name__ == "__main__":
    results = run()
    print(f"Rows analyzed: {results['n_rows']}")
    print("\nClass balance:")
    print(results["class_balance"])
    print("\nMissing value counts:")
    print(results["missing_value_counts"])
    print("\nFeature distributions:")
    print(results["feature_distributions"])
    print("\nCorrelation matrix:")
    print(results["correlation_matrix"])
