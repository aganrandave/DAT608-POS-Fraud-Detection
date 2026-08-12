"""Five-level synthetic data validation (DAT610 framework), retargeted at
this project's real feature schema (velocity_1h, geo_jump_km,
bin_spend_rate, terminal_reversal_count, amount_ngn,
amount_vs_bin_avg_ratio, is_fraud) instead of the DAT610 coursework's
FintechPay stand-in columns.

Column typing for Level 3 (mirrors the categorical override in
synthetic_augmentation.py's CATEGORICAL_OVERRIDE_COLUMNS): velocity_1h
and terminal_reversal_count are low-cardinality integer counts (6 and 3
distinct values respectively, both heavily spiked near their minimum) -
tested via chi-squared (3b) like is_fraud, not KS/Wasserstein (3a/3c),
since a discrete spike distribution isn't what those continuous-shape
tests are designed to compare. geo_jump_km, bin_spend_rate, amount_ngn,
and amount_vs_bin_avg_ratio are genuinely continuous and use KS/Wasserstein.
Level 1 (summary stats) and Level 5 (privacy DNNR) still use all six
features together, since both are dtype-agnostic.

is_approved_for_training() is the production gate: train_xgboost.py /
train_isolation_forest.py only load synthetic_output/synthetic_features.csv
if this function returns True AND the operator has opted in via
USE_SYNTHETIC_AUGMENTATION=true (see excel_reader.load_augmented_training_frame).
The bar is deliberately strict and consistent with the DAT610 report's own
conclusion that passing ML-utility (Level 4) alone is not sufficient
justification: ALL continuous columns must pass KS (3a) and score
acceptable-or-better on Wasserstein (3c), ALL categorical columns
(including the fraud rate) must be chi-squared aligned (3b), TSTR/TRTR
AUC gap must be acceptable (4), and privacy must not indicate
memorisation (5).
"""
import json
import os

import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.stats import wasserstein_distance  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from synthetic_augmentation import FEATURE_COLUMNS, load_real_data, OUTPUT_DIR, SYNTHETIC_CSV_PATH  # noqa: E402
from train_xgboost import build_model, compute_scale_pos_weight  # noqa: E402

NUMERIC_COLS = FEATURE_COLUMNS  # all six - used for Level 1 and Level 5 only
CONTINUOUS_COLS = ["geo_jump_km", "bin_spend_rate", "amount_ngn", "amount_vs_bin_avg_ratio"]  # Level 3a/3c
CATEGORICAL_COLS = ["is_fraud", "velocity_1h", "terminal_reversal_count"]  # Level 3b

RESULTS_PATH = os.path.join(OUTPUT_DIR, "validation_results.json")
RANDOM_SEED = 42


def level1_summary_statistics(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> dict:
    results = {}
    for col in NUMERIC_COLS:
        real_mean, synth_mean = real_df[col].mean(), synth_df[col].mean()
        pct_diff = abs(real_mean - synth_mean) / abs(real_mean) * 100 if real_mean != 0 else float("inf")
        flag = "ok" if pct_diff <= 10 else ("warn" if pct_diff <= 20 else "fail")
        results[col] = {
            "real_mean": round(float(real_mean), 4),
            "synthetic_mean": round(float(synth_mean), 4),
            "mean_pct_diff": round(float(pct_diff), 2),
            "flag": flag,
        }
    return results


def level2_visual_distributions(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> list:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    saved_paths = []

    # 2x3 to fit all six NUMERIC_COLS - a 2x2 grid here (from before
    # amount_ngn/amount_vs_bin_avg_ratio were added) was silently truncating
    # via zip() and never plotting the last two columns.
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, col in zip(axes.flat, NUMERIC_COLS):
        sns.kdeplot(real_df[col], ax=ax, label="real", fill=True, alpha=0.3)
        sns.kdeplot(synth_df[col], ax=ax, label="synthetic", fill=True, alpha=0.3)
        ax.set_title(col)
        ax.legend()
    fig.tight_layout()
    kde_path = os.path.join(OUTPUT_DIR, "level2_kde_distributions.png")
    fig.savefig(kde_path, dpi=120)
    plt.close(fig)
    saved_paths.append(kde_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(real_df[NUMERIC_COLS].corr(), ax=axes[0], annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    axes[0].set_title("Real: correlation")
    sns.heatmap(synth_df[NUMERIC_COLS].corr(), ax=axes[1], annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    axes[1].set_title("Synthetic: correlation")
    fig.tight_layout()
    heatmap_path = os.path.join(OUTPUT_DIR, "level2_correlation_heatmaps.png")
    fig.savefig(heatmap_path, dpi=120)
    plt.close(fig)
    saved_paths.append(heatmap_path)

    return saved_paths


def level3a_ks_test(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> dict:
    results = {}
    for col in CONTINUOUS_COLS:
        ks_stat, p_value = stats.ks_2samp(real_df[col].dropna(), synth_df[col].dropna())
        results[col] = {
            "ks_statistic": round(float(ks_stat), 4),
            "p_value": round(float(p_value), 4),
            "aligned": bool(p_value > 0.05),
        }
    return results


def level3b_chi_squared(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> dict:
    results = {}
    for col in CATEGORICAL_COLS:
        categories = sorted(
            {v.item() if hasattr(v, "item") else v for v in real_df[col].unique()}
            | {v.item() if hasattr(v, "item") else v for v in synth_df[col].unique()}
        )
        real_counts = real_df[col].value_counts().reindex(categories, fill_value=0)
        synth_props = synth_df[col].value_counts(normalize=True).reindex(categories, fill_value=0)

        f_obs = real_counts.values.astype(float)
        f_exp = (synth_props.values * real_counts.sum()).astype(float)
        f_exp = np.where(f_exp == 0, 1e-6, f_exp)

        chi2_stat, p_value = stats.chisquare(f_obs=f_obs, f_exp=f_exp)
        results[col] = {
            "chi2_statistic": round(float(chi2_stat), 4),
            "p_value": round(float(p_value), 4),
            "aligned": bool(p_value > 0.05),
            "categories": categories,
        }
    return results


def level3c_wasserstein(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> dict:
    results = {}
    for col in CONTINUOUS_COLS:
        raw_distance = wasserstein_distance(real_df[col], synth_df[col])
        sigma_real = real_df[col].std()
        normalized = raw_distance / sigma_real if sigma_real > 0 else float("inf")
        rating = "excellent" if normalized < 0.05 else ("acceptable" if normalized <= 0.15 else "poor")
        results[col] = {
            "raw_distance": round(float(raw_distance), 4),
            "normalized_score": round(float(normalized), 4),
            "rating": rating,
        }
    return results


def level4_tstr_vs_trtr(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> dict:
    real_train, real_test = train_test_split(
        real_df, test_size=0.3, random_state=RANDOM_SEED, stratify=real_df["is_fraud"]
    )

    def _fit_and_score(train_df):
        X_train, y_train = train_df[NUMERIC_COLS], train_df["is_fraud"]
        model = build_model(compute_scale_pos_weight(y_train))
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(real_test[NUMERIC_COLS])[:, 1]
        return float(roc_auc_score(real_test["is_fraud"], y_prob))

    trtr_auc = _fit_and_score(real_train)
    tstr_auc = _fit_and_score(synth_df)

    gap = abs(tstr_auc - trtr_auc)
    if gap < 0.02:
        verdict = "fully_equivalent"
    elif gap <= 0.05:
        verdict = "acceptable_with_monitoring"
    else:
        verdict = "investigate_before_deployment"

    return {
        "trtr_auc": round(trtr_auc, 4),
        "tstr_auc": round(tstr_auc, 4),
        "auc_gap": round(gap, 4),
        "verdict": verdict,
        "real_test_fraud_rate": round(float(real_test["is_fraud"].mean()), 4),
    }


def level5_privacy_dnnr(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> dict:
    scaler = StandardScaler().fit(real_df[NUMERIC_COLS])
    real_scaled = scaler.transform(real_df[NUMERIC_COLS])
    synth_scaled = scaler.transform(synth_df[NUMERIC_COLS])

    nn_real = NearestNeighbors(n_neighbors=2).fit(real_scaled)
    d_rr, _ = nn_real.kneighbors(real_scaled)
    d_rr = d_rr[:, 1]

    nn_for_synth = NearestNeighbors(n_neighbors=1).fit(real_scaled)
    d_sr, _ = nn_for_synth.kneighbors(synth_scaled)
    d_sr = d_sr[:, 0]

    dnnr = float(np.median(d_sr) / np.median(d_rr))
    interpretation = "privacy_preserved" if dnnr > 1.5 else ("borderline" if dnnr > 1.0 else "memorization_risk")

    return {
        "median_d_sr": round(float(np.median(d_sr)), 4),
        "median_d_rr": round(float(np.median(d_rr)), 4),
        "dnnr": round(dnnr, 4),
        "interpretation": interpretation,
    }


def is_approved_for_training(results: dict) -> tuple[bool, list[str]]:
    """The production gate. Deliberately strict - see module docstring."""
    reasons = []

    n_cat = len(CATEGORICAL_COLS)
    n_chi_aligned = sum(1 for v in results["level3b"].values() if v["aligned"])
    if n_chi_aligned < n_cat:
        reasons.append(f"Level 3b: only {n_chi_aligned}/{n_cat} categorical columns chi-squared aligned (all required)")

    n_cont = len(CONTINUOUS_COLS)
    n_ks_aligned = sum(1 for v in results["level3a"].values() if v["aligned"])
    if n_ks_aligned < n_cont:
        reasons.append(f"Level 3a: only {n_ks_aligned}/{n_cont} continuous columns pass KS (all required)")

    n_wass_ok = sum(1 for v in results["level3c"].values() if v["rating"] != "poor")
    if n_wass_ok < n_cont:
        reasons.append(
            f"Level 3c: only {n_wass_ok}/{n_cont} continuous columns rated acceptable/excellent (all required)"
        )

    if results["level4"]["verdict"] == "investigate_before_deployment":
        reasons.append(f"Level 4: TSTR/TRTR AUC gap too large ({results['level4']['auc_gap']})")

    if results["level5"]["interpretation"] == "memorization_risk":
        reasons.append(f"Level 5: privacy DNNR indicates memorisation risk ({results['level5']['dnnr']})")

    return len(reasons) == 0, reasons


def print_summary(results: dict, approved: bool, reasons: list[str]) -> None:
    print("\n=== Validation Summary ===")
    n_fail1 = sum(1 for v in results["level1"].values() if v["flag"] == "fail")
    print(f"L1 Summary statistics  : {n_fail1}/{len(NUMERIC_COLS)} columns FAIL (>20% mean diff)")
    n_ks_fail = sum(1 for v in results["level3a"].values() if not v["aligned"])
    print(f"L3a KS test            : {n_ks_fail}/{len(CONTINUOUS_COLS)} columns NOT aligned")
    n_chi_fail = sum(1 for v in results["level3b"].values() if not v["aligned"])
    print(f"L3b Chi-squared        : {n_chi_fail}/{len(CATEGORICAL_COLS)} columns NOT aligned")
    n_wass_poor = sum(1 for v in results["level3c"].values() if v["rating"] == "poor")
    print(f"L3c Wasserstein        : {n_wass_poor}/{len(CONTINUOUS_COLS)} columns rated POOR")
    print(f"L4 TSTR vs TRTR        : gap={results['level4']['auc_gap']} -> {results['level4']['verdict']}")
    print(f"L5 Privacy DNNR        : {results['level5']['dnnr']} -> {results['level5']['interpretation']}")
    print(f"\nAPPROVED FOR TRAINING AUGMENTATION: {approved}")
    for reason in reasons:
        print(f"  - {reason}")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    real_df = load_real_data()
    synth_df = pd.read_csv(SYNTHETIC_CSV_PATH)

    results = {
        "level1": level1_summary_statistics(real_df, synth_df),
        "level2_plots": level2_visual_distributions(real_df, synth_df),
        "level3a": level3a_ks_test(real_df, synth_df),
        "level3b": level3b_chi_squared(real_df, synth_df),
        "level3c": level3c_wasserstein(real_df, synth_df),
        "level4": level4_tstr_vs_trtr(real_df, synth_df),
        "level5": level5_privacy_dnnr(real_df, synth_df),
    }
    approved, reasons = is_approved_for_training(results)
    results["approved_for_training"] = approved
    results["gate_failure_reasons"] = reasons

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print_summary(results, approved, reasons)
    print(f"\nFull results written -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
