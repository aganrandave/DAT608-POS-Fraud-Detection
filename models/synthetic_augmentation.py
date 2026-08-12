"""Generates a validated CTGAN synthetic augmentation of the real training
data (data/features.xlsx joined with is_fraud via excel_reader.py).

This retargets the DAT610 (Ethics & Privacy) synthetic-data-validation
exercise's CTGAN pipeline and five-level validation framework at this
project's actual schema, instead of the coursework's FintechPay stand-in
columns. Context: even after bulk-generating 30,000 rows via
producer/bulk_generate.py + pipeline/batch_feature_engineering.py
(SCRUM-20/23), fraud is intentionally rare (~2% FRAUD_RATE in
producer/config.py), so the model only ever sees a few hundred positive
examples. This script asks whether a CTGAN-synthesized augmentation of
the minority class can be validated as safe to add on top of that.

Conditional sampling is used from the start, jointly on all of
JOINT_CONDITION_COLUMNS (is_fraud, terminal_reversal_count, velocity_1h,
has_geo_jump - see sample_conditionally_on_joint_distribution) - the
DAT610 exercise found unconditional sampling badly inflates the synthetic
fraud rate, and each narrower version of this conditioning left the next
column's own rare-category rate uncorrected; no need to repeat any of
those mistakes here. velocity_1h, terminal_reversal_count, and the
training-only has_geo_jump helper (see add_training_helper_columns) are
explicitly overridden to categorical sdtype (see
CATEGORICAL_OVERRIDE_COLUMNS below) - all are low-cardinality/binary
columns that CTGAN's default "numerical" mode-specific normalisation
modelled poorly, which was the diagnosed root cause of the validation
gate's persistent KS/Wasserstein/chi-squared failures on exactly these
columns across every prior run of this pipeline.

Deliberately does NOT fabricate transaction_id/terminal_id/card_bin/
timestamp identifier columns, and does NOT write synthetic rows into
data/features.xlsx - the real Excel data store stays exclusively real,
preserving provenance. Output goes to synthetic_output/, a separate,
clearly-labelled artifact that train_xgboost.py/train_isolation_forest.py
only load if explicitly opted into via USE_SYNTHETIC_AUGMENTATION=true
AND validate_synthetic_features.py's gate has actually passed - see
that script's is_approved_for_training().
"""
import json
import os
from datetime import datetime, timezone

import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.sampling import Condition
from sdv.single_table import CTGANSynthesizer

from excel_reader import FEATURE_COLUMNS, load_training_frame

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "synthetic_output")
SYNTHETIC_CSV_PATH = os.path.join(OUTPUT_DIR, "synthetic_features.csv")
DOC_CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "documentation_checkpoint.json")

CTGAN_EPOCHS = 300


def load_real_data() -> pd.DataFrame:
    rows = load_training_frame()
    df = pd.DataFrame(rows)[FEATURE_COLUMNS + ["is_fraud"]]
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].astype(float)
    df["is_fraud"] = df["is_fraud"].astype(int)
    return df


# geo_jump_km is 98.7% exactly zero (only cloned_card fraud creates a real
# jump; every other row has geo_jump_km == 0 by construction, see
# feature_windows.with_geo_jump). CTGAN's continuous mode-specific
# normalisation can't produce an exact spike at zero - it was smearing the
# whole column into near-zero noise (49% exactly 0 in synthetic vs. 98.7%
# real, tail median 0.012 vs. real's 123 when nonzero). has_geo_jump is a
# training-time helper column (added here, stripped from the output before
# writing synthetic_features.csv) that turns "should this row be zero"
# into a conditionable categorical decision, the same technique already
# applied to velocity_1h/terminal_reversal_count - so CTGAN's remaining
# job is only fitting the tail's shape within the has_geo_jump=1 subset.
def add_training_helper_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(has_geo_jump=(df["geo_jump_km"] > 0).astype(int))


# Columns auto-detected as "numerical" that are actually low-cardinality
# integer counts (velocity_1h: 6 distinct values, 94% at the minimum;
# terminal_reversal_count: 3 distinct values, 98.5% at zero). CTGAN models
# "numerical" columns via mode-specific Gaussian-mixture normalisation,
# which is a poor fit for a near-constant spike distribution - this was
# the diagnosed root cause of validate_synthetic_features.py's KS/
# Wasserstein failures on exactly these two columns. Overriding them to
# "categorical" lets CTGAN model each observed integer as its own class.
CATEGORICAL_OVERRIDE_COLUMNS = ["velocity_1h", "terminal_reversal_count", "has_geo_jump"]


def detect_metadata(real_df: pd.DataFrame) -> SingleTableMetadata:
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(real_df)
    print("Auto-detected metadata:")
    print(json.dumps(metadata.to_dict(), indent=2))

    # The lecture's own pipeline step 2 says "Always verify" the detected
    # metadata rather than trust it blindly - this is that verification,
    # acted on rather than just printed.
    for column in CATEGORICAL_OVERRIDE_COLUMNS:
        metadata.update_column(column_name=column, sdtype="categorical")
    print("Metadata after manual correction (velocity_1h, terminal_reversal_count -> categorical):")
    print(json.dumps(metadata.to_dict(), indent=2))

    return metadata


def train_synthesizer(real_df: pd.DataFrame, metadata: SingleTableMetadata) -> CTGANSynthesizer:
    synthesizer = CTGANSynthesizer(metadata, epochs=CTGAN_EPOCHS, verbose=True)
    synthesizer.fit(real_df)
    return synthesizer


JOINT_CONDITION_COLUMNS = ["is_fraud", "terminal_reversal_count", "velocity_1h", "has_geo_jump"]


def sample_conditionally_on_joint_distribution(
    synthesizer: CTGANSynthesizer, training_df: pd.DataFrame
) -> pd.DataFrame:
    """Sample the same total row count as training_df, conditioning
    JOINTLY on all of JOINT_CONDITION_COLUMNS at their exact real joint
    counts - not just is_fraud alone (the original version of this
    function). Each prior version left the next categorical column's own
    rarity uncorrected: conditioning on is_fraud alone left
    terminal_reversal_count over-generated within each fraud class; adding
    terminal_reversal_count left velocity_1h's chi-squared still failing;
    adding velocity_1h left geo_jump_km's zero-vs-nonzero split
    uncorrected (see has_geo_jump above and validate_synthetic_features.py's
    module docstring for the full history). This generalises the same
    conditional-sampling technique to all four at once via their real
    joint distribution (28 cells for this dataset).

    Several joint cells are extremely rare (as low as 1 of 30,008 real
    rows) - max_tries_per_batch is raised well above SDV's default to give
    the conditional sampler a real chance at them rather than silently
    dropping them."""
    joint_counts = training_df.groupby(JOINT_CONDITION_COLUMNS).size()
    conditions = [
        Condition(
            num_rows=int(count),
            column_values=dict(zip(JOINT_CONDITION_COLUMNS, (v.item() if hasattr(v, "item") else v for v in key))),
        )
        for key, count in joint_counts.items()
        if count > 0
    ]
    synthetic_df = synthesizer.sample_from_conditions(conditions=conditions, max_tries_per_batch=2000)
    return synthetic_df.sample(frac=1, random_state=42).reset_index(drop=True)


def write_documentation_checkpoint(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> None:
    checkpoint = {
        "date_generated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "generator_model": (
            f"CTGAN via SDV (epochs={CTGAN_EPOCHS}, conditional sampling jointly on "
            f"{JOINT_CONDITION_COLUMNS})"
        ),
        "training_data_source": "data/features.xlsx joined with data/transactions_raw.xlsx is_fraud label",
        "real_records": len(real_df),
        "real_fraud_rate": round(float(real_df["is_fraud"].mean()), 4),
        "synthetic_records": len(synthetic_df),
        "synthetic_fraud_rate": round(float(synthetic_df["is_fraud"].mean()), 4),
        "purpose": "Minority-class training augmentation for train_xgboost.py / train_isolation_forest.py",
        "data_sensitivity": "Derived from producer-simulated transactions - no real customer PII",
        "validation_status": "PENDING - run validate_synthetic_features.py before enabling augmentation",
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(DOC_CHECKPOINT_PATH, "w") as f:
        json.dump(checkpoint, f, indent=2)
    print(f"Documentation checkpoint written -> {DOC_CHECKPOINT_PATH}")


def main() -> None:
    real_df = load_real_data()
    training_df = add_training_helper_columns(real_df)
    metadata = detect_metadata(training_df)

    synthesizer = train_synthesizer(training_df, metadata)
    synthetic_df = sample_conditionally_on_joint_distribution(synthesizer, training_df)

    # has_geo_jump=0 implies geo_jump_km=0 deterministically in every real
    # row (see add_training_helper_columns) - CTGAN's continuous decoder
    # cannot reliably produce an exact point mass at zero (measured: only
    # 35% exactly 0 vs. the true 98.7%, driving a KS statistic of 0.63
    # despite an "excellent" Wasserstein score, since the misplaced mass
    # sits very close to zero but not AT zero). Enforcing the known-exact
    # relationship directly is more reliable than hoping the network
    # learns a hard logical constraint it isn't built for.
    synthetic_df.loc[synthetic_df["has_geo_jump"] == 0, "geo_jump_km"] = 0.0
    synthetic_df = synthetic_df[FEATURE_COLUMNS + ["is_fraud"]]  # drop has_geo_jump - training aid only

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    synthetic_df.to_csv(SYNTHETIC_CSV_PATH, index=False)
    write_documentation_checkpoint(real_df, synthetic_df)

    print(f"Generated {len(synthetic_df)} synthetic rows -> {SYNTHETIC_CSV_PATH}")
    print(f"Synthetic fraud rate: {synthetic_df['is_fraud'].mean():.2%} (real was {real_df['is_fraud'].mean():.2%})")


if __name__ == "__main__":
    main()
