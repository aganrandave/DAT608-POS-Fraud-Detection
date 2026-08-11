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

Conditional sampling on is_fraud is used from the start (the DAT610
exercise found unconditional sampling badly inflates the synthetic fraud
rate - no need to repeat that mistake here).

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

from excel_reader import load_training_frame

FEATURE_COLUMNS = [
    "velocity_1h",
    "geo_jump_km",
    "bin_spend_rate",
    "terminal_reversal_count",
]

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


def detect_metadata(real_df: pd.DataFrame) -> SingleTableMetadata:
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(real_df)
    print("Detected metadata:")
    print(json.dumps(metadata.to_dict(), indent=2))
    return metadata


def train_synthesizer(real_df: pd.DataFrame, metadata: SingleTableMetadata) -> CTGANSynthesizer:
    synthesizer = CTGANSynthesizer(metadata, epochs=CTGAN_EPOCHS, verbose=True)
    synthesizer.fit(real_df)
    return synthesizer


def sample_conditionally_on_fraud_rate(
    synthesizer: CTGANSynthesizer, real_df: pd.DataFrame, n_total: int
) -> pd.DataFrame:
    fraud_rate = real_df["is_fraud"].mean()
    n_fraud = round(n_total * fraud_rate)
    n_non_fraud = n_total - n_fraud

    conditions = [
        Condition(num_rows=n_non_fraud, column_values={"is_fraud": 0}),
        Condition(num_rows=n_fraud, column_values={"is_fraud": 1}),
    ]
    synthetic_df = synthesizer.sample_from_conditions(conditions=conditions)
    return synthetic_df.sample(frac=1, random_state=42).reset_index(drop=True)


def write_documentation_checkpoint(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> None:
    checkpoint = {
        "date_generated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "generator_model": f"CTGAN via SDV (epochs={CTGAN_EPOCHS}, conditional sampling on is_fraud)",
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
    metadata = detect_metadata(real_df)

    synthesizer = train_synthesizer(real_df, metadata)
    synthetic_df = sample_conditionally_on_fraud_rate(synthesizer, real_df, n_total=len(real_df))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    synthetic_df.to_csv(SYNTHETIC_CSV_PATH, index=False)
    write_documentation_checkpoint(real_df, synthetic_df)

    print(f"Generated {len(synthetic_df)} synthetic rows -> {SYNTHETIC_CSV_PATH}")
    print(f"Synthetic fraud rate: {synthetic_df['is_fraud'].mean():.2%} (real was {real_df['is_fraud'].mean():.2%})")


if __name__ == "__main__":
    main()
