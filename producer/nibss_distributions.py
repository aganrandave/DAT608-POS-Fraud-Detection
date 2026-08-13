"""Reference distributions modeled loosely on published NIBSS (Nigeria Inter-
Bank Settlement System) POS statistics and EFInA (Enhancing Financial
Innovation & Access) Access to Finance survey figures.

These constants drive the synthetic transaction generator so that generated
volumes are proportionally realistic across states and issuing banks,
without depending on any live data source.
"""

# POS terminal deployers/acquirers a synthetic terminal is registered under.
# Reflects the agent-banking/POS-acquiring operators NIBSS and EFInA both
# report as dominant in the Nigerian agent-banking channel. Single-sourced
# here so dashboard/generate_reference_data.py doesn't maintain its own copy.
OPERATORS = ["Moniepoint", "Opay", "Palmpay", "Interswitch Quickteller", "Paystack Terminal"]

# Merchant category -> typical POS transaction amount range in NGN, per
# EFInA merchant-spend banding. Deliberately NOT wired into
# transaction_generator.py's _sample_amount(): that function drives the
# amount_ngn distribution the CTGAN validation gate and both trained models
# were tuned against this session, and swapping it for category-conditioned
# amounts would force a full bulk-data regenerate and retrain. Used by
# dashboard/generate_reference_data.py for merchant category labeling.
MERCHANT_CATEGORIES = {
    "Retail": (500, 25_000),
    "Groceries": (500, 15_000),
    "Electronics": (5_000, 150_000),
    "Fashion": (1_500, 50_000),
    "Pharmacy": (500, 10_000),
    "Restaurant": (1_000, 20_000),
    "Fuel Station": (2_000, 30_000),
    "Supermarket": (1_000, 40_000),
    "Hospitality": (5_000, 150_000),
    "Services": (500, 50_000),
}

# Baseline fraud injection rate per NIBSS fraud-landscape reporting.
# producer/config.py's env-overridable FRAUD_RATE defaults to this value.
FRAUD_RATE = 0.02

# Relative transaction volume share by state (approximate, POS-heavy states)
STATE_WEIGHTS = {
    "Lagos": 0.42,
    "Abuja": 0.18,
    "Rivers": 0.12,
    "Kano": 0.15,
    "Oyo": 0.13,
}

STATE_LGAS = {
    "Lagos": ["Ikeja", "Alimosho", "Eti-Osa", "Surulere", "Ajeromi-Ifelodun"],
    "Abuja": ["Municipal Area Council", "Bwari", "Gwagwalada", "Kuje"],
    "Rivers": ["Port Harcourt", "Obio-Akpor", "Eleme"],
    "Kano": ["Kano Municipal", "Fagge", "Nassarawa"],
    "Oyo": ["Ibadan North", "Ibadan South-West", "Egbeda"],
}

# Approximate (lat, lon) bounding centroid per state, used to jitter
# synthetic terminal coordinates
STATE_CENTROIDS = {
    "Lagos": (6.5244, 3.3792),
    "Abuja": (9.0765, 7.3986),
    "Rivers": (4.8156, 7.0498),
    "Kano": (12.0022, 8.5920),
    "Oyo": (7.3775, 3.9470),
}

# card_bin -> (issuing bank, scheme). Illustrative BIN-range prefixes
# covering the issuing banks NIBSS lists among the largest card-present POS
# volumes; not verified against the live BIN registry (synthetic dataset).
CARD_BINS = {
    "539983": ("GTBank", "Mastercard"),
    "440066": ("Access Bank", "Visa"),
    "507338": ("Interswitch", "Verve"),
    "506121": ("Zenith Bank", "Verve"),
    "412119": ("First Bank", "Visa"),
    "533280": ("UBA", "Mastercard"),
    "418742": ("Ecobank", "Visa"),
    "506080": ("FCMB", "Verve"),
    "539312": ("Wema Bank", "Mastercard"),
    "462639": ("Stanbic IBTC", "Visa"),
}

FRAUD_TYPES = ["cloned_card", "agent_collusion", "fake_reversal"]

# Typical POS transaction amount bands in NGN with relative likelihood
AMOUNT_BANDS = [
    (500, 2_500, 0.35),
    (2_500, 15_000, 0.35),
    (15_000, 50_000, 0.20),
    (50_000, 150_000, 0.10),
]
