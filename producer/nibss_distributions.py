"""Reference distributions modeled loosely on published NIBSS POS statistics.

These constants drive the synthetic transaction generator so that generated
volumes are proportionally realistic across states and issuing banks,
without depending on any live data source.
"""

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

# card_bin -> (issuing bank, scheme)
CARD_BINS = {
    "539983": ("GTBank", "Mastercard"),
    "440066": ("Access Bank", "Visa"),
    "507338": ("Interswitch", "Verve"),
    "506121": ("Zenith Bank", "Verve"),
    "412119": ("First Bank", "Visa"),
    "533280": ("UBA", "Mastercard"),
}

FRAUD_TYPES = ["cloned_card", "agent_collusion", "fake_reversal"]

# Typical POS transaction amount bands in NGN with relative likelihood
AMOUNT_BANDS = [
    (500, 2_500, 0.35),
    (2_500, 15_000, 0.35),
    (15_000, 50_000, 0.20),
    (50_000, 150_000, 0.10),
]
