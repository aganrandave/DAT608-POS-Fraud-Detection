"""Single source of truth for alert-tier colors across the dashboard.

Tier order and severity (critical > high > medium > low) mirrors
alerts/alert_tiers.py's TIER_THRESHOLDS - kept as a separate constant here
rather than importing that module, since the dashboard reads Excel files
directly and has no other runtime dependency on the alerts/ package.
"""

TIER_ORDER = ["critical", "high", "medium", "low"]

# RGB triples for pydeck layers (0-255 per channel).
TIER_RGB = {
    "critical": [214, 39, 40],
    "high": [255, 127, 14],
    "medium": [255, 215, 0],
    "low": [44, 160, 44],
}

# Hex equivalents for CSS / pandas Styler use in tables.
TIER_HEX = {
    "critical": "#d62728",
    "high": "#ff7f0e",
    "medium": "#ffd700",
    "low": "#2ca02c",
}

UNFLAGGED_RGB = [128, 128, 128]  # neutral grey for terminals with no open alert
UNFLAGGED_HEX = "#808080"


def highest_severity_tier(tiers: list[str]) -> str | None:
    """Given a list of alert_tier values (e.g. all alerts for one terminal),
    return the most severe one present, or None if the list is empty."""
    present = set(tiers)
    for tier in TIER_ORDER:
        if tier in present:
            return tier
    return None


def tier_rgb(tier: str | None) -> list[int]:
    return TIER_RGB.get(tier, UNFLAGGED_RGB)


def tier_hex(tier: str | None) -> str:
    return TIER_HEX.get(tier, UNFLAGGED_HEX)
