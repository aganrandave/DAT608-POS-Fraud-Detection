"""Regenerates data/reference/terminals.xlsx and merchants.xlsx at the real
scale the producer actually uses (500 terminals, 200 merchants), instead of
the 6/5-row hand-written samples left over from the initial scaffold.

Terminal id/merchant_id/state/lga/lat/lon come directly from
producer.transaction_generator.TERMINAL_REGISTRY - the same deterministic,
seeded registry the producer itself builds at import time - so the
reference data the dashboard reads actually matches the terminal_id space
used in the 30,008 real bulk-generated transactions (TRM00001-TRM00500),
rather than being a disconnected, separately-invented sample.

terminal_name/operator/is_active (terminals) and merchant_name/category/
registration_date/is_flagged (merchants) have no equivalent in the producer
- there's no "real" business name to draw from - so they're synthesized
here, deterministically (RANDOM_SEED), clearly in the same spirit as the
rest of this project's disclosed synthetic Nigerian data.
"""
import os
import random
import sys
from datetime import date, timedelta

import openpyxl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "producer"))
from nibss_distributions import MERCHANT_CATEGORIES, OPERATORS, STATE_LGAS  # noqa: E402
from transaction_generator import TERMINAL_REGISTRY  # noqa: E402

RANDOM_SEED = 42
CATEGORIES = list(MERCHANT_CATEGORIES)
BUSINESS_SUFFIXES = ["Ltd", "Stores", "Enterprises", "Plaza", "Mart", "Concepts", "Ventures"]

TERMINALS_XLSX = os.path.join(os.path.dirname(__file__), "..", "data", "reference", "terminals.xlsx")
MERCHANTS_XLSX = os.path.join(os.path.dirname(__file__), "..", "data", "reference", "merchants.xlsx")


def _generate_merchants(rng: random.Random) -> dict[str, dict]:
    merchant_ids = sorted({p["merchant_id"] for p in TERMINAL_REGISTRY.values()})
    merchants = {}
    for mid in merchant_ids:
        state = rng.choice(list(STATE_LGAS))
        lga = rng.choice(STATE_LGAS[state])
        category = rng.choice(CATEGORIES)
        name = f"{lga} {category} {rng.choice(BUSINESS_SUFFIXES)}"
        registered = date(2026, 1, 1) - timedelta(days=rng.randint(30, 365 * 4))
        merchants[mid] = {
            "merchant_id": mid,
            "merchant_name": name,
            "category": category,
            "state": state,
            "lga": lga,
            "registration_date": registered.isoformat(),
            "is_flagged": rng.random() < 0.03,
        }
    return merchants


def _generate_terminals(rng: random.Random, merchants: dict[str, dict]) -> list[dict]:
    rows = []
    for i, (terminal_id, profile) in enumerate(sorted(TERMINAL_REGISTRY.items()), start=1):
        merchant = merchants[profile["merchant_id"]]
        rows.append(
            {
                "terminal_id": terminal_id,
                "terminal_name": f"{merchant['merchant_name']} Counter {((i - 1) % 3) + 1}",
                "merchant_id": profile["merchant_id"],
                "state": profile["state"],
                "lga": profile["lga"],
                "latitude": profile["latitude"],
                "longitude": profile["longitude"],
                "operator": rng.choice(OPERATORS),
                "is_active": rng.random() > 0.02,
            }
        )
    return rows


def write_xlsx(path: str, columns: list[str], rows: list[dict]) -> None:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(columns)
    for row in rows:
        ws.append([row[c] for c in columns])
    wb.save(path)


def main() -> None:
    rng = random.Random(RANDOM_SEED)

    merchants = _generate_merchants(rng)
    write_xlsx(
        MERCHANTS_XLSX,
        ["merchant_id", "merchant_name", "category", "state", "lga", "registration_date", "is_flagged"],
        list(merchants.values()),
    )

    terminal_columns = [
        "terminal_id", "terminal_name", "merchant_id", "state", "lga",
        "latitude", "longitude", "operator", "is_active",
    ]
    terminals = _generate_terminals(rng, merchants)
    write_xlsx(TERMINALS_XLSX, terminal_columns, terminals)

    print(f"Wrote {len(merchants)} merchants -> {MERCHANTS_XLSX}")
    print(f"Wrote {len(terminals)} terminals -> {TERMINALS_XLSX}")


if __name__ == "__main__":
    main()
