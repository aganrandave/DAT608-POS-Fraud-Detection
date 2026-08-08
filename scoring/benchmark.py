"""Simple latency benchmark for the /score endpoint, used by tests/integration/test_api_latency.py."""
import statistics
import time

import requests

SCORE_URL = "http://localhost:8000/score"

SAMPLE_PAYLOAD = {
    "transaction_id": "benchmark-txn",
    "terminal_id": "TRM00001",
    "card_bin": "539983",
    "velocity_1h": 3,
    "geo_jump_km": 0.0,
    "bin_spend_rate": 12000.0,
    "terminal_reversal_count": 0,
}


def run_benchmark(n_requests: int = 100) -> dict:
    latencies_ms = []
    for _ in range(n_requests):
        start = time.perf_counter()
        requests.post(SCORE_URL, json=SAMPLE_PAYLOAD, timeout=5)
        latencies_ms.append((time.perf_counter() - start) * 1000)

    return {
        "p50_ms": statistics.median(latencies_ms),
        "p95_ms": statistics.quantiles(latencies_ms, n=20)[18],
        "max_ms": max(latencies_ms),
    }


if __name__ == "__main__":
    print(run_benchmark())
