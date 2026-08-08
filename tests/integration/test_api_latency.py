"""Latency SLA check for the FastAPI scoring endpoint using scoring/benchmark.py."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scoring"))

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION_TESTS") != "1",
    reason="Set RUN_INTEGRATION_TESTS=1 with the scoring service running to enable.",
)

P95_LATENCY_SLA_MS = 250


def test_score_endpoint_meets_latency_sla():
    from benchmark import run_benchmark

    results = run_benchmark(n_requests=50)
    assert results["p95_ms"] < P95_LATENCY_SLA_MS
