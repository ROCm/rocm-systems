"""Shared fixtures for integration tests.

Discovers fixture .db files under tests/fixtures/. If a fixture is
missing, the dependent test is skipped rather than fails — this keeps
CI green when fixtures haven't been regenerated locally.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest


FIXTURE_DIR = Path(__file__).parent.parent / "fixtures"


def _find_fixture(name: str) -> Path:
    """Return path to the named fixture, or None if absent."""
    p = FIXTURE_DIR / name
    return p if p.exists() else None


@pytest.fixture
def compute_bound_db():
    p = _find_fixture("compute_bound.db")
    if p is None:
        pytest.skip("compute_bound.db fixture not present")
    return p


@pytest.fixture
def trace_only_elementwise_db():
    """Trace-only DB (no --pmc counters) — expected to yield data_insufficient verdict."""
    p = _find_fixture("trace_only_elementwise.db")
    if p is None:
        pytest.skip("trace_only_elementwise.db fixture not present")
    return p


@pytest.fixture
def memory_bound_db():
    p = _find_fixture("memory_bound.db")
    if p is None:
        pytest.skip("memory_bound.db fixture not present")
    return p


@pytest.fixture
def latency_bound_db():
    p = _find_fixture("latency_bound.db")
    if p is None:
        pytest.skip("latency_bound.db fixture not present")
    return p


@pytest.fixture
def test_gfx_id():
    return "gfx942"


@pytest.fixture
def regression_db_pair():
    """Tuple (baseline, candidate) — candidate is intentionally slower."""
    b = _find_fixture("baseline.db")
    c = _find_fixture("regressed.db")
    if b is None or c is None:
        pytest.skip("regression fixture pair not present")
    return (b, c)


@pytest.fixture
def fake_provider(monkeypatch):
    """Mock LLM provider for parity tests."""
    mock = MagicMock()
    monkeypatch.setenv("PERFXPERT_PROVIDER", "anthropic")
    return mock
