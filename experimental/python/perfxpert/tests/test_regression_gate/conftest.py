"""Fixtures for regression-gate false-positive suite."""

import pytest

from tests.test_regression_gate.proven_optimization_runner import (
    ProvenOptimizationRunner,
)


@pytest.fixture(scope="session")
def proven_runner() -> ProvenOptimizationRunner:
    return ProvenOptimizationRunner()
