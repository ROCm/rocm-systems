"""Fixtures for test_parity suite."""

from __future__ import annotations

import pytest

from .fixtures_inventory import available_parity_fixtures
from .parity_runner import ParityRunner


@pytest.fixture(scope="session")
def parity_runner() -> ParityRunner:
    return ParityRunner()


def pytest_generate_tests(metafunc):
    """Parametrize every test that takes `fx` over all available fixtures."""
    if "fx" in metafunc.fixturenames:
        fixtures = available_parity_fixtures()
        ids = [fx.id for fx in fixtures]
        metafunc.parametrize("fx", fixtures, ids=ids)
