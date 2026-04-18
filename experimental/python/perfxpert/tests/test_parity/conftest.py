"""Fixtures for test_parity suite."""

from __future__ import annotations

import pytest

from .fixtures_inventory import available_fixtures
from .parity_runner import ParityRunner


@pytest.fixture(scope="session")
def parity_runner() -> ParityRunner:
    return ParityRunner()


def pytest_generate_tests(metafunc):
    """Parametrize every test that takes `fx` over all available fixtures.

    Trace-only fixtures (has_pmc_counters=False) are marked xfail because the
    legacy path uses heuristics while the agentic path correctly returns
    data_insufficient when no PMC counters are available.  These will be flipped
    to normal asserts once re-profiled with --pmc (Phase 7 Part C).
    """
    if "fx" in metafunc.fixturenames:
        fixtures = available_fixtures()
        params = []
        ids = []
        for fx in fixtures:
            if fx.has_pmc_counters:
                params.append(pytest.param(fx))
            else:
                params.append(
                    pytest.param(
                        fx,
                        marks=pytest.mark.xfail(
                            reason=(
                                f"Tier-1 trace-only fixture '{fx.id}' has no PMC counters. "
                                "Legacy path uses heuristics; agentic path returns "
                                "data_insufficient (correct behavior). "
                                "xfail pending --pmc re-profile (Phase 7 Part C)."
                            ),
                            strict=False,
                        ),
                    )
                )
            ids.append(fx.id)
        metafunc.parametrize("fx", params, ids=ids)
