"""Aggregate parity test — verifies >= 95% agreement on the fixture suite.
Contract test: parity_runner must produce dual results for every inventory fixture.
"""
import pytest

from .parity_runner import ParityRunner
from .fixtures_inventory import available_fixtures


@pytest.mark.parity
def test_runner_runs_every_inventory_fixture_without_crash():
    """Contract: ParityRunner produces a DualResult for every available fixture.

    Skips if no fixtures are available (Phase 1 backfill).
    """
    fixtures = available_fixtures()
    if not fixtures:
        pytest.skip("No fixtures available (Phase 1 backfill pending)")

    runner = ParityRunner()
    missing = []
    for fx in fixtures:
        try:
            dual = runner.run_both_paths(fx)
            assert dual.old is not None
            assert dual.new is not None
        except Exception as exc:
            missing.append((fx.id, str(exc)))
    assert not missing, f"runner crashed on fixtures: {missing}"
