"""Tier 0 source-only contract for fixtures that are excluded from parity."""

from __future__ import annotations

import pytest

from .fixtures_inventory import available_source_only_fixtures
from .parity_runner import ParityRunner


SOURCE_ONLY_FIXTURES = available_source_only_fixtures()


def test_source_only_fixture_inventory_nonempty() -> None:
    assert SOURCE_ONLY_FIXTURES, "Tier 0 source-only fixtures are required for this contract"


@pytest.mark.parametrize(
    "source_fx",
    SOURCE_ONLY_FIXTURES,
    ids=[fx.id for fx in SOURCE_ONLY_FIXTURES],
)
def test_source_only_fixture_contract(source_fx) -> None:
    """Tier 0 fixtures should validate against the current source-scan path."""
    assert source_fx.source_dir is not None
    result = ParityRunner().run_fixture(source_fx)
    assert result.agree_bottleneck()
    assert result.agree_rec_type()
    assert result.agree_rec_technique()
