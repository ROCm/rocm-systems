"""Tier 0 source-only contract for fixtures that are excluded from parity."""

from __future__ import annotations

import pytest

from perfxpert.ai_analysis.api import analyze_source

from .fixtures_inventory import available_source_only_fixtures
from .parity_runner import (
    _extract_bottleneck,
    _extract_rec_technique,
    _extract_rec_type,
)


SOURCE_ONLY_FIXTURES = available_source_only_fixtures()


@pytest.mark.parametrize(
    "source_fx",
    SOURCE_ONLY_FIXTURES,
    ids=[fx.id for fx in SOURCE_ONLY_FIXTURES],
)
def test_source_only_fixture_contract(source_fx) -> None:
    """Tier 0 fixtures should validate against the source-only API directly."""
    assert source_fx.source_dir is not None

    result = analyze_source(source_dir=str(source_fx.source_dir))

    assert _extract_bottleneck(result) == source_fx.expected_bottleneck
    assert _extract_rec_type(result) == source_fx.expected_rec_type
    assert _extract_rec_technique(result) == source_fx.expected_rec_technique
