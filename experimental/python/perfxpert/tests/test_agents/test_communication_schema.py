"""Tests for :class:`perfxpert.agents.schemas.CommunicationBlock` +
:class:`perfxpert.agents.schemas.CollectiveEntry`.

These are the Pydantic types that front the RCCL payload block produced
by :func:`perfxpert.tools.rccl_analysis.analyze_collectives`. The test
cases cover (1) a well-formed payload round-tripping through the
validator, (2) an invalid ``efficiency_label`` being rejected, and
(3) a missing required field (``ranks``) being rejected.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from perfxpert.agents.schemas import CollectiveEntry, CommunicationBlock


def _valid_entry_kwargs(**overrides):
    base = {
        "op_type": "AllReduce",
        "msg_bytes": 1048576,
        "duration_ns": 1_000_000,
        "effective_bw_gbps": 1.57,
        "peak_bw_gbps": 340.0,
        "efficiency_pct": 0.46,
        "efficiency_label": "poor",
        "overlap_ratio": 12.5,
        "algo_hint": "Ring",
        "topology_hint": "multi-node",
        "regime": "bandwidth-bound",
        "ranks": 8,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 1. Valid payload — round-trip through the validator and back to dict.
# ---------------------------------------------------------------------------

def test_communication_block_accepts_valid_payload():
    block = CommunicationBlock(
        collectives=[CollectiveEntry(**_valid_entry_kwargs())],
        summary={
            "op_count": 1,
            "ranks": 8,
            "dominant_op": "AllReduce",
            "avg_bw_gbps": 1.57,
            "avg_efficiency_pct": 0.46,
            "overlap_pct": 12.5,
            "capture_incomplete": False,
        },
        capture_incomplete=False,
    )
    dumped = block.model_dump()
    # Field-shape invariants the formatters downstream rely on.
    assert dumped["capture_incomplete"] is False
    assert dumped["collectives"][0]["op_type"] == "AllReduce"
    assert dumped["collectives"][0]["efficiency_label"] == "poor"
    assert dumped["summary"]["op_count"] == 1


# ---------------------------------------------------------------------------
# 2. Invalid efficiency_label — rejected at the boundary.
# ---------------------------------------------------------------------------

def test_collective_entry_rejects_invalid_efficiency_label():
    with pytest.raises(ValidationError) as excinfo:
        CollectiveEntry(**_valid_entry_kwargs(efficiency_label="excellent"))
    msg = str(excinfo.value)
    # The allowed literal set should appear in the error so the
    # diagnostic is actionable.
    assert "efficiency_label" in msg
    assert "poor" in msg and "fair" in msg and "good" in msg


# ---------------------------------------------------------------------------
# 3. Missing required field — rejected at the boundary.
# ---------------------------------------------------------------------------

def test_collective_entry_rejects_missing_required_field():
    kwargs = _valid_entry_kwargs()
    del kwargs["ranks"]
    with pytest.raises(ValidationError) as excinfo:
        CollectiveEntry(**kwargs)
    msg = str(excinfo.value)
    assert "ranks" in msg
    assert "missing" in msg.lower() or "required" in msg.lower()
