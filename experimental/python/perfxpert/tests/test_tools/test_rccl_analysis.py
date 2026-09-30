"""Tests for perfxpert.tools.rccl_analysis + tools.interconnect."""

from pathlib import Path

import pytest

from perfxpert.tools import interconnect, rccl_analysis
from perfxpert.tools._class import ToolClass

_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def _require(path: Path) -> str:
    """Regenerate fixtures on demand — the CI checkout may not carry them."""
    if not path.exists():
        from tests.fixtures import _generate_rccl_fixture  # type: ignore
        _generate_rccl_fixture.main()
    return str(path)


# --------------------------------------------------------------------------- #
# analyze_collectives                                                         #
# --------------------------------------------------------------------------- #

def test_analyze_collectives_empty_returns_empty_list():
    """DB with no RCCL spans at all -> {collectives: [], capture_incomplete: False}."""
    path = _require(_FIXTURES / "rccl_empty.db")
    r = rccl_analysis.analyze_collectives(path, gfx_id="gfx942")
    assert r["collectives"] == []
    assert r["summary"]["capture_incomplete"] is False
    assert r["summary"]["op_count"] == 0


def test_analyze_collectives_computes_busbw_allreduce():
    """4-rank AllReduce, 1 MiB, 1 ms -> busBW = 1MiB * 2*3/4 / 1e-3 ~ 1.57 GB/s."""
    path = _require(_FIXTURES / "rccl_allreduce.db")
    r = rccl_analysis.analyze_collectives(path, gfx_id="gfx942")
    assert len(r["collectives"]) == 4
    first = r["collectives"][0]
    assert first["op_type"] == "AllReduce"
    assert first["ranks"] == 4
    # 1 MiB = 1048576, factor 2*(4-1)/4 = 1.5, duration 1 ms = 1e-3 s.
    # busBW = 1048576 * 1.5 / 1e-3 / 1e9 = 1.572864 GB/s
    expected = 1048576 * 1.5 / 1e-3 / 1e9
    assert abs(first["effective_bw_gbps"] - expected) < 0.01
    assert first["peak_bw_gbps"] == 340.0   # achievable RCCL baseline for MI300X
    # 4 collectives, dominant AllReduce.
    assert r["summary"]["dominant_op"] == "AllReduce"
    assert r["summary"]["op_count"] == 4
    assert r["summary"]["capture_incomplete"] is False


def test_analyze_collectives_fallback_regex_marks_incomplete():
    """DB with kernel names matching RCCL regex but no category='RCCL'."""
    path = _require(_FIXTURES / "rccl_fallback.db")
    r = rccl_analysis.analyze_collectives(path, gfx_id="gfx942")
    assert r["summary"]["capture_incomplete"] is True
    assert len(r["collectives"]) >= 1
    # msg_bytes is zero because the fallback has no arg binding
    assert all(c["msg_bytes"] == 0 for c in r["collectives"])


def test_analyze_collectives_is_read_only_class():
    assert (
        rccl_analysis.analyze_collectives.__tool_class__ == ToolClass.READ_ONLY
    )


# --------------------------------------------------------------------------- #
# interconnect.lookup_peaks                                                   #
# --------------------------------------------------------------------------- #

def test_interconnect_lookup_peaks_returns_dict_per_arch():
    """gfx90a / gfx942 / gfx950 each return all required fields."""
    required = {
        "name", "xgmi_peak_gbps", "xgmi_links", "xgmi_per_link_gbps",
        "pcie_tier", "pcie_peak_gbps", "achievable_gbps", "source_url",
        "measured_with",
    }
    for gfx in ("gfx90a", "gfx942", "gfx950"):
        entry = interconnect.lookup_peaks(gfx)
        missing = required - set(entry)
        assert not missing, f"{gfx} missing fields: {missing}"


def test_interconnect_lookup_peaks_uses_public_amd_link_specs():
    mi250x = interconnect.lookup_peaks("gfx90a")
    assert mi250x["xgmi_links"] == 8
    assert mi250x["xgmi_per_link_gbps"] == pytest.approx(100.0)
    assert mi250x["xgmi_peak_gbps"] == pytest.approx(800.0)

    mi300x = interconnect.lookup_peaks("gfx942")
    assert mi300x["xgmi_links"] == 8
    assert mi300x["xgmi_per_link_gbps"] == pytest.approx(128.0)
    assert mi300x["xgmi_peak_gbps"] == pytest.approx(1024.0)

    mi350x = interconnect.lookup_peaks("gfx950")
    assert mi350x["xgmi_links"] == 7
    assert mi350x["xgmi_per_link_gbps"] == pytest.approx(153.0)
    assert mi350x["xgmi_peak_gbps"] == pytest.approx(1071.0)


def test_interconnect_lookup_peaks_unknown_raises():
    with pytest.raises(KeyError):
        interconnect.lookup_peaks("gfx9999")


def test_interconnect_lookup_peaks_is_read_only_class():
    assert interconnect.lookup_peaks.__tool_class__ == ToolClass.READ_ONLY


def test_interconnect_lookup_peaks_includes_nic_entries():
    """ConnectX-7 carried as a first-class entry for inter-node analysis."""
    cx7 = interconnect.lookup_peaks("connectx7")
    assert "400" in cx7["name"] or "ConnectX-7" in cx7["name"]
    assert cx7["achievable_gbps"] > 0


# --------------------------------------------------------------------------- #
# MCP registry                                                                #
# --------------------------------------------------------------------------- #

def test_rccl_and_interconnect_exposed_via_mcp():
    """Both Phase-10 tools discovered by the MCP read-only registry."""
    from mcp_server._registry import discover_read_only_tools
    reg = discover_read_only_tools()
    assert "rccl_analysis.analyze_collectives" in reg
    assert "interconnect.lookup_peaks" in reg
