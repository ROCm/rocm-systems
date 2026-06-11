"""Tests for perfxpert.tools.predict_impact — Change-Impact Prediction (Phase 10)."""

from __future__ import annotations

import pytest

from perfxpert.tools import predict_impact
from perfxpert.tools._class import ToolClass


@pytest.fixture(autouse=True)
def _clear_prediction_store():
    """Ensure every test starts from an empty in-process store."""
    predict_impact._reset_store_for_tests()
    yield
    predict_impact._reset_store_for_tests()


# --------------------------------------------------------------------- #
# Happy path                                                            #
# --------------------------------------------------------------------- #


def test_readonly_sqlite_uri_escapes_path_metacharacters(tmp_path):
    db = tmp_path / "trace db#1.db"
    uri = predict_impact._readonly_sqlite_uri(str(db))
    assert uri.startswith("file:")
    assert uri.endswith("?mode=ro")
    assert "trace%20db%231.db" in uri


def test_baseline_has_counters_empty_path_returns_false():
    assert predict_impact._baseline_has_counters("") is False


def test_predict_change_impact_vgpr_reduction_returns_bracket(tmp_path):
    """Fake DB path; bypass DB probes via change_params. Assert bracket > 0."""
    result = predict_impact.predict_change_impact(
        baseline_db=str(tmp_path / "fake.db"),
        kernel_name="hot_kernel",
        change_type="vgpr_reduction",
        change_params={
            "kernel_time_pct": 0.40,
            "counter_data_available": True,
        },
    )
    assert result["predicted_speedup_range"] is not None
    lo, hi = result["predicted_speedup_range"]
    assert lo < hi
    assert result["confidence"] > 0.0
    assert "vgpr_reduction" in result["rationale"]
    assert "source_citation" in result and result["source_citation"]


def test_predict_change_impact_amdahl_guard_returns_zero_confidence(tmp_path):
    """Kernel contributes 2% of runtime — Amdahl guard fires."""
    result = predict_impact.predict_change_impact(
        baseline_db=str(tmp_path / "fake.db"),
        kernel_name="tiny_kernel",
        change_type="vgpr_reduction",
        change_params={"kernel_time_pct": 0.02, "counter_data_available": True},
    )
    assert result["confidence"] == 0.0
    assert result["predicted_speedup_range"] is None
    assert "Amdahl" in result["rationale"] or "amdahl" in result["rationale"].lower()


def test_predict_change_impact_tier1_no_counters_returns_zero_confidence(tmp_path):
    """Tier-1 (no counters) must return zero-confidence + tier-2 rationale."""
    fake_db = tmp_path / "trace_only.db"
    fake_db.write_bytes(b"")  # zero-byte file triggers sqlite open failure branch
    result = predict_impact.predict_change_impact(
        baseline_db=str(fake_db),
        kernel_name="kernel_a",
        change_type="vgpr_reduction",
        change_params={
            "kernel_time_pct": 0.40,
            "counter_data_available": False,
        },
    )
    assert result["confidence"] == 0.0
    assert result["predicted_speedup_range"] is None
    assert "--pmc" in result["rationale"] or "counter" in result["rationale"].lower()


def test_predict_change_impact_unknown_technique_returns_null_range(tmp_path):
    result = predict_impact.predict_change_impact(
        baseline_db=str(tmp_path / "fake.db"),
        kernel_name="any",
        change_type="unicorn_magic",
        change_params={"kernel_time_pct": 0.40, "counter_data_available": True},
    )
    assert result["predicted_speedup_range"] is None
    assert result["confidence"] == 0.0
    assert "unicorn_magic" in result["rationale"]


def test_predict_change_impact_conservative_bracket_is_0_85_of_catalog_hi(tmp_path):
    """Conservative bracket: emitted hi = catalog_hi × 0.85 (rounded)."""
    catalog_entries = predict_impact.list_supported_changes()
    vgpr_entry = next(e for e in catalog_entries if e["id"] == "vgpr_reduction")
    # catalog hi is 1.70 per the seed YAML
    result = predict_impact.predict_change_impact(
        baseline_db=str(tmp_path / "fake.db"),
        kernel_name="kernel_a",
        change_type="vgpr_reduction",
        change_params={"kernel_time_pct": 0.40, "counter_data_available": True},
    )
    _lo, hi = result["predicted_speedup_range"]
    # Catalog hi = 1.70 × 0.85 = 1.445 → rounded to 1.445 in code (3 dp).
    expected_hi = round(1.70 * 0.85, 3)
    assert abs(hi - expected_hi) < 1e-6
    # Sanity on ignored-but-present vgpr_entry
    assert "required_metrics" in vgpr_entry


# --------------------------------------------------------------------- #
# list_supported_changes                                                #
# --------------------------------------------------------------------- #


def test_list_supported_changes_returns_5():
    entries = predict_impact.list_supported_changes()
    assert len(entries) == 5
    ids = {e["id"] for e in entries}
    assert ids == {
        "vgpr_reduction",
        "lds_tiling",
        "mfma_enablement",
        "fast_math_flag",
        "hip_stream_overlap",
    }
    for e in entries:
        assert isinstance(e["id"], str)
        assert isinstance(e["applies_to"], dict)
        assert isinstance(e["required_metrics"], list)


# --------------------------------------------------------------------- #
# explain_prediction                                                    #
# --------------------------------------------------------------------- #


def test_explain_prediction_roundtrip(tmp_path):
    result = predict_impact.predict_change_impact(
        baseline_db=str(tmp_path / "fake.db"),
        kernel_name="kernel_x",
        change_type="hip_stream_overlap",
        change_params={"kernel_time_pct": 0.30, "counter_data_available": True},
    )
    pid = result["prediction_id"]
    rehydrated = predict_impact.explain_prediction(pid)
    assert rehydrated == result


def test_explain_prediction_unknown_raises():
    with pytest.raises(KeyError):
        predict_impact.explain_prediction("deadbeef00000000")


# --------------------------------------------------------------------- #
# Tool class metadata                                                   #
# --------------------------------------------------------------------- #


def test_all_three_tools_are_read_only():
    assert predict_impact.predict_change_impact.__tool_class__ == ToolClass.READ_ONLY
    assert predict_impact.list_supported_changes.__tool_class__ == ToolClass.READ_ONLY
    assert predict_impact.explain_prediction.__tool_class__ == ToolClass.READ_ONLY
