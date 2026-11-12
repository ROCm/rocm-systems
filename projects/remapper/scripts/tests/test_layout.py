import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "att_merge"))
import layout


def traced(pass_index, run, sweep, simd, accepted=True, attempt=0):
    return dict(
        pass_=pass_index,
        run=run,
        sweep=sweep,
        attempt=attempt,
        accepted=accepted,
        trace=dict(target_cu=0, simd=simd, shader_engine_mask=255),
    )


def setup(tmp_path, records, decoded, simd_override=None):
    dispatch = tmp_path / "dispatch-7"
    dispatch.mkdir()
    passes = [dict(pass_=0, stage="record", accepted=True)] + records
    for p in passes:
        p["pass"] = p.pop("pass_")
    (dispatch / "passes.json").write_text(
        json.dumps(dict(dispatch_id=7, passes=passes))
    )
    out = tmp_path / "rocprof" / "host" / "1234"
    for p in passes:
        if p.get("run") and p["pass"] in decoded:
            ui = out / f"ui_output_agent_5_dispatch_7_pass_{p['pass']}"
            ui.mkdir(parents=True)
            simd = p["trace"]["simd"] if simd_override is None else simd_override
            (ui / "replay_target.json").write_text(
                json.dumps(dict(target_cu=0, simd_select=simd))
            )
    # Another dispatch's passes are ignored.
    (out / "ui_output_agent_5_dispatch_8_pass_1").mkdir(parents=True)
    return dispatch, tmp_path / "rocprof"


def test_accepted_passes_become_cells_and_drifted_ones_are_set_aside(tmp_path):
    records = [
        traced(1, "table", 0, 0, accepted=False),
        traced(2, "table", 0, 0, attempt=1),
        traced(3, "table", 0, 1),
        traced(4, "original", 0, 0),
    ]
    dispatch, rocprof = setup(tmp_path, records, {1, 2, 3, 4})
    runs = layout.arrange(dispatch, rocprof)
    assert set(runs) == {"table", "original"}
    cell = (
        runs["table"]
        / "sweep-001"
        / "cu-00-simd-0-attempt-0"
        / "ui_output_agent_5_dispatch_1"
    )
    assert (
        cell.is_symlink()
        and cell.resolve().name == "ui_output_agent_5_dispatch_7_pass_2"
    )
    assert (
        runs["table"]
        / "sweep-001"
        / "cu-00-simd-1-attempt-0"
        / "ui_output_agent_5_dispatch_1"
    ).exists()
    rejected = (
        runs["table"]
        / "rejected-drift"
        / "sweep-001-cu-00-simd-0-try-0"
        / "ui_output_agent_5_dispatch_1"
    )
    assert rejected.resolve().name == "ui_output_agent_5_dispatch_7_pass_1"
    # Arranging twice replaces the links.
    layout.arrange(dispatch, rocprof)


def test_a_missing_or_mistargeted_pass_fails(tmp_path):
    for d in ("a", "b"):
        (tmp_path / d).mkdir()
    dispatch, rocprof = setup(tmp_path / "a", [traced(1, "table", 0, 0)], set())
    with pytest.raises(ValueError, match="no decoded output"):
        layout.arrange(dispatch, rocprof)
    dispatch, rocprof = setup(
        tmp_path / "b", [traced(1, "table", 0, 2)], {1}, simd_override=3
    )
    with pytest.raises(ValueError, match="planned cu=0 simd=2"):
        layout.arrange(dispatch, rocprof)


def test_merge_refuses_a_failed_remap(tmp_path):
    (tmp_path / "report.json").write_text(
        json.dumps(dict(status="failed", reason="placement drifted"))
    )
    with pytest.raises(ValueError, match="placement drifted"):
        layout.merge(tmp_path, {"table": tmp_path})
