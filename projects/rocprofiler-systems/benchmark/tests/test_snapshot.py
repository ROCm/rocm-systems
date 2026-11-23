# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import copy
import json
import math
from typing import Any

import pytest
from bench import cli, snapshot


def scenario_file(tmp_path, name, workload="true"):
    path = tmp_path / f"{name}.json"
    path.write_text(
        json.dumps(
            {
                "tools": [{"name": "v1", "tool": "env"}],
                "workload": workload,
                "run": {"runs": 3, "warmup": 0, "pause_ms": 0},
            }
        )
    )
    return str(path)


def take(tmp_path, *names):
    files = [scenario_file(tmp_path, n) for n in names]
    assert cli.scenario_main([*files, "--snapshot-dir", str(tmp_path / "snaps")]) == 0
    (directory,) = (tmp_path / "snaps").iterdir()
    return directory


def test_snapshot_dir_layout_and_content(tmp_path, capsys):
    directory = take(tmp_path, "a", "b")
    doc = json.loads((directory / "snapshot.json").read_text())
    assert directory.name.startswith("snapshot-")
    assert sorted(p.name for p in (directory / "results").iterdir()) == [
        "a.json",
        "b.json",
    ]
    assert [s["name"] for s in doc["scenarios"]] == ["a", "b"]
    assert doc["scenarios"][0]["result"] == "results/a.json"
    assert {
        "cpu",
        "memory_total_kb",
        "kernel",
        "gpus",
        "rocm_version",
        "benchmark_suite",
    } <= set(doc["system"])
    assert doc["created"] and "v1" in doc["scenarios"][0]["tool_versions"]


def test_several_scenarios_need_snapshot_dir(tmp_path, capsys):
    with pytest.raises(SystemExit):
        cli.scenario_main([scenario_file(tmp_path, "a"), scenario_file(tmp_path, "b")])


def test_invalid_scenario_aborts_before_any_run(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    rc = cli.scenario_main(
        [
            scenario_file(tmp_path, "a"),
            str(bad),
            "--snapshot-dir",
            str(tmp_path / "snaps"),
        ]
    )
    assert rc == 2 and not (tmp_path / "snaps").exists()


def slow_down(doc, factor):
    doc = copy.deepcopy(doc)
    for sc in doc["scenarios"]:
        for stats in sc["summary"]["v1"].values():
            stats["median"] *= factor
    return doc


def test_compare_identical_is_noise_and_hardware_diff_is_listed(tmp_path, capsys):
    a = json.loads((take(tmp_path, "a") / "snapshot.json").read_text())
    b = copy.deepcopy(a)
    b["system"]["kernel"] = "other"
    cmp = snapshot.compare_snapshots(a, b)
    assert cmp["system"] == [("kernel", a["system"]["kernel"], "other")]
    assert not snapshot.has_regression(cmp)


def test_compare_flags_real_slowdown_and_ignores_speedup(tmp_path, capsys):
    a = json.loads((take(tmp_path, "a") / "snapshot.json").read_text())
    for sc in a["scenarios"]:  # floor out the noise so only the factor matters
        for stats in sc["summary"]["v1"].values():
            stats["median"] = max(stats["median"], 1.0)
    assert snapshot.has_regression(snapshot.compare_snapshots(a, slow_down(a, 2.0)))
    assert not snapshot.has_regression(snapshot.compare_snapshots(a, slow_down(a, 0.5)))


def test_compare_flags_failed_scenario_and_vanished_command(tmp_path, capsys):
    a = json.loads((take(tmp_path, "a") / "snapshot.json").read_text())
    failed = copy.deepcopy(a)
    failed["scenarios"][0]["exit"] = 1
    vanished = copy.deepcopy(a)
    vanished["scenarios"][0]["summary"]["v1"] = {}
    for b in (failed, vanished):
        cmp = snapshot.compare_snapshots(a, b)
        assert cmp["failed_in_b"] == ["a"] and snapshot.has_regression(cmp)
        assert "failed runs" in snapshot.format_comparison(a, b, cmp)
    assert not snapshot.compare_snapshots(a, a)["failed_in_b"]


def snap(reference: str, **wall_ms: float) -> dict[str, Any]:
    """Snapshot with one scenario 's'; wall_ms: command name -> wall median in ms."""
    summary = {
        cmd: {"wall": {"median": ms * 1e6, "mad": 2e6, "n": 150}}
        for cmd, ms in wall_ms.items()
    }
    entry = {
        "name": "s",
        "exit": 0,
        "reference": reference,
        "tool_versions": {},
        "summary": summary,
    }
    return {"created": "x", "system": {}, "scenarios": [entry]}


@pytest.mark.parametrize(
    "b,regression",
    [
        pytest.param(
            snap("baseline", baseline=1030, tool=1133),
            False,
            id="machine-3pct-slower-overhead-unchanged",
        ),
        pytest.param(
            snap("baseline", baseline=950, tool=1100),
            True,
            id="machine-5pct-faster-overhead-100ms-to-150ms",
        ),
        pytest.param(
            snap("baseline", baseline=1500, tool=1600),
            False,
            id="only-the-reference-got-slower",
        ),
    ],
)
def test_compare_judges_overhead_not_absolute_time(
    b: dict[str, Any], regression: bool
) -> None:
    a = snap("baseline", baseline=1000, tool=1100)
    assert snapshot.has_regression(snapshot.compare_snapshots(a, b)) is regression


def test_comparison_shows_overhead_and_marks_the_reference() -> None:
    a = snap("baseline", baseline=1000, tool=1100)
    b = snap("baseline", baseline=950, tool=1100)
    text = snapshot.format_comparison(a, b, snapshot.compare_snapshots(a, b))
    ref_row, tool_row = [line for line in text.splitlines() if " wall " in line]
    assert "1000.00 ms" in ref_row and "950.00 ms" in ref_row
    assert ref_row.endswith("reference moved")
    assert "100.00 ms" in tool_row and "150.00 ms" in tool_row
    assert tool_row.endswith("REGRESSION")


@pytest.mark.parametrize(
    "overhead_a_ms,pct",
    [
        pytest.param(92.27, 130.9, id="percent-of-the-overhead-in-a"),
        pytest.param(0.5, None, id="overhead-in-a-below-1pct-of-the-median"),
        pytest.param(-30.0, None, id="negative-overhead-in-a"),
    ],
)
def test_comparison_pct_is_relative_to_the_overhead_in_a(
    overhead_a_ms: float, pct: float | None
) -> None:
    reference_ms, added_ms = 2372.73, 120.77
    a = snap("baseline", baseline=reference_ms, tool=reference_ms + overhead_a_ms)
    b = snap(
        "baseline", baseline=reference_ms, tool=reference_ms + overhead_a_ms + added_ms
    )
    rows = snapshot.compare_snapshots(a, b)["scenarios"]["s"]
    (row,) = [r for r in rows if r["command"] == "tool"]
    if pct is None:
        assert math.isnan(row["pct"])
    else:
        assert row["pct"] == pytest.approx(pct, abs=0.05)


def test_compare_rejects_a_changed_reference() -> None:
    a = snap("baseline", baseline=1000, tool=1100)
    b = snap("tool", baseline=1000, tool=1100)
    with pytest.raises(ValueError, match="reference"):
        snapshot.compare_snapshots(a, b)


def test_comparison_shows_run_counts_and_warns_about_too_few_runs() -> None:
    a = snap("baseline", baseline=1000, tool=1100)
    b = copy.deepcopy(a)
    for stats in b["scenarios"][0]["summary"].values():
        stats["wall"]["n"] = 3
    text = snapshot.format_comparison(a, b, snapshot.compare_snapshots(a, b))
    assert "SCENARIO s  (runs per command: A 150, B 3)" in text
    assert "fewer than 20 runs per command (verdicts are rough): s" in text
    plenty = snapshot.format_comparison(a, a, snapshot.compare_snapshots(a, a))
    assert "fewer than" not in plenty


def test_long_command_names_do_not_run_into_the_metric_column() -> None:
    a = snap("baseline", baseline=1000, only_sampling_on=1100)
    text = snapshot.format_comparison(a, a, snapshot.compare_snapshots(a, a))
    assert "  only_sampling_on  wall " in text


def test_snapshot_records_a_settings_hash_per_scenario(tmp_path, capsys):
    doc = json.loads((take(tmp_path, "a") / "snapshot.json").read_text())
    settings_hash = doc["scenarios"][0]["settings_hash"]
    assert len(settings_hash) == 12 and int(settings_hash, 16) >= 0


def result_doc(
    workload: str = "/home/u/build/app",
    env: dict[str, str] | None = None,
    args: tuple[str, ...] = (),
) -> dict[str, Any]:
    command = ["rocprof-sys-run", "--", workload, *args]
    settings = {"ROCPROFSYS_TRACE": "true"} if env is None else env
    return {"commands": {"tool": {"command": command, "env": settings}}}


def test_settings_hash_ignores_the_checkout_location_but_not_the_settings() -> None:
    reference = snapshot.settings_hash(result_doc())
    assert snapshot.settings_hash(result_doc(workload="/srv/ci/build/app")) == reference
    assert (
        snapshot.settings_hash(result_doc(env={"ROCPROFSYS_TRACE": "false"})) != reference
    )
    assert snapshot.settings_hash(result_doc(args=("--size=512",))) != reference


def test_compare_shows_but_does_not_judge_a_scenario_whose_settings_changed() -> None:
    a = snap("baseline", baseline=1000, tool=1100)
    b = snap("baseline", baseline=1000, tool=1500)
    a["scenarios"][0]["settings_hash"] = "aaa"
    b["scenarios"][0]["settings_hash"] = "bbb"
    cmp = snapshot.compare_snapshots(a, b)
    assert cmp["settings_changed"] == ["s"] and not snapshot.has_regression(cmp)
    text = snapshot.format_comparison(a, b, cmp)
    assert [line for line in text.splitlines() if " wall " in line][1].endswith(
        "settings changed"
    )
    b["scenarios"][0]["settings_hash"] = "aaa"
    assert snapshot.has_regression(snapshot.compare_snapshots(a, b))


def test_cli_compare_exit_codes(tmp_path, capsys):
    directory = take(tmp_path, "a")
    assert (
        cli.snapshot_main(["compare", str(directory), str(directory / "snapshot.json")])
        == 0
    )
    assert "SCENARIO a" in capsys.readouterr().out
    assert cli.snapshot_main(["compare", str(directory), str(tmp_path / "missing")]) == 2


def test_suite_file_runs_listed_scenarios_into_snapshot(tmp_path, capsys):
    scenario_file(tmp_path, "a")
    scenario_file(tmp_path, "b")
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"scenarios": ["a.json", "b.json"]}))
    assert cli.snapshot_main([str(suite)]) == 0
    (directory,) = (tmp_path / "results").iterdir()
    assert sorted(p.name for p in (directory / "results").iterdir()) == [
        "a.json",
        "b.json",
    ]


def test_bad_suite_rejected(tmp_path, capsys):
    suite = tmp_path / "suite.json"
    suite.write_text("{}")
    assert cli.snapshot_main([str(suite)]) == 2
