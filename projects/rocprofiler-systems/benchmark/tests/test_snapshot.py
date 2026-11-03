# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import copy
import json

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
