# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
import shlex
import sys

import pytest
from bench import cli, scenario
from bench.scenario import ScenarioError


def write(tmp_path, doc):
    path = tmp_path / "s.json"
    path.write_text(json.dumps(doc) if isinstance(doc, dict) else doc)
    return str(path)


def valid(**over):
    doc = {
        "tools": [
            {
                "name": "v1",
                "tool": "rocprof-sys-run",
                "tool_env": [{"key": "ROCPROFSYS_USE_SAMPLER", "value": "true"}],
                "tool_arguments": [
                    {"arg_name": "--log-level", "value": "info"},
                    {"arg_name": "--verbose"},
                ],
            },
            {"name": "v2", "tool": "./build/bin/rocprof-sys-run"},
        ],
        "workload": "../workloads/build/cpu_compute",
        "workload_arguments": [{"arg_name": "--size", "value": "64"}],
        "run": {"mode": "light", "warmup": 1, "clean": ["out"]},
    }
    doc.update(over)
    return doc


def test_valid_file_loads(tmp_path):
    assert scenario.load(write(tmp_path, valid()))["workload"].endswith("cpu_compute")


def test_tool_reference_with_two_tools_loads(tmp_path):
    assert scenario.load(write(tmp_path, valid(reference="v1")))["reference"] == "v1"


@pytest.mark.parametrize(
    "doc,msg",
    [
        ({"tools": [{"name": "a", "tool": "t"}]}, "'workload' is a required property"),
        (
            valid(workload_arguments=[{"arg_name": "size"}]),
            "at workload_arguments/0/arg_name",
        ),
        (valid(tools=[]), "at tools"),
        (valid(tools=[{"name": "a b", "tool": "t"}]), "at tools/0/name"),
        (valid(run={"runs": 0}), "at run/runs"),
        (valid(run={"bogus": 1}), "at run"),
        (valid(tool="old-single-tool-field"), "Additional properties"),
        (
            valid(tools=[{"name": "a", "tool": "t"}, {"name": "a", "tool": "u"}]),
            "duplicate tool name",
        ),
        (valid(tools=[{"name": "baseline", "tool": "t"}]), "reserved"),
        (valid(reference="v9"), "reference 'v9'"),
        (valid(reference="a", tools=[{"name": "a", "tool": "t"}]), "at least 2 tools"),
    ],
)
def test_invalid_rejected_with_message(tmp_path, doc, msg):
    with pytest.raises(ScenarioError, match=msg):
        scenario.load(write(tmp_path, doc))


def test_bad_json_and_missing_file(tmp_path):
    with pytest.raises(ScenarioError):
        scenario.load(write(tmp_path, "{not json"))
    with pytest.raises(ScenarioError):
        scenario.load(str(tmp_path / "missing.json"))


def test_resolve_path():
    assert scenario.resolve_path("rocprof-sys-run", "/s") == "rocprof-sys-run"
    assert scenario.resolve_path("../w/app", "/s/dir") == "/s/w/app"
    assert scenario.resolve_path("/abs/app", "/s") == "/abs/app"


def test_workload_reference_puts_baseline_first_with_workload_env_only():
    doc = valid(
        workload_env=[{"key": "W", "value": "1"}, {"key": "SHARED", "value": "w"}]
    )
    doc["tools"][0]["tool_env"].append({"key": "SHARED", "value": "t"})
    cmds = scenario.build_commands(doc, "/s/dir")
    assert [c[0] for c in cmds] == ["baseline", "v1", "v2"]
    _name, command, env = cmds[0]
    assert command == "/s/workloads/build/cpu_compute --size=64"
    assert env == {"W": "1", "SHARED": "w"}
    _, command, env = cmds[1]
    assert command == (
        "rocprof-sys-run --log-level=info --verbose -- "
        "/s/workloads/build/cpu_compute --size=64"
    )
    assert env == {"W": "1", "SHARED": "t", "ROCPROFSYS_USE_SAMPLER": "true"}
    assert cmds[2][1].startswith("/s/dir/build/bin/rocprof-sys-run -- ")
    assert cmds[2][2] == {"W": "1", "SHARED": "w"}


def test_tool_reference_runs_reference_first_and_no_baseline():
    cmds = scenario.build_commands(valid(reference="v2"), "/s")
    assert [c[0] for c in cmds] == ["v2", "v1"]


def test_argument_values_with_spaces_survive_shlex_round_trip():
    doc = valid(workload_arguments=[{"arg_name": "--name", "value": "a b"}])
    assert shlex.split(scenario.build_commands(doc, "/s")[0][1])[-1] == "--name=a b"


def options(argv, run=None, base="/s"):
    args = cli.build_scenario_parser().parse_args(["f.json", *argv])
    return cli.run_options(args, run, base)


def test_options_defaults_json_then_cli():
    assert options([])["mode"] == "medium" and options([])["warmup"] == 3
    run = {"mode": "light", "warmup": 1, "clean": ["out", "/abs"], "output": "r.json"}
    o = options([], run)
    assert (o["mode"], o["warmup"], o["clean"], o["output"]) == (
        "light",
        1,
        ["/s/out", "/abs"],
        "/s/r.json",
    )
    o = options(["--warmup", "0", "--clean", "x", "--output", "cli.json"], run)
    assert (o["warmup"], o["clean"], o["output"]) == (0, ["x"], "cli.json")


def scenario_file(tmp_path, **over):
    doc = {
        "tools": [
            {"name": "v1", "tool": "env", "tool_env": [{"key": "T", "value": "1"}]},
            {"name": "v2", "tool": "env"},
        ],
        "workload": "true",
        "workload_arguments": [{"arg_name": "--ignored"}],
        "run": {"runs": 3, "warmup": 0, "pause_ms": 0, "output": "out/r.json"},
    }
    doc.update(over)
    (tmp_path / "out").mkdir()
    return write(tmp_path, doc)


def test_end_to_end_workload_reference(tmp_path, capsys):
    rc = cli.scenario_main([scenario_file(tmp_path)])
    doc = json.loads((tmp_path / "out" / "r.json").read_text())
    assert rc == 0
    assert list(doc["commands"]) == ["baseline", "v1", "v2"]
    assert doc["commands"]["v1"]["command"] == ["env", "--", "true", "--ignored"]
    assert doc["commands"]["v1"]["env"] == {"T": "1"}
    assert doc["meta"]["reference"] == "baseline" and doc["meta"]["repetitions"] == 3
    assert doc["meta"]["scenario"].endswith("s.json")
    assert "v1_vs_baseline" in doc["comparison"]


def test_end_to_end_tool_reference(tmp_path, capsys):
    rc = cli.scenario_main([scenario_file(tmp_path, reference="v2"), "--runs", "2"])
    doc = json.loads((tmp_path / "out" / "r.json").read_text())
    assert rc == 0
    assert list(doc["commands"]) == ["v2", "v1"]
    assert doc["meta"]["mode"] == "custom" and doc["meta"]["repetitions"] == 2
    assert "v1_vs_v2" in doc["comparison"]


def test_invalid_scenario_exit_code_2(tmp_path, capsys):
    assert cli.scenario_main([write(tmp_path, {"tools": []})]) == 2
    assert "bench-scenario: error:" in capsys.readouterr().err


def test_version_argv_defaults_to_tool_dash_dash_version_and_honours_override():
    assert scenario.version_argv({"name": "v1", "tool": "rocprof-sys-run"}, "/s") == [
        "rocprof-sys-run",
        "--version",
    ]
    custom = {"name": "v1", "tool": "t", "tool_version_cmd": "./bin/t -V --short"}
    assert scenario.version_argv(custom, "/s") == ["/s/bin/t", "-V", "--short"]


def test_probe_version_returns_first_line_and_none_on_failure(capsys):
    py = [sys.executable, "-c"]
    assert scenario.probe_version(py + ["print('tool 1.2\\nmore')"]) == "tool 1.2"
    assert scenario.probe_version(py + ["import sys; sys.exit(3)"]) is None
    assert "exited 3" in capsys.readouterr().err


def test_end_to_end_records_tool_versions(tmp_path, capsys):
    cli.scenario_main([scenario_file(tmp_path)])
    doc = json.loads((tmp_path / "out" / "r.json").read_text())
    assert set(doc["meta"]["tool_versions"]) == {"v1", "v2"}
