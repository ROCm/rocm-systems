# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import dataclasses
import json

import pytest
from bench import cli, runner


def parse(*extra):
    return cli.build_parser().parse_args(["--cmd", "A=true", *extra])


def config(args):
    return cli.make_config(cli.run_options(args))[0]


@pytest.mark.parametrize(
    "extra,runs",
    [
        ([], 100),
        (["--mode", "light"], 30),
        (["--mode", "precise"], 300),
        (["--mode", "light", "--runs", "7"], 7),
    ],
)
def test_runs_resolution(extra, runs):
    assert config(parse(*extra)).runs == runs


@pytest.mark.parametrize("value", ["0", "-1", "x"])
def test_runs_below_one_rejected(value):
    with pytest.raises(SystemExit):
        parse("--runs", value)


def test_default_warmup_and_cmd_split():
    args = cli.build_parser().parse_args(["--cmd", "A=x --k=v", "--cmd", "B=y"])
    assert args.cmd == [("A", "x --k=v"), ("B", "y")]
    assert config(args).warmup == 3


def test_bad_cmd_rejected():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["--cmd", "novalue"])


def test_end_to_end_json(tmp_path, capsys):
    out = tmp_path / "r.json"
    rc = cli.main(
        [
            "--runs",
            "5",
            "--warmup",
            "1",
            "--pause",
            "0",
            "--output",
            str(out),
            "--cmd",
            "A=true",
            "--cmd",
            "B=sleep 0.02",
        ]
    )
    doc = json.loads(out.read_text())
    assert rc == 0
    assert doc["meta"]["repetitions"] == 5
    assert len(doc["commands"]["A"]["runs"]) == 5
    run_keys = {f.name for f in dataclasses.fields(runner.RunResult)} - {"name", "index"}
    assert set(doc["commands"]["A"]["runs"][0]) == run_keys
    assert doc["comparison"]["B_vs_A"]["resolvable_from_noise"]
    assert "WALL-CLOCK TIME" in capsys.readouterr().out


def test_failures_set_exit_code(capsys):
    rc = cli.main(["--runs", "2", "--warmup", "0", "--cmd", "A=false"])
    assert rc == 1


def test_all_runs_of_a_command_failing_does_not_crash(capsys):
    rc = cli.main(
        [
            "--runs",
            "3",
            "--warmup",
            "0",
            "--pause",
            "0",
            "--cmd",
            "A=true",
            "--cmd",
            "B=does_not_exist_xyz",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "3 FAILED" in captured.err
    assert "'B' has no successful runs" in captured.out
    assert "B: 3 run(s) failed" in captured.out
    assert "WALL-CLOCK TIME" in captured.out


def test_progress_header_and_status_lines_on_stderr(capsys):
    cli.main(
        [
            "--runs",
            "2",
            "--warmup",
            "0",
            "--pause",
            "0",
            "--cmd",
            "A=true",
            "--cmd",
            "B=true",
        ]
    )
    err = capsys.readouterr().err
    assert "══ RUNNING" in err
    assert "baseline" in err and "tool" in err
    assert "2/2 runs  done" in err


def test_run_session_records_env_per_command(tmp_path, capsys):
    out = tmp_path / "r.json"
    cfg = runner.RunConfig(runs=2, warmup=0, pause_ms=0)
    rc, _ = cli.run_session(
        [("A", "true"), ("B", "sh -c 'test \"$X\" = y'")],
        cfg,
        {
            "mode": "custom",
            "repetitions": 2,
            "warmup": 0,
            "cpu_affinity": None,
            "mem_sample_ms": None,
        },
        str(out),
        {"B": {"X": "y"}},
    )
    doc = json.loads(out.read_text())
    assert rc == 0
    assert doc["commands"]["A"]["env"] == {}
    assert doc["commands"]["B"]["env"] == {"X": "y"}


def test_mode_flag_overrides_runs_from_run_block():
    run_block = {"mode": "medium", "runs": 150}
    args = parse("--mode", "light")
    assert cli.make_config(cli.run_options(args, run_block))[0].runs == 30


@pytest.mark.parametrize(
    "flag,value",
    [("--warmup", "-1"), ("--pause", "-1"), ("--timeout", "0"), ("--mem-sample-ms", "0")],
)
def test_out_of_range_run_options_rejected(flag, value):
    with pytest.raises(SystemExit):
        parse(flag, value)
