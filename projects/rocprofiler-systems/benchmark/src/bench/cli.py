# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Command-line entry points: bench, bench-scenario, bench-snapshot.

All argument parsing lives here; scenario and snapshot are plain libraries.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from bench import analysis, progress, report, scenario, snapshot, system, text_report
from bench.runner import RunConfig, RunResult, run_command

MODES = {"light": 30, "medium": 100, "precise": 300}
# Keys equal the scenario "run" block keys. Precedence: CLI flag > run block > these defaults.
RUN_DEFAULTS: dict[str, Any] = {
    "mode": "medium",
    "runs": None,
    "warmup": 3,
    "timeout": None,
    "pause_ms": 10.0,
    "cpu_affinity": None,
    "mem_sample_ms": None,
    "clean": [],
    "output": None,
}


def parse_cmd(value: str) -> tuple[str, str]:
    name, sep, command = value.partition("=")
    if not sep or not name or not command.strip():
        raise argparse.ArgumentTypeError(f"expected NAME=command, got {value!r}")
    return name, command


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {number}")
    return number


def non_negative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {number}")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {number}")
    return number


def non_negative_float(value: str) -> float:
    number = float(value)
    if number < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {number}")
    return number


def add_run_options(p: argparse.ArgumentParser) -> None:
    """Options shared by bench and bench-scenario; all default to None (= not given)."""
    p.add_argument("--mode", choices=MODES)
    p.add_argument("--runs", type=positive_int, help="overrides --mode")
    p.add_argument("--warmup", type=non_negative_int)
    p.add_argument("--timeout", type=positive_float)
    p.add_argument("--pause", dest="pause_ms", type=non_negative_float, metavar="MS")
    p.add_argument("--cpu-affinity", metavar="LIST")
    p.add_argument("--mem-sample-ms", type=positive_float)
    p.add_argument(
        "--clean",
        action="append",
        metavar="PATH",
        help="repeatable; replaces run.clean of a scenario",
    )
    p.add_argument("--output", metavar="FILE")


def run_options(
    args: argparse.Namespace, run: dict[str, Any] | None = None, base_dir: str = ""
) -> dict[str, Any]:
    """CLI flag > scenario run block > RUN_DEFAULTS. Run-block paths are relative to base_dir."""
    run = run or {}
    block = {**run, "clean": [os.path.join(base_dir, c) for c in run.get("clean", [])]}
    if "output" in run:
        block["output"] = os.path.join(base_dir, run["output"])
    flags = {k: getattr(args, k) for k in RUN_DEFAULTS if getattr(args, k) is not None}
    if "mode" in flags and "runs" not in flags:
        block.pop("runs", None)  # --mode must beat the block's runs, else it is ignored
    return {**RUN_DEFAULTS, **block, **flags}


def make_config(opts: dict[str, Any]) -> tuple[RunConfig, dict[str, Any]]:
    """RunConfig and the meta fields every result records."""
    runs = opts["runs"] if opts["runs"] is not None else MODES[opts["mode"]]
    cfg = RunConfig(
        runs=runs,
        warmup=opts["warmup"],
        timeout=opts["timeout"],
        pause_ms=opts["pause_ms"],
        cpu_affinity=opts["cpu_affinity"],
        mem_sample_ms=opts["mem_sample_ms"],
        clean=opts["clean"],
    )
    meta = {
        "mode": "custom" if opts["runs"] is not None else opts["mode"],
        "repetitions": cfg.runs,
        "warmup": cfg.warmup,
        "cpu_affinity": cfg.cpu_affinity,
        "mem_sample_ms": cfg.mem_sample_ms,
    }
    return cfg, meta


def _run_all(
    cmds: list[tuple[str, str]], cfg: RunConfig, envs: dict[str, dict[str, str]]
) -> dict[str, list[RunResult]]:
    progress.announce(cmds, cfg)
    name_w = max(len(n) for n, _ in cmds)
    results: dict[str, list[RunResult]] = {}
    for i, (name, command) in enumerate(cmds):
        role = progress.role_of(i)
        results[name] = []
        progress.warming_up(name, role, name_w, cfg)
        for result in run_command(name, command, cfg, envs.get(name)):
            results[name].append(result)
            progress.update(name, role, name_w, len(results[name]), cfg.runs)
        progress.finish(name, role, name_w, results[name], cfg.runs)
    return results


def _color_enabled() -> bool:
    """Color only for an interactive terminal; NO_COLOR (no-color.org) turns it off."""
    return sys.stdout.isatty() and "NO_COLOR" not in os.environ


def run_session(
    cmds: list[tuple[str, str]],
    cfg: RunConfig,
    meta: dict[str, Any],
    output: str | None = None,
    envs: dict[str, dict[str, str]] | None = None,
) -> tuple[int, dict[str, Any]]:
    """Run cmds (first = baseline), print the report, optionally write the JSON.

    envs: command name -> variables added to os.environ for that command only.
    Returns (exit code, JSON document).
    """
    envs = envs or {}
    results = _run_all(cmds, cfg, envs)
    doc = report.build_json(results, dict(cmds), analysis.analyze(results), meta, envs)
    print(text_report.format_text(doc, color=_color_enabled()))
    if output:
        report.write_json(doc, output)
    return (1 if doc["errors"] else 0), doc


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bench",
        description="Run commands N times each and "
        "compare them with the first one (the baseline).",
    )
    p.add_argument(
        "--cmd",
        action="append",
        type=parse_cmd,
        required=True,
        metavar='NAME="command"',
        help="first is the baseline",
    )
    add_run_options(p)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    names = [n for n, _ in args.cmd]
    if len(set(names)) != len(names):
        parser.error("duplicate --cmd names")
    opts = run_options(args)
    cfg, meta = make_config(opts)
    return run_session(args.cmd, cfg, meta, opts["output"])[0]


def build_scenario_parser() -> argparse.ArgumentParser:
    """Same options as bench (minus --cmd); unset flags fall back to the JSON run block."""
    p = argparse.ArgumentParser(
        prog="bench-scenario", description="Run one or more scenario JSON files."
    )
    p.add_argument(
        "scenarios", nargs="+", metavar="FILE", help="one or more scenario files"
    )
    p.add_argument(
        "--snapshot-dir",
        metavar="ROOT",
        help="run all scenarios into ROOT/snapshot-<date-time>/ (snapshot.json + "
        "results/); required for more than one scenario; replaces --output",
    )
    add_run_options(p)
    return p


def _run_scenario(
    path: str, doc: dict[str, Any], args: argparse.Namespace, output: str | None
) -> tuple[int, dict[str, Any]]:
    """Run one validated scenario; output overrides the scenario's own output path."""
    base_dir = os.path.dirname(os.path.abspath(path))
    opts = run_options(args, doc.get("run", {}), base_dir)
    cfg, meta = make_config(opts)
    commands = scenario.build_commands(doc, base_dir)
    meta.update(
        {
            "scenario": os.path.abspath(path),
            "reference": commands[0][0],
            "tool_versions": {
                t["name"]: scenario.probe_version(scenario.version_argv(t, base_dir))
                for t in doc["tools"]
            },
        }
    )
    return run_session(
        [(n, c) for n, c, _ in commands],
        cfg,
        meta,
        output or opts["output"],
        {n: e for n, _, e in commands},
    )


def scenario_main(argv: Sequence[str] | None = None) -> int:
    parser = build_scenario_parser()
    args = parser.parse_args(argv)
    if len(args.scenarios) > 1 and not args.snapshot_dir:
        parser.error("several scenarios need --snapshot-dir")
    if args.snapshot_dir and args.output:
        parser.error(
            "--output conflicts with --snapshot-dir (results go to <snapshot>/results/)"
        )
    stems = [Path(s).stem for s in args.scenarios]
    if len(set(stems)) != len(stems):
        parser.error("scenario file names (without extension) must be unique")
    docs = []
    for path in args.scenarios:  # validate all first: fail before any long run starts
        try:
            docs.append(scenario.load(path))
        except scenario.ScenarioError as exc:
            print(f"bench-scenario: error: {exc}", file=sys.stderr)
            return 2
    if not args.snapshot_dir:
        return _run_scenario(args.scenarios[0], docs[0], args, args.output)[0]
    now = datetime.now()
    machine = (
        system.collect()
    )  # before any result file exists, so the dirty flag is not self-inflicted
    directory = snapshot.new_snapshot_dir(args.snapshot_dir, now)
    exit_code, entries = 0, []
    for path, stem, doc in zip(args.scenarios, stems, docs):
        result_path = directory / snapshot.RESULTS_DIR / f"{stem}.json"
        rc, result = _run_scenario(path, doc, args, str(result_path))
        exit_code = max(exit_code, rc)
        entries.append(
            snapshot.scenario_entry(stem, os.path.abspath(path), result_path, rc, result)
        )
    print(
        f"snapshot written: {snapshot.write_snapshot(directory, now, machine, entries)}"
    )
    return exit_code


def _snapshot_compare(argv: Sequence[str]) -> int:
    p = argparse.ArgumentParser(prog="bench-snapshot compare")
    p.add_argument("a", metavar="A", help="snapshot directory or snapshot.json (older)")
    p.add_argument("b", metavar="B", help="snapshot directory or snapshot.json (newer)")
    args = p.parse_args(argv)
    try:
        a, b = snapshot.load_snapshot(args.a), snapshot.load_snapshot(args.b)
    except ValueError as exc:
        print(f"bench-snapshot: error: {exc}", file=sys.stderr)
        return 2
    cmp = snapshot.compare_snapshots(a, b)
    print(snapshot.format_comparison(a, b, cmp))
    return 1 if snapshot.has_regression(cmp) else 0


def snapshot_main(argv: Sequence[str] | None = None) -> int:
    """bench-snapshot SUITE.json [bench-scenario options]  |  bench-snapshot compare A B"""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(
            f"{snapshot.__doc__}\nusage: bench-snapshot SUITE.json [--snapshot-dir ROOT] "
            "[bench-scenario options]\n       bench-snapshot compare A B"
        )
        return 0 if argv else 2
    if argv[0] == "compare":
        return _snapshot_compare(argv[1:])
    try:
        scenarios = snapshot.load_suite(argv[0])
    except ValueError as exc:
        print(f"bench-snapshot: error: {exc}", file=sys.stderr)
        return 2
    default_root = str(Path(argv[0]).resolve().parent / "results")
    # user options come last so an explicit --snapshot-dir wins over the default
    return scenario_main([*scenarios, "--snapshot-dir", default_root, *argv[1:]])
