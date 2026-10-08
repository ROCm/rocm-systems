#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Run Catch2 quietly while preserving console diagnostics and JUnit results."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", required=True, type=Path)
    parser.add_argument(
        "--timeout",
        type=float,
        help="seconds before the suite and its workloads are killed",
    )
    parser.add_argument("--config", type=Path, help="suite YAML; decides which cases run")
    parser.add_argument("--os", dest="os_name", choices=("auto", "linux", "windows"),
                        help="OS the rules see; default auto when --config is set")
    parser.add_argument("--arch", help="auto, none, or comma-separated gfx names")
    parser.add_argument("--select", help="Catch2 selector; overrides the one in the config")
    parser.add_argument("--ignore-config", action="store_true",
                        help="run the cases the config would skip")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        parser.error("a Catch2 command is required after --")
    if args.config is None and (args.os_name or args.arch or args.select or args.ignore_config):
        parser.error("--os, --arch, --select and --ignore-config require --config")
    return args


# Same prefix hrr_test_config.skip_message writes. Repeated here because the
# report job checks this file out without the config module.
CONFIG_SKIP_PREFIX = "hrr-config:"


def is_config_skip(case: ET.Element) -> bool:
    skipped = case.find("skipped")
    if skipped is None:
        return False
    message = skipped.attrib.get("message", "") or (skipped.text or "")
    return message.startswith(CONFIG_SKIP_PREFIX)


def test_result(case: ET.Element) -> str:
    if case.find("failure") is not None or case.find("error") is not None:
        return "FAIL"
    if is_config_skip(case):
        return "CONFIG"
    if case.find("skipped") is not None:
        return "SKIP"
    return "PASS"


def group_cases(root: ET.Element) -> dict[str, list[ET.Element]]:
    """Group <testcase> records by top-level case, in order of first appearance.

    Catch2 writes one record per SECTION named "Case/Section", and an assertion
    that fails inside a section is attached to that record only. A case whose
    assertions all live in sections has no record under its own name at all.
    Counting only slash-free names would therefore lose both.
    """
    groups: dict[str, list[ET.Element]] = {}
    for case in root.iterfind(".//testcase"):
        name = case.attrib.get("name")
        if name:
            groups.setdefault(name.split("/", 1)[0], []).append(case)
    return groups


def group_result(records: list[ET.Element]) -> str:
    """A failing section fails its case; a config skip beats a code skip."""
    results = {test_result(record) for record in records}
    for result in ("FAIL", "CONFIG", "SKIP"):
        if result in results:
            return result
    return "PASS"


def _ignore_config(args: argparse.Namespace) -> bool:
    if args.ignore_config:
        return True
    return os.environ.get("HRR_TEST_IGNORE_CONFIG", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _parse_arch(text: str) -> list[str]:
    return [arch for arch in text.split(",") if arch]


def _archs_for(arch_arg: str, ignore: bool) -> list[str]:
    """Architectures the rules are matched against.

    ``none`` is a run with no GPU. ``auto`` asks the machine, and fails when
    nothing is visible so an arch rule cannot be silently ignored. An ignored
    config does not need a device.
    """
    if arch_arg == "none" or (ignore and arch_arg == "auto"):
        return []
    if arch_arg == "auto":
        import hrr_test_config
        archs = hrr_test_config.detect_archs()
        if not archs:
            print("error: could not detect a GPU architecture; "
                  "pass --arch none or --arch gfxNNNN", file=sys.stderr)
            raise SystemExit(1)
        return archs
    archs = _parse_arch(arch_arg)
    if not archs or not all(arch.startswith("gfx") for arch in archs):
        print(f"error: --arch must be auto, none, or gfx names, not {arch_arg!r}",
              file=sys.stderr)
        raise SystemExit(1)
    return archs


def _is_filter(arg: str) -> bool:
    return arg == "*" or arg.startswith("[") or arg.startswith("~")


def _strip_filters(command: list[str]) -> list[str]:
    """Drop a trailing Catch2 selector. --config supplies its own."""
    prefix = list(command)
    while len(prefix) > 1 and _is_filter(prefix[-1]):
        prefix.pop()
    return prefix


def _add_skip(suite: ET.Element, name: str, message: str) -> None:
    case = ET.SubElement(suite, "testcase", {"name": name, "classname": "hrr-config"})
    ET.SubElement(case, "skipped", {"message": message})


def _write_junit(path: Path, skips: list[tuple[str, str]]) -> None:
    root = ET.Element("testsuites")
    suite = ET.SubElement(root, "testsuite", {
        "name": "hrr-config",
        "tests": str(len(skips)),
        "failures": "0",
        "errors": "0",
        "skipped": str(len(skips)),
    })
    for name, message in skips:
        _add_skip(suite, name, message)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="unicode", xml_declaration=True)


def _inject_skips(path: Path, skips: list[tuple[str, str]]) -> None:
    tree = ET.parse(path)
    root = tree.getroot()
    suite = root.find("testsuite")
    if suite is None:
        suite = root.find(".//testsuite")
    if suite is None:
        suite = ET.SubElement(root, "testsuite", {"name": "hrr-config"})
    for name, message in skips:
        _add_skip(suite, name, message)
    total = len(list(suite.iterfind("testcase")))
    skipped = len(list(suite.iterfind("testcase/skipped")))
    suite.set("tests", str(total))
    suite.set("skipped", str(skipped))
    ET.indent(tree, space="  ")
    tree.write(path, encoding="unicode", xml_declaration=True)


def _apply_config(args: argparse.Namespace) -> tuple[list[str], list[tuple[str, str]], str | None]:
    """Return the Catch2 command, the config skips, and the case-list file.

    An empty command means every selected case was skipped, so Catch2 is not
    started. Detection of an architecture fails closed: --arch auto with no
    visible device is an error, not a run that ignores arch rules.
    """
    import hrr_test_config

    try:
        config = hrr_test_config.load(args.config)
    except hrr_test_config.ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    errors = hrr_test_config.lint(config)
    if errors:
        print(f"FAIL: {args.config}", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        raise SystemExit(1)
    os_name = args.os_name or "auto"
    if os_name == "auto":
        os_name = hrr_test_config.detect_os()
    ignore = _ignore_config(args)
    archs = _archs_for(args.arch or "auto", ignore)
    select = args.select or config["select"]
    try:
        listed = hrr_test_config.list_cases(args.command[0], select)
    except hrr_test_config.ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    # The executable may be a wrapper (python fake.py spec); only its trailing
    # selector is Catch2's. Listing still uses command[0], which is the binary
    # in CI. Tests mock list_cases.
    names = [case["name"] for case in listed]
    _run, skipped = hrr_test_config.resolve(config, names, os_name, archs, ignore=ignore)
    records = [
        (item["name"], hrr_test_config.skip_message(os_name, item["archs"], item["rule"]))
        for item in skipped
    ]
    if not _run:
        return [], records, None
    prefix = _strip_filters(args.command)
    handle = tempfile.NamedTemporaryFile(
        "w", prefix="hrr-cases-", suffix=".txt", delete=False, encoding="utf-8",
    )
    with handle:
        for name in _run:
            handle.write(name + "\n")
    return prefix + ["--input-file", handle.name], records, handle.name


def _report(root: ET.Element) -> tuple[dict[str, int], str]:
    """Print one line per case. Returns the counts and the joined stdout."""
    labels = {"PASS": "PASS", "FAIL": "FAIL", "SKIP": "SKIP", "CONFIG": "SKIP (config)"}
    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0, "CONFIG": 0}
    lines: list[str] = []
    for name, records in group_cases(root).items():
        result = group_result(records)
        counts[result] += 1
        lines.append(f"{labels[result]}: {name}")
        if result == "CONFIG":
            for record in records:
                skipped = record.find("skipped")
                if skipped is not None and is_config_skip(record):
                    lines.append(skipped.attrib.get("message", ""))
        if result == "FAIL":
            for record in records:
                for failure in record.findall("failure") + record.findall("error"):
                    if failure.text:
                        lines.append(failure.text.strip())
    lines.append(
        f"{counts['PASS']} passed, {counts['FAIL']} failed, {counts['SKIP']} skipped"
    )
    if counts["CONFIG"]:
        lines.append(f"{counts['CONFIG']} config-skipped")
    text = "\n".join(lines) + "\n"
    print(text, end="")
    return counts, text


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # A mutable cell so the input file is removed even when the suite times out.
    state = {"input_file": None}
    try:
        return _run_suite(args, state)
    finally:
        path = state["input_file"]
        if path and os.path.exists(path):
            os.unlink(path)


def _run_suite(args: argparse.Namespace, state: dict) -> int:
    # Durations name every case as it finishes, so the transcript of a hung
    # suite ends just before the case that hung.
    skips: list[tuple[str, str]] = []
    if args.config:
        try:
            command, skips, state["input_file"] = _apply_config(args)
        except SystemExit as exc:
            return exc.code if isinstance(exc.code, int) else 1
        if not command:
            _write_junit(args.xml, skips)
            _report(ET.parse(args.xml).getroot())
            return 0
    else:
        command = list(args.command)
    command = command + [
        "--durations",
        "yes",
        "--reporter",
        "console",
        "--reporter",
        f"junit::out={args.xml}",
    ]
    # A session of its own, so a timeout also kills the workloads the suite
    # spawned: they hold the pipe open and would block the read forever.
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    ) as process:
        try:
            stdout, _ = process.communicate(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            # SIGTERM first: Catch2's fatal-signal handler names the running
            # case and flushes the console reporter, which SIGKILL would lose.
            stdout = ""
            for sig, grace in ((signal.SIGTERM, 10), (signal.SIGKILL, None)):
                try:
                    os.killpg(process.pid, sig)
                except ProcessLookupError:
                    pass
                try:
                    stdout, _ = process.communicate(timeout=grace)
                    break
                except subprocess.TimeoutExpired:
                    continue
            print(f"Catch2 timed out after {args.timeout:g} s; it hung after:")
            print(stdout, end="")
            return 1
    completed = subprocess.CompletedProcess(command, process.returncode, stdout)

    try:
        if skips:
            _inject_skips(args.xml, skips)
        root = ET.parse(args.xml).getroot()
    except (ET.ParseError, OSError) as error:
        print(f"Could not read JUnit results: {error}", file=sys.stderr)
        print(completed.stdout, end="")
        return completed.returncode or 1

    counts, _ = _report(root)
    for line in completed.stdout.splitlines():
        if "hrr spawn:" in line:
            print(line)

    # A crash can leave syntactically valid, but incomplete, JUnit without a
    # failed case. Preserve the raw transcript only for that exceptional path.
    # Catch2 returns 4 when the selected cases were all skipped; that is not a
    # crash and must not dump the banner. Any other non-zero status with no
    # failed case is one, skipped cases or not (a crash at exit after some
    # cases skipped, for instance, must still leave a log).
    if completed.returncode not in (0, 4) and counts["FAIL"] == 0:
        print("\nCatch2 terminated without a JUnit failure:")
        print(completed.stdout, end="")
    # Catch2 uses 4 for "tests were skipped / none ran". That is a skip, not a
    # failed suite: returning it would paint gfx90a red after every GPU case
    # correctly skipped for hipErrorNoDevice.
    if completed.returncode == 4 and counts["FAIL"] == 0:
        return 0
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
