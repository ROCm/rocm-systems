#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Where each HRR test case runs.

The suites are built once, as one fat binary covering every architecture. This
file is what decides, on the machine actually running the binary, which of the
registered cases run and which are skipped. The decision is data: one YAML file
per suite lists every case, and a case with no rule runs on every OS and GPU.

A rule matches when every key it sets matches. Keys in one ``when`` are ANDed;
a list under a key is ORed; a missing key matches anything. Glob patterns
(``gfx11*``) are allowed. Several visible GPUs match when any of them does.
``kind: disabled`` is a temporary regression and needs a public issue URL.
``kind: unsupported`` is permanent and needs only a reason.

Usage:
    hrr_test_config.py lint CONFIG [CONFIG ...] [--workflow hrr-ci.yml]
    hrr_test_config.py check --binary EXE --config CONFIG [--update]
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

CONFIG_SKIP_PREFIX = "hrr-config:"
CASE_NAME = re.compile(r"^[A-Za-z0-9_]+$")
ARCH_TOKEN = re.compile(r"gfx[0-9][0-9a-z]*")
WORKLOAD_TAGS = {"hrr-direct", "direct"}
KINDS = {"disabled", "unsupported"}
WHEN_KEYS = {"os", "arch"}
RULE_KEYS = {"when", "kind", "reason", "issue"}
CASE_KEYS = {"skip"}
TOP_KEYS = {"version", "suite", "binary", "select", "targets", "skip", "cases"}
HEADER = """\
# HRR test configuration. Every case the binary registers has an entry here.
# An empty entry runs on every OS and GPU architecture.
#
# A skip rule matches when every key under "when" matches (a list is OR, a
# missing key matches anything, and a value may be a glob such as gfx11*).
# kind: disabled is a temporary regression and requires a public issue URL.
# kind: unsupported is permanent and needs only a reason.
#
# Regenerate the case names from a built binary with:
#   hrr_test_config.py check --binary <exe> --config <this file> --update
# Listing needs no GPU. Say so in the commit message when the set changes.
"""


class ConfigError(Exception):
    """The config or the case listing cannot be used."""


def load(path: str | Path) -> dict:
    """Read a suite config. Raises ConfigError on unreadable YAML."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except OSError as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"cannot parse {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must be a YAML mapping")
    return data


def _normalize_tag(tag: str) -> str:
    return tag.strip().lstrip("[").rstrip("]")


def is_workload(tags: list[str]) -> bool:
    """A direct workload is spawned by a driver; CI does not select it."""
    return bool(WORKLOAD_TAGS & {_normalize_tag(tag) for tag in tags})


def _is_glob(value: str) -> bool:
    return any(ch in value for ch in "*?[")


def _as_list(value) -> list:
    return value if isinstance(value, list) else [value]


def value_matches(pattern, value: str) -> bool:
    """True when value matches a scalar or any entry of a list."""
    return any(fnmatch.fnmatchcase(value, str(item)) for item in _as_list(pattern))


def rule_matches(rule: dict, os_name: str, archs: list[str]) -> list[str] | None:
    """The architectures that made the rule match, or None.

    An ``arch`` key matches when any visible architecture matches, and does
    not match when no architecture was supplied. An empty ``when`` matches
    every machine.
    """
    when = rule.get("when") or {}
    if "os" in when and not value_matches(when["os"], os_name):
        return None
    matched = list(archs)
    if "arch" in when:
        matched = [arch for arch in archs if value_matches(when["arch"], arch)]
        if not matched:
            return None
    return matched


def _rules_for(config: dict, name: str) -> list[dict]:
    case = config["cases"].get(name) or {}
    return list(config.get("skip") or []) + list(case.get("skip") or [])


def resolve(config: dict, names: list[str], os_name: str, archs: list[str],
            ignore: bool = False) -> tuple[list[str], list[dict]]:
    """Split ``names`` into cases to run and cases a rule skips.

    The first matching rule wins, suite rules before the case's own. Each
    skip record carries the case name, the rule, and the architectures that
    matched. ``ignore`` runs everything.
    """
    if ignore:
        return list(names), []
    run: list[str] = []
    skipped: list[dict] = []
    for name in names:
        hit = None
        matched: list[str] = []
        for rule in _rules_for(config, name):
            matched = rule_matches(rule, os_name, archs)
            if matched is not None:
                hit = rule
                break
        if hit is None:
            run.append(name)
        else:
            skipped.append({"name": name, "rule": hit, "archs": matched})
    return run, skipped


def skip_message(os_name: str, archs: list[str], rule: dict) -> str:
    """The JUnit skip message. The prefix is how the summary recognises it."""
    where = os_name if not archs else os_name + "/" + ",".join(archs)
    message = f"{CONFIG_SKIP_PREFIX} {where}: {rule['reason']}"
    if rule.get("issue"):
        message += f" ({rule['issue']})"
    return message


def _error(errors: list[str], where: str, text: str) -> None:
    errors.append(f"{where}: {text}")


def _check_pattern(errors: list[str], where: str, key: str, pattern, vocab: list[str]) -> None:
    if isinstance(pattern, list):
        if not pattern:
            _error(errors, where, f"'{key}' is an empty list")
            return
        for item in pattern:
            _check_pattern(errors, where, key, item, vocab)
        return
    if not isinstance(pattern, str) or not pattern:
        _error(errors, where, f"'{key}' must be a string or a list of strings")
        return
    if _is_glob(pattern):
        if not any(fnmatch.fnmatchcase(item, pattern) for item in vocab):
            _error(errors, where, f"'{pattern}' matches no {key} in targets")
        return
    if pattern not in vocab:
        _error(errors, where, f"'{pattern}' is not a known {key}")


def _check_rule(errors: list[str], where: str, rule, targets: dict) -> None:
    if not isinstance(rule, dict):
        _error(errors, where, "a skip rule must be a mapping")
        return
    unknown = set(rule) - RULE_KEYS
    if unknown:
        _error(errors, where, "unknown rule keys: " + ", ".join(sorted(unknown)))
    when = rule.get("when", {})
    if not isinstance(when, dict):
        _error(errors, where, "'when' must be a mapping")
    else:
        unknown_when = set(when) - WHEN_KEYS
        if unknown_when:
            _error(errors, where, "unknown when keys: " + ", ".join(sorted(unknown_when)))
        for key in ("os", "arch"):
            if key in when:
                _check_pattern(errors, where, key, when[key], targets.get(key, []))
    kind = rule.get("kind")
    if kind not in KINDS:
        _error(errors, where, "'kind' must be disabled or unsupported")
    reason = rule.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        _error(errors, where, "'reason' is required")
    issue = rule.get("issue")
    if kind == "disabled" and (not isinstance(issue, str) or not issue.startswith("https://")):
        _error(errors, where, "'disabled' requires an https:// issue URL")
    elif issue is not None and (not isinstance(issue, str) or not issue.startswith("https://")):
        _error(errors, where, "'issue' must be an https:// URL")


def lint(config: dict, cases: list[dict] | None = None,
         workflow_archs: list[str] | None = None) -> list[str]:
    """Schema and vocabulary errors. ``cases`` enables the workload-rule check.

    ``workflow_archs``, when given, must be the same set as ``targets.arch``.
    """
    errors: list[str] = []
    unknown = set(config) - TOP_KEYS
    if unknown:
        _error(errors, "config", "unknown keys: " + ", ".join(sorted(unknown)))
    if config.get("version") != 1:
        _error(errors, "config", "'version' must be 1")
    for key in ("suite", "binary", "select"):
        if not isinstance(config.get(key), str) or not config[key]:
            _error(errors, "config", f"'{key}' must be a non-empty string")
    targets = config.get("targets")
    if not isinstance(targets, dict) or set(targets) != {"os", "arch"}:
        _error(errors, "config", "'targets' must have exactly 'os' and 'arch'")
        targets = {"os": [], "arch": []}
    else:
        for key in ("os", "arch"):
            values = targets[key]
            if not isinstance(values, list) or not values or not all(
                    isinstance(item, str) and item for item in values):
                _error(errors, "targets", f"'{key}' must be a non-empty list of strings")
                targets[key] = []
    if workflow_archs is not None and targets.get("arch") is not None:
        got = set(targets["arch"])
        want = set(workflow_archs)
        if got != want:
            missing = ", ".join(sorted(want - got)) or "-"
            extra = ", ".join(sorted(got - want)) or "-"
            _error(errors, "targets.arch",
                   f"does not match HRR_ARCHS (missing: {missing}; extra: {extra})")
    suite_skip = config.get("skip", [])
    if not isinstance(suite_skip, list):
        _error(errors, "skip", "must be a list")
        suite_skip = []
    for index, rule in enumerate(suite_skip):
        _check_rule(errors, f"skip[{index}]", rule, targets)
    cases_map = config.get("cases")
    if not isinstance(cases_map, dict):
        _error(errors, "cases", "must be a mapping")
        return errors
    tags_by_name = {case["name"]: case.get("tags", []) for case in cases or []}
    for name, case in cases_map.items():
        where = f"cases.{name}"
        if not isinstance(name, str) or not CASE_NAME.match(name):
            _error(errors, where, "case name must match [A-Za-z0-9_]+")
        if case is None:
            case = {}
        if not isinstance(case, dict):
            _error(errors, where, "must be a mapping")
            continue
        unknown_case = set(case) - CASE_KEYS
        if unknown_case:
            _error(errors, where, "unknown keys: " + ", ".join(sorted(unknown_case)))
        rules = case.get("skip", [])
        if not isinstance(rules, list):
            _error(errors, where, "'skip' must be a list")
            continue
        for index, rule in enumerate(rules):
            _check_rule(errors, f"{where}.skip[{index}]", rule, targets)
        if rules and is_workload(tags_by_name.get(name, [])):
            _error(errors, where, "a direct workload cannot carry skip rules")
    return errors


def list_cases(binary: str, select: str = "*") -> list[dict]:
    """Every case ``select`` matches, hidden ones included when ``select`` is ``*``.

    Catch2's JSON reporter is the listing format. Each result is
    ``{"name", "tags"}`` with tags as Catch2 spelled them.
    """
    proc = subprocess.run(
        [binary, "--list-tests", select, "--reporter", "json"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise ConfigError(
            f"{binary} --list-tests failed ({proc.returncode})\n{proc.stderr.strip()}"
        )
    try:
        payload = json.loads(proc.stdout)
        tests = payload["listings"]["tests"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ConfigError(f"could not parse --list-tests JSON from {binary}: {exc}") from exc
    if not isinstance(tests, list):
        raise ConfigError(f"{binary} --list-tests JSON has no tests array")
    cases = []
    for test in tests:
        name = test.get("name") if isinstance(test, dict) else None
        if not name:
            raise ConfigError(f"{binary} listed a test without a name")
        tags = test.get("tags") or []
        cases.append({"name": name, "tags": list(tags)})
    return cases


def listed_names(binary: str) -> list[str]:
    """Every registered case name, sorted. An empty binary is itself a failure."""
    names = sorted({case["name"] for case in list_cases(binary, "*")})
    if not names:
        raise ConfigError(
            f"{binary} listed no test cases at all -- that is itself the failure "
            "this check exists to catch"
        )
    return names


def _yaml_str(value: str) -> str:
    # '*' alone is a YAML alias, and ':', '#', quotes and brackets need quoting.
    if re.fullmatch(r"[A-Za-z0-9_./~-]+", value):
        return value
    return json.dumps(value)


def _yaml_pattern(value) -> str:
    if isinstance(value, list):
        return "[" + ", ".join(_yaml_str(str(item)) for item in value) + "]"
    return _yaml_str(str(value))


def _emit_rule(lines: list[str], rule: dict, indent: str) -> None:
    when = rule.get("when") or {}
    parts = [f"{key}: {_yaml_pattern(when[key])}" for key in ("os", "arch") if key in when]
    lines.append(f"{indent}- when: {{{', '.join(parts)}}}")
    lines.append(f"{indent}  kind: {rule['kind']}")
    lines.append(f"{indent}  reason: {_yaml_str(rule['reason'])}")
    if rule.get("issue"):
        lines.append(f"{indent}  issue: {_yaml_str(rule['issue'])}")


def render(config: dict) -> str:
    """Stable YAML for ``config``, case names sorted, empty rules omitted."""
    lines = [HEADER.rstrip("\n"), "version: 1", f"suite: {_yaml_str(config['suite'])}",
             f"binary: {_yaml_str(config['binary'])}", f"select: {_yaml_str(config['select'])}",
             "targets:", "  os: [" + ", ".join(config["targets"]["os"]) + "]",
             "  arch: [" + ", ".join(config["targets"]["arch"]) + "]"]
    suite_skip = config.get("skip") or []
    if suite_skip:
        lines.append("skip:")
        for rule in suite_skip:
            _emit_rule(lines, rule, "  ")
    lines.append("cases:")
    for name in sorted(config["cases"]):
        case = config["cases"][name] or {}
        rules = case.get("skip") or []
        if not rules:
            lines.append(f"  {name}: {{}}")
            continue
        lines.append(f"  {name}:")
        lines.append("    skip:")
        for rule in rules:
            _emit_rule(lines, rule, "      ")
    return "\n".join(lines) + "\n"


def apply_listing(config: dict, names: list[str]) -> tuple[list[str], list[str]]:
    """Add new cases as empty entries and drop cases the binary no longer has.

    Returns ``(added, removed)``. Rules on a case that is still present stay.
    """
    current = set(config["cases"])
    found = set(names)
    added = sorted(found - current)
    removed = sorted(current - found)
    for name in added:
        config["cases"][name] = {}
    for name in removed:
        del config["cases"][name]
    return added, removed


def workflow_archs(path: str | Path) -> list[str]:
    """The ``HRR_ARCHS`` list from the HRR CI workflow."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc
    match = re.search(r'^ {2}HRR_ARCHS:\s*"([^"]+)"\s*$', text, re.MULTILINE)
    if not match:
        raise ConfigError(f"no HRR_ARCHS list in {path}")
    return [arch for arch in match.group(1).split(";") if arch]


def detect_os() -> str:
    return "windows" if os.name == "nt" else "linux"


def _archs_in(text: str) -> list[str]:
    found = []
    for line in text.splitlines():
        found.extend(ARCH_TOKEN.findall(line))
    return list(dict.fromkeys(found))


def _run_tool(argv: list[str]) -> str | None:
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def detect_archs(runner=None) -> list[str]:
    """GPU architectures visible on this machine, most specific tool first.

    Tries ``offload-arch``, then ``amdgpu-arch`` (under ``ROCM_SDK`` and on
    ``PATH``), then ``rocminfo``. Feature suffixes such as ``:xnack-`` are
    dropped. An empty list means no device could be identified.
    """
    run = runner or _run_tool
    candidates = [["offload-arch"]]
    sdk = os.environ.get("ROCM_SDK")
    if sdk:
        candidates.append([str(Path(sdk) / "lib" / "llvm" / "bin" / "amdgpu-arch")])
    candidates.append(["amdgpu-arch"])
    if sdk:
        candidates.append([str(Path(sdk) / "bin" / "rocminfo")])
    candidates.append(["rocminfo"])
    for argv in candidates:
        text = run(argv)
        if text:
            archs = _archs_in(text)
            if archs:
                return sorted(set(archs))
    return []


def _print_errors(path: str, errors: list[str]) -> None:
    print(f"FAIL: {path}", file=sys.stderr)
    for error in errors:
        print(f"  {error}", file=sys.stderr)


def _cmd_lint(args: argparse.Namespace) -> int:
    archs = workflow_archs(args.workflow) if args.workflow else None
    failed = False
    for path in args.configs:
        try:
            config = load(path)
        except ConfigError as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            failed = True
            continue
        errors = lint(config, workflow_archs=archs)
        if errors:
            _print_errors(path, errors)
            failed = True
        else:
            print(f"OK: {path}")
    return 1 if failed else 0


def _cmd_check(args: argparse.Namespace) -> int:
    try:
        config = load(args.config)
        found = list_cases(args.binary, "*")
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    names = sorted({case["name"] for case in found})
    if not names:
        print(f"error: {args.binary} listed no test cases at all -- that is itself "
              "the failure this check exists to catch", file=sys.stderr)
        return 1
    if args.update:
        added, removed = apply_listing(config, names)
        Path(args.config).write_text(render(config), encoding="utf-8")
        print(f"wrote {len(config['cases'])} cases to {args.config} "
              f"({len(added)} added, {len(removed)} removed)")
        for name in removed:
            print(f"  removed {name}")
    expected = set(config["cases"])
    actual = set(names)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    errors = lint(config, cases=found)
    if not missing and not unexpected and not errors:
        print(f"OK: {len(actual)} test cases present, matching {args.config}")
        return 0
    print(f"FAIL: {args.binary} does not match {args.config}", file=sys.stderr)
    if missing or unexpected:
        print(f"  expected {len(expected)} cases, found {len(actual)}", file=sys.stderr)
    for name in missing:
        print(f"  MISSING (compiled out or deleted): {name}", file=sys.stderr)
    for name in unexpected:
        print(f"  UNEXPECTED (added without updating the config): {name}", file=sys.stderr)
    for error in errors:
        print(f"  {error}", file=sys.stderr)
    if missing:
        print("\nA MISSING case is usually a guard that stopped being satisfied -- "
              "check that every -D its #if needs is still passed by CMake.",
              file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    lint_cmd = commands.add_parser("lint", help="validate config files")
    lint_cmd.add_argument("configs", nargs="+")
    lint_cmd.add_argument("--workflow", help="HRR workflow whose HRR_ARCHS must match targets.arch")
    lint_cmd.set_defaults(func=_cmd_lint)

    check_cmd = commands.add_parser("check", help="compare a binary's cases to a config")
    check_cmd.add_argument("--binary", required=True)
    check_cmd.add_argument("--config", required=True)
    check_cmd.add_argument("--update", action="store_true",
                           help="rewrite the config's case list from the binary, keeping rules")
    check_cmd.set_defaults(func=_cmd_check)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
