# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Scenario file: load and validate it, turn it into the commands bench runs."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jsonschema

SCHEMA_PATH = Path(__file__).with_name("config.schema.json")
BASELINE = "baseline"
REFERENCE_WORKLOAD = "workload"
CONFIG_FILE_KEY = "ROCPROFSYS_CONFIG_FILE"
VERSION_TIMEOUT_S = 10.0


class ScenarioError(ValueError):
    """The scenario file is invalid; the message says where and why."""


def check_semantics(scenario: dict[str, Any]) -> None:
    """Rules the JSON schema cannot express."""
    names = [t["name"] for t in scenario["tools"]]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ScenarioError(f"duplicate tool name(s): {dup}")
    if BASELINE in names:
        raise ScenarioError(
            f"tool name '{BASELINE}' is reserved for the plain workload run"
        )
    reference = scenario.get("reference", REFERENCE_WORKLOAD)
    if reference != REFERENCE_WORKLOAD and reference not in names:
        raise ScenarioError(
            f"reference '{reference}' is neither '{REFERENCE_WORKLOAD}' "
            f"nor a tool name {names}"
        )
    if reference != REFERENCE_WORKLOAD and len(names) < 2:
        raise ScenarioError(f"reference '{reference}' needs at least 2 tools to compare")


def load(path: str) -> dict[str, Any]:
    """Read, schema-validate and semantically check a scenario file."""
    try:
        scenario = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ScenarioError(f"{path}: {exc}") from exc
    schema = json.loads(SCHEMA_PATH.read_text())
    try:
        jsonschema.validate(scenario, schema)
    except jsonschema.ValidationError as exc:
        where = "/".join(map(str, exc.absolute_path)) or "<root>"
        raise ScenarioError(f"{path}: at {where}: {exc.message}") from exc
    check_semantics(scenario)
    return scenario


def resolve_path(value: str, base_dir: str) -> str:
    """A value containing '/' is relative to base_dir; a bare name is left for PATH lookup."""
    if "/" not in value or os.path.isabs(value):
        return value
    return os.path.normpath(os.path.join(base_dir, value))


def render_args(args: list[dict[str, str]]) -> list[str]:
    return [
        f"{a['arg_name']}={a['value']}" if "value" in a else a["arg_name"] for a in args
    ]


def env_of(entries: list[dict[str, str]]) -> dict[str, str]:
    return {e["key"]: e["value"] for e in entries}


def resolve_config_files(env: dict[str, str], base_dir: str) -> dict[str, str]:
    """Make the ':'-separated files of CONFIG_FILE_KEY relative to base_dir, not to the
    directory bench is started from, so a scenario can name a shared config file."""
    if CONFIG_FILE_KEY not in env:
        return env
    files = [
        os.path.normpath(os.path.join(base_dir, f))
        for f in env[CONFIG_FILE_KEY].split(":")
        if f
    ]
    return {**env, CONFIG_FILE_KEY: ":".join(files)}


def config_keys(path: str) -> set[str]:
    """Setting names a text config file assigns ('KEY = value' or 'KEY value' lines)."""
    # ponytail: text config files only; a .json or .xml config file is not parsed
    lines = (line.strip() for line in Path(path).read_text().splitlines())
    return {
        re.split(r"[\s=]", line, maxsplit=1)[0]
        for line in lines
        if line and not line.startswith("#")
    }


def shadowed_keys(env: dict[str, str], ambient: Mapping[str, str]) -> list[str]:
    """Keys of the config files named in env that a variable of ambient would override.

    The profiler applies the environment after its config files, so an exported variable
    beats the same key in a file. A key the command sets itself (env) is safe: it replaces
    the exported value. Raises OSError when a config file cannot be read.
    """
    files = [f for f in env.get(CONFIG_FILE_KEY, "").split(":") if f]
    in_files = set().union(*(config_keys(f) for f in files))
    return sorted((in_files & set(ambient)) - set(env))


def check_environment(
    path: str, scenario: dict[str, Any], ambient: Mapping[str, str]
) -> None:
    """Raise ScenarioError when a variable of ambient, the caller's environment, would
    override a config-file key of a command, or when a config file cannot be read."""
    base_dir = os.path.dirname(os.path.abspath(path))
    for name, _command, env in build_commands(scenario, base_dir):
        try:
            shadowed = shadowed_keys(env, ambient)
        except OSError as exc:
            raise ScenarioError(f"{path}: tool '{name}': config file: {exc}") from exc
        if shadowed:
            raise ScenarioError(
                f"{path}: tool '{name}': unset {', '.join(shadowed)}: an exported "
                "variable overrides the same key in the tool's config file"
            )


def version_argv(tool: dict[str, Any], base_dir: str) -> list[str]:
    """Argv printing the tool's version: tool_version_cmd if given, else '<tool> --version'."""
    argv = (
        shlex.split(tool["tool_version_cmd"])
        if "tool_version_cmd" in tool
        else [tool["tool"], "--version"]
    )
    if not argv:
        raise ScenarioError(f"tool '{tool['name']}': tool_version_cmd is empty")
    return [resolve_path(argv[0], base_dir), *argv[1:]]


def probe_version(argv: list[str]) -> str | None:
    """First line of the version output, or None. A failed probe is a warning, not a fatal error:
    the benchmark is still valid without the version."""
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=VERSION_TIMEOUT_S, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(
            f"bench-scenario: warning: version probe '{shlex.join(argv)}' failed: {exc}",
            file=sys.stderr,
        )
        return None
    lines = (proc.stdout.strip() or proc.stderr.strip()).splitlines()
    if proc.returncode != 0 or not lines:
        print(
            f"bench-scenario: warning: version probe '{shlex.join(argv)}' exited "
            f"{proc.returncode} without version output",
            file=sys.stderr,
        )
        return None
    return lines[0].strip()


def build_commands(
    scenario: dict[str, Any], base_dir: str
) -> list[tuple[str, str, dict[str, str]]]:
    """Return (name, command, env) per command; the reference comes first."""
    workload = [
        resolve_path(scenario["workload"], base_dir),
        *render_args(scenario.get("workload_arguments", [])),
    ]
    workload_env = env_of(scenario.get("workload_env", []))
    cmds = [
        (
            t["name"],
            shlex.join(
                [
                    resolve_path(t["tool"], base_dir),
                    *render_args(t.get("tool_arguments", [])),
                    "--",
                    *workload,
                ]
            ),
            resolve_config_files(
                {**workload_env, **env_of(t.get("tool_env", []))}, base_dir
            ),
        )
        for t in scenario["tools"]
    ]
    reference = scenario.get("reference", REFERENCE_WORKLOAD)
    if reference == REFERENCE_WORKLOAD:
        return [(BASELINE, shlex.join(workload), workload_env), *cmds]
    return sorted(cmds, key=lambda c: c[0] != reference)
