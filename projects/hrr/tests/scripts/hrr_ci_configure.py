# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Decide which HRR CI jobs run and on which GPU families.

Writes these GitHub Actions outputs:

  run_cpu           "true" when the API check, script and unit jobs run.
  run_gpu           "true" when the capture-enabled build and the rocJitsu
                    simulator suite run. (The name predates the simulator;
                    no job here uses a real GPU.)
  families          JSON list of {family, runner, backend, arch, config}.
  simulator_families
                    The same rows, as hrr-gpu.yml's input.
  skipped_families  JSON list of {"family", "reason"}: the other ci.yaml
                    simulator families, when the suite runs.
  reason            One line explaining the decision, shown in the summary.
  rocm_sdk_version  The ROCm SDK the unit job installs.

The policy lives in projects/hrr/tests/config/ci.yaml. The suite needs an
amdhip64 with HRR capture compiled in, which only the HRR GPU workflow builds.
It runs when a change touches a capture or suite path, on a pull request with
the capture label that touches any HRR path (hrr-ci.yml's paths filter starts
no run otherwise), nightly and on workflow_dispatch.

Every job runs on the rocJitsu simulator. Its families are
simulator.default_families; the workflow_dispatch families input replaces the
selection. A family with no simulator config is an error. ci:gfx labels are
TheRock's, so this script does not read them.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "ci.yaml"
# Its targets.arch is what the suite is built for.
INTEGRATION_CONFIG = DEFAULT_CONFIG.with_name("integration.yaml")

EVENTS = ("pull_request", "push", "schedule", "workflow_dispatch")

# The pull request files API returns at most this many files. A larger change
# is treated as touching everything rather than guessed at.
MAX_LISTED_FILES = 3000


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Simulator:
    runner: str
    default_families: tuple[str, ...]
    configs: dict[str, str]


@dataclass(frozen=True)
class Config:
    rocm_sdk_version: str
    cpu_paths: tuple[str, ...]
    capture_paths: tuple[str, ...]
    suite_paths: tuple[str, ...]
    excluded_paths: tuple[str, ...]
    capture_label: str
    simulator: Simulator


@dataclass
class Decision:
    run_cpu: bool
    run_gpu: bool
    reason: str
    families: list[dict] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)


def _mapping(
    where: str, value, required: Iterable[str], optional: Iterable[str] = ()
) -> dict:
    if not isinstance(value, dict):
        raise ConfigError(f"{where}: expected a mapping")
    unknown = sorted(set(value) - set(required) - set(optional))
    if unknown:
        raise ConfigError(f"{where}: unknown key {', '.join(map(str, unknown))}")
    missing = sorted(set(required) - set(value))
    if missing:
        raise ConfigError(f"{where}: missing {', '.join(missing)}")
    return value


def _string(where: str, value) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{where}: expected a non-empty string")
    return value


def _strings(where: str, value) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{where}: expected a non-empty list")
    return tuple(_string(f"{where}[{i}]", v) for i, v in enumerate(value))


def _simulator(value) -> Simulator:
    value = _mapping("simulator", value, ("runner", "default_families", "configs"))
    default_families = _strings("simulator.default_families", value["default_families"])
    configs = value["configs"]
    if not isinstance(configs, dict) or not configs:
        raise ConfigError("simulator.configs: expected a mapping")
    parsed = {
        _string("simulator.configs", key): _string(f"simulator.configs.{key}", path)
        for key, path in configs.items()
    }
    unknown = [key for key in default_families if key not in parsed]
    if unknown:
        raise ConfigError(f"simulator.configs: missing {', '.join(unknown)}")
    return Simulator(
        runner=_string("simulator.runner", value["runner"]),
        default_families=default_families,
        configs=parsed,
    )


def parse_config(data) -> Config:
    data = _mapping(
        "top level",
        data,
        (
            "version",
            "rocm_sdk_version",
            "paths",
            "capture_label",
            "simulator",
        ),
    )
    if type(data["version"]) is not int or data["version"] != 1:
        raise ConfigError(f"version: expected 1, got {data['version']!r}")
    paths = _mapping("paths", data["paths"], ("cpu", "capture", "suite", "excluded"))
    return Config(
        rocm_sdk_version=_string("rocm_sdk_version", data["rocm_sdk_version"]),
        cpu_paths=_strings("paths.cpu", paths["cpu"]),
        capture_paths=_strings("paths.capture", paths["capture"]),
        suite_paths=_strings("paths.suite", paths["suite"]),
        excluded_paths=_strings("paths.excluded", paths["excluded"]),
        capture_label=_string("capture_label", data["capture_label"]),
        simulator=_simulator(data["simulator"]),
    )


def load_archs(path: Path) -> list[str]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return list(_strings("targets.arch", data["targets"]["arch"]))
    except (ConfigError, KeyError, TypeError) as e:
        raise ConfigError(f"{path}: no targets.arch list ({e})") from None


def load_config(path: Path) -> Config:
    try:
        return parse_config(yaml.safe_load(path.read_text(encoding="utf-8")))
    except ConfigError as e:
        raise ConfigError(f"{path}: {e}") from None


def check_families(config: Config, matrix: dict) -> None:
    unknown = [key for key in config.simulator.configs if key not in matrix]
    if unknown:
        raise ConfigError(
            f"families: {', '.join(unknown)} not in TheRock's family matrix"
            f" (known: {', '.join(sorted(matrix))})"
        )


def family_targets(linux: dict) -> list[str]:
    """The GPU targets of a family's test machines, without feature suffixes.

    TheRock lists them as fetch-gfx-targets; a family that names a single
    target (gfx1150) may leave that list empty.
    """
    targets = [t.split(":")[0] for t in linux.get("fetch-gfx-targets", [])]
    family = str(linux.get("family", ""))
    if not targets and re.fullmatch(r"gfx[0-9a-f]+", family):
        targets = [family]
    return list(dict.fromkeys(targets))


def check_targets(config: Config, matrix: dict, archs: Iterable[str]) -> None:
    """The suite must be built for every target of every simulator family
    ci.yaml lists, default or not, since workflow_dispatch can request any."""
    archs = set(archs)
    problems = []
    for key in config.simulator.configs:
        targets = family_targets(matrix[key].get("linux", {}))
        missing = [t for t in targets if t not in archs]
        if not targets:
            problems.append(f"TheRock lists no GPU targets for {key}")
        elif missing:
            problems.append(f"{key} needs {', '.join(missing)}")
    if problems:
        raise ConfigError(
            f"targets.arch of {INTEGRATION_CONFIG.name} does not cover the families: "
            + "; ".join(problems)
        )


def _matches(path: str, prefixes: Iterable[str]) -> bool:
    return any(path == p or path.startswith(p) for p in prefixes)


def relevant_paths(paths: Iterable[str], config: Config) -> list[str]:
    return [p for p in paths if not _matches(p, config.excluded_paths)]


def touches_cpu(paths: Iterable[str], config: Config) -> bool:
    return any(_matches(p, config.cpu_paths) for p in relevant_paths(paths, config))


def touches_capture(paths: Iterable[str], config: Config) -> bool:
    return any(_matches(p, config.capture_paths) for p in relevant_paths(paths, config))


def touches_suite(paths: Iterable[str], config: Config) -> bool:
    return any(_matches(p, config.suite_paths) for p in relevant_paths(paths, config))


def label_matters(label: str, config: Config) -> bool:
    """Whether adding this label can change what runs. hrr-ci.yml's concurrency
    group makes the same test, so that other labels do not cancel a run.
    ci:gfx labels belong to TheRock and must not restart this workflow."""
    label = label.strip().lower()
    return label == config.capture_label.lower()


def _arch(matrix: dict, key: str) -> str:
    return ",".join(family_targets(matrix.get(key, {}).get("linux", {})))


def _job(family: str, runner: str, backend: str, arch: str, config: str = "") -> dict:
    return {
        "family": family,
        "runner": runner,
        "backend": backend,
        "arch": arch,
        "config": config,
    }


def _requested(config: Config, requested: list[str]) -> set[str]:
    unknown = [key for key in requested if key not in config.simulator.configs]
    if unknown:
        raise ConfigError(
            f"requested families {', '.join(unknown)} have no simulator config in"
            f" ci.yaml (known: {', '.join(config.simulator.configs)})"
        )
    return set(requested)


def select_simulator(
    config: Config,
    requested: list[str],
    matrix: dict,
) -> tuple[list[dict], list[dict]]:
    """rocJitsu families. default_families, unless the dispatch input replaces
    them. A family with no config is an error."""
    if requested:
        chosen = _requested(config, requested)
    else:
        chosen = set(config.simulator.default_families)

    families, skipped = [], []
    for key in config.simulator.configs:
        if key not in chosen:
            reason = "not requested" if requested else "not a default simulator family"
            skipped.append({"family": key, "reason": reason})
            continue
        families.append(
            _job(
                key,
                config.simulator.runner,
                "simulator",
                _arch(matrix, key),
                config.simulator.configs[key],
            )
        )
    return families, skipped


def decide(
    event: str,
    paths: list[str] | None,
    labels: list[str],
    requested: list[str],
    config: Config,
    matrix: dict,
    added_label: str | None = None,
) -> Decision:
    """paths is None when the change set could not be listed in full.

    added_label is the label a labeled event added; the run for any other
    pull request event already covers a label that cannot change what runs.
    """
    if event not in EVENTS:
        return Decision(False, False, f"event {event} does not run HRR CI")
    if added_label is not None and not label_matters(added_label, config):
        return Decision(
            False, False, f"the label {added_label} does not change what HRR CI runs"
        )

    if event in ("schedule", "workflow_dispatch"):
        run_cpu, capture, why = True, True, f"{event} run"
    elif paths is None:
        run_cpu, capture, why = True, True, "change set too large to list"
    else:
        run_cpu = touches_cpu(paths, config)
        capture = touches_capture(paths, config) or touches_suite(paths, config)
        if touches_capture(paths, config):
            why = "the change touches capture or replay"
        elif touches_suite(paths, config):
            why = "the change touches the suite config"
        elif run_cpu:
            why = "the change touches HRR"
        else:
            why = "the change does not touch HRR"

    labelled = event == "pull_request" and config.capture_label.lower() in {
        l.strip().lower() for l in labels
    }
    if labelled and not capture:
        run_cpu, capture, why = (
            True,
            True,
            f"the pull request is labelled {config.capture_label}",
        )

    # Capture paths that the CPU list omits (hip_context.cpp, hip_code_object.*)
    # still need the host unit suite.
    run_cpu = run_cpu or capture
    if not capture:
        return Decision(run_cpu, False, why)
    # The simulator jobs need no secrets and no GPU runner (hrr-gpu.yml has
    # read-only permissions), so a fork pull request runs them like any other.
    families, skipped = select_simulator(config, requested, matrix)
    return Decision(run_cpu, True, why, families, skipped)


def list_pr_files(repo: str, number: str) -> list[str] | None:
    out = subprocess.run(
        [
            "gh",
            "api",
            "--paginate",
            f"repos/{repo}/pulls/{number}/files",
            "--jq",
            ".[] | .filename, (.previous_filename // empty)",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    files = [line for line in out.splitlines() if line]
    return None if len(files) >= MAX_LISTED_FILES else files


def list_push_files(repo: str, before: str, after: str) -> list[str] | None:
    if not before or set(before) == {"0"}:
        return None
    out = subprocess.run(
        [
            "gh",
            "api",
            f"repos/{repo}/compare/{before}...{after}",
            "--jq",
            ".files[] | .filename, (.previous_filename // empty)",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    files = [line for line in out.splitlines() if line]
    # The compare API lists at most 300 files.
    return None if len(files) >= 300 else files


def load_matrix(therock_dir: Path):
    sys.path.insert(0, str(therock_dir / "build_tools" / "github_actions"))
    import amdgpu_family_matrix  # noqa: E402

    return amdgpu_family_matrix.get_all_families_for_trigger_types([])


def _dump(rows: list[dict]) -> str:
    return json.dumps(rows, separators=(",", ":"))


def write_outputs(
    decision: Decision, rocm_sdk_version: str, output_path: str | None
) -> None:
    lines = [
        f"run_cpu={'true' if decision.run_cpu else 'false'}",
        f"run_gpu={'true' if decision.run_gpu else 'false'}",
        f"families={_dump(decision.families)}",
        f"simulator_families={_dump(decision.families)}",
        f"skipped_families={_dump(decision.skipped)}",
        f"reason={decision.reason}",
        f"rocm_sdk_version={rocm_sdk_version}",
    ]
    text = "\n".join(lines) + "\n"
    if output_path:
        with open(output_path, "a", encoding="utf-8") as out:
            out.write(text)
    sys.stdout.write(text)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--event", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--pr-number", default="")
    parser.add_argument("--before", default="")
    parser.add_argument("--after", default="")
    parser.add_argument("--labels", default="[]", help="JSON list of label names")
    parser.add_argument("--action", default="", help="The pull_request event's action")
    parser.add_argument("--label", default="", help="The label a labeled event added")
    parser.add_argument(
        "--families", default="", help="Comma-separated family keys (workflow_dispatch)"
    )
    parser.add_argument("--therock-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--integration-config", type=Path, default=INTEGRATION_CONFIG)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    labels = json.loads(args.labels or "[]")
    requested = [f.strip().lower() for f in args.families.split(",") if f.strip()]

    try:
        config = load_config(args.config)
        matrix = load_matrix(args.therock_dir)
        check_families(config, matrix)
        check_targets(config, matrix, load_archs(args.integration_config))

        paths: list[str] | None = []
        if args.event == "pull_request" and args.pr_number:
            paths = list_pr_files(args.repo, args.pr_number)
        elif args.event == "push":
            paths = list_push_files(args.repo, args.before, args.after)

        decision = decide(
            args.event,
            paths,
            labels,
            requested,
            config,
            matrix,
            added_label=args.label if args.action == "labeled" else None,
        )
    except ConfigError as e:
        print(f"::error::{e}")
        return 1
    write_outputs(decision, config.rocm_sdk_version, os.environ.get("GITHUB_OUTPUT"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
