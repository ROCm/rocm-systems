# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Decide which HRR CI jobs run and on which GPU families.

Writes these GitHub Actions outputs:

  run_cpu           "true" when the generated-source, script and unit jobs run.
  run_gpu           "true" when the capture-enabled build and the GPU suite run.
  families          JSON list of {"family", "runner"} for the GPU suite.
  skipped_families  JSON list of {"family", "reason"}: the other ci.yaml
                    families, when the GPU suite runs.
  reason            One line explaining the decision, shown in the summary.
  rocm_sdk_version  The ROCm SDK the unit job installs.

The policy lives in projects/hrr/tests/config/ci.yaml. The GPU suite needs an
amdhip64 with HRR capture compiled in, which only the HRR GPU workflow builds.
It runs when a change touches a capture path, on any pull request with the
capture label, nightly and on workflow_dispatch.

It runs on every enabled family whose triggers include the event. A ci:gfx...
label adds that family, the same label TheRock CI uses; the workflow_dispatch
families input replaces the selection. Both can name a disabled family. The
runner labels come from TheRock's family matrix unless ci.yaml overrides them;
a family takes TheRock's multi-GPU runner only on its multi_gpu_triggers.
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
from typing import Callable, Iterable

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "ci.yaml"
# Its targets.arch is what the GPU suite is built for.
INTEGRATION_CONFIG = DEFAULT_CONFIG.with_name("integration.yaml")

EVENTS = ("pull_request", "push", "schedule", "workflow_dispatch")

# The pull request files API returns at most this many files. A larger change
# is treated as touching everything rather than guessed at.
MAX_LISTED_FILES = 3000

NO_RUNNER = "no runner label in TheRock's runner config"


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Family:
    enabled: bool = True
    triggers: tuple[str, ...] = EVENTS
    runner: str = ""
    multi_gpu_triggers: tuple[str, ...] = ()


@dataclass(frozen=True)
class Config:
    rocm_sdk_version: str
    cpu_paths: tuple[str, ...]
    capture_paths: tuple[str, ...]
    excluded_paths: tuple[str, ...]
    capture_label: str
    families: dict[str, Family]


@dataclass
class Decision:
    run_cpu: bool
    run_gpu: bool
    reason: str
    families: list[dict] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)


def _mapping(where: str, value, required: Iterable[str], optional: Iterable[str] = ()) -> dict:
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


def _bool(where: str, value) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{where}: expected true or false")
    return value


def _triggers(where: str, value) -> tuple[str, ...]:
    triggers = _strings(where, value)
    for trigger in triggers:
        if trigger not in EVENTS:
            raise ConfigError(
                f"{where}: unknown trigger {trigger!r} (expected {', '.join(EVENTS)})"
            )
    return triggers


def _family(where: str, value) -> Family:
    value = _mapping(
        where, value or {}, (), ("enabled", "triggers", "runner", "multi_gpu_triggers")
    )
    return Family(
        enabled=_bool(f"{where}.enabled", value.get("enabled", True)),
        triggers=_triggers(f"{where}.triggers", value.get("triggers", list(EVENTS))),
        runner=_string(f"{where}.runner", value["runner"]) if "runner" in value else "",
        multi_gpu_triggers=(
            _triggers(f"{where}.multi_gpu_triggers", value["multi_gpu_triggers"])
            if "multi_gpu_triggers" in value else ()
        ),
    )


def parse_config(data) -> Config:
    data = _mapping(
        "top level", data, ("version", "rocm_sdk_version", "paths", "capture_label", "families")
    )
    if type(data["version"]) is not int or data["version"] != 1:
        raise ConfigError(f"version: expected 1, got {data['version']!r}")
    paths = _mapping("paths", data["paths"], ("cpu", "capture", "excluded"))
    families = data["families"]
    if not isinstance(families, dict) or not families:
        raise ConfigError("families: expected a mapping with at least one family")
    return Config(
        rocm_sdk_version=_string("rocm_sdk_version", data["rocm_sdk_version"]),
        cpu_paths=_strings("paths.cpu", paths["cpu"]),
        capture_paths=_strings("paths.capture", paths["capture"]),
        excluded_paths=_strings("paths.excluded", paths["excluded"]),
        capture_label=_string("capture_label", data["capture_label"]),
        families={
            _string("families", key): _family(f"families.{key}", value)
            for key, value in families.items()
        },
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
    unknown = [key for key in config.families if key not in matrix]
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
    """The suite must be built for every target of every family ci.yaml lists,
    enabled or not, since a label can request any of them."""
    archs = set(archs)
    problems = []
    for key in config.families:
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


def labelled_families(labels: Iterable[str], matrix: dict) -> list[str]:
    """Family keys named by ci:gfx... labels, e.g. ci:gfx110x or ci:gfx110X-all."""
    wanted = []
    for label in labels:
        label = label.strip().lower()
        if not label.startswith("ci:gfx"):
            continue
        name = label[len("ci:"):]
        for key, config in matrix.items():
            linux = config.get("linux", {})
            if name in (key.lower(), str(linux.get("family", "")).lower()):
                wanted.append(key)
    return wanted


def label_matters(label: str, config: Config) -> bool:
    """Whether adding this label can change what runs. hrr-ci.yml's concurrency
    group makes the same test, so that other labels do not cancel a run."""
    label = label.strip().lower()
    return label == config.capture_label.lower() or label.startswith("ci:gfx")


def runner_for(
    linux: dict, pick_weighted: Callable[[list, str], str], multi_gpu: bool = False
) -> str:
    if multi_gpu:
        labels = linux.get("test-runs-on-multi-gpu-labels")
        if labels:
            return pick_weighted(labels, "hrr-multi-gpu")
        if linux.get("test-runs-on-multi-gpu"):
            return linux["test-runs-on-multi-gpu"]
    labels = linux.get("test-runs-on-labels")
    if labels:
        return pick_weighted(labels, "hrr")
    return linux.get("test-runs-on", "")


def select_families(
    event: str,
    config: Config,
    requested: list[str],
    labels: list[str],
    matrix: dict,
    pick_weighted: Callable[[list, str], str],
) -> tuple[list[dict], list[dict]]:
    """GPU families to test and the other ci.yaml families with the reason, in
    ci.yaml order."""
    if requested:
        unknown = [key for key in requested if key not in config.families]
        if unknown:
            raise ConfigError(
                f"requested families {', '.join(unknown)} are not in ci.yaml"
                f" (known: {', '.join(config.families)})"
            )
        chosen = set(requested)
    else:
        chosen = {
            key for key, family in config.families.items()
            if family.enabled and event in family.triggers
        }
    chosen |= set(labelled_families(labels, matrix)) & set(config.families)

    families, skipped = [], []
    for key, family in config.families.items():
        if key not in chosen:
            if requested:
                reason = "not requested"
            elif not family.enabled:
                reason = "disabled in ci.yaml"
            else:
                reason = f"not run on {event} (ci.yaml triggers)"
            skipped.append({"family": key, "reason": reason})
            continue
        linux = matrix.get(key, {}).get("linux", {})
        multi_gpu = event in family.multi_gpu_triggers
        runner = family.runner or runner_for(linux, pick_weighted, multi_gpu)
        if not runner:
            skipped.append({"family": key, "reason": NO_RUNNER})
            continue
        families.append({"family": key, "runner": runner})
    return families, skipped


def decide(
    event: str,
    paths: list[str] | None,
    labels: list[str],
    is_fork: bool,
    requested: list[str],
    config: Config,
    matrix: dict,
    pick_weighted: Callable[[list, str], str],
    added_label: str | None = None,
) -> Decision:
    """paths is None when the change set could not be listed in full.

    added_label is the label a labeled event added; the run for any other
    pull request event already covers a label that cannot change what runs.
    """
    if event not in EVENTS:
        return Decision(False, False, f"event {event} does not run HRR CI")
    if added_label is not None and not label_matters(added_label, config):
        return Decision(False, False, f"the label {added_label} does not change what HRR CI runs")

    if event in ("schedule", "workflow_dispatch"):
        run_cpu, capture, why = True, True, f"{event} run"
    elif paths is None:
        run_cpu, capture, why = True, True, "change set too large to list"
    else:
        run_cpu = touches_cpu(paths, config)
        capture = touches_capture(paths, config)
        why = "the change touches capture or replay" if capture else (
            "the change touches HRR" if run_cpu else "the change does not touch HRR"
        )

    labelled = event == "pull_request" and config.capture_label.lower() in {l.strip().lower() for l in labels}
    if labelled and not capture:
        run_cpu, capture, why = True, True, f"the pull request is labelled {config.capture_label}"

    # Capture paths that the CPU list omits (hip_context.cpp, hip_code_object.*)
    # still need the host unit suite.
    run_cpu = run_cpu or capture
    if not capture:
        return Decision(run_cpu, False, why)
    if is_fork:
        return Decision(run_cpu, False, why + "; GPU suite skipped for a fork pull request")

    families, skipped = select_families(event, config, requested, labels, matrix, pick_weighted)
    no_runner = [s["family"] for s in skipped if s["reason"] == NO_RUNNER]
    if no_runner:
        why += f"; no runner for {', '.join(no_runner)}"
    if not families:
        return Decision(run_cpu, False, why + "; no GPU family with a runner was selected")
    return Decision(run_cpu, True, why, families, skipped)


def is_fork(event: str, head_repo: str, repo: str) -> bool:
    """A pull request from another repository, or from one since deleted."""
    return event == "pull_request" and head_repo != repo


def list_pr_files(repo: str, number: str) -> list[str] | None:
    out = subprocess.run(
        [
            "gh", "api", "--paginate", f"repos/{repo}/pulls/{number}/files",
            "--jq", ".[] | .filename, (.previous_filename // empty)",
        ],
        check=True, capture_output=True, text=True,
    ).stdout
    files = [line for line in out.splitlines() if line]
    return None if len(files) >= MAX_LISTED_FILES else files


def list_push_files(repo: str, before: str, after: str) -> list[str] | None:
    if not before or set(before) == {"0"}:
        return None
    out = subprocess.run(
        [
            "gh", "api", f"repos/{repo}/compare/{before}...{after}",
            "--jq", ".files[] | .filename, (.previous_filename // empty)",
        ],
        check=True, capture_output=True, text=True,
    ).stdout
    files = [line for line in out.splitlines() if line]
    # The compare API lists at most 300 files.
    return None if len(files) >= 300 else files


def load_matrix(therock_dir: Path):
    sys.path.insert(0, str(therock_dir / "build_tools" / "github_actions"))
    import amdgpu_family_matrix  # noqa: E402

    families = amdgpu_family_matrix.get_all_families_for_trigger_types([])
    return families, amdgpu_family_matrix.select_weighted_label


def write_outputs(decision: Decision, rocm_sdk_version: str, output_path: str | None) -> None:
    lines = [
        f"run_cpu={'true' if decision.run_cpu else 'false'}",
        f"run_gpu={'true' if decision.run_gpu else 'false'}",
        f"families={json.dumps(decision.families, separators=(',', ':'))}",
        f"skipped_families={json.dumps(decision.skipped, separators=(',', ':'))}",
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
        "--head-repo", default="", help="owner/name of the pull request's head repository"
    )
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
        matrix, pick_weighted = load_matrix(args.therock_dir)
        check_families(config, matrix)
        check_targets(config, matrix, load_archs(args.integration_config))

        paths: list[str] | None = []
        if args.event == "pull_request" and args.pr_number:
            paths = list_pr_files(args.repo, args.pr_number)
        elif args.event == "push":
            paths = list_push_files(args.repo, args.before, args.after)

        decision = decide(
            args.event, paths, labels, is_fork(args.event, args.head_repo, args.repo), requested,
            config, matrix, pick_weighted,
            added_label=args.label if args.action == "labeled" else None,
        )
    except ConfigError as e:
        print(f"::error::{e}")
        return 1
    write_outputs(decision, config.rocm_sdk_version, os.environ.get("GITHUB_OUTPUT"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
