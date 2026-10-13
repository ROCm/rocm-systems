# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Decide which HRR CI jobs run and on which GPU families.

Writes these GitHub Actions outputs:

  run_cpu    "true" when the generated-source, script and unit jobs run.
  run_gpu    "true" when the capture-enabled build and the GPU suite run.
  families   JSON list of {"family", "runner", "gfx_targets"} for the GPU suite.
  reason     One line explaining the decision, shown in the summary.

The GPU suite needs an amdhip64 with HRR capture compiled in, which only the
HRR GPU workflow builds. It runs automatically when a change touches capture
or replay (CAPTURE_PATHS), on any pull request labelled ci:hrr-capture, on
pushes to develop that touch capture, nightly, and on workflow_dispatch.

GPU families follow TheRock's trigger policy (amdgpu_family_matrix.py):
pull requests test the presubmit families, pushes the postsubmit families and
the nightly schedule the nightly families, each limited to HRR_FAMILIES. A
ci:gfx... label adds that family, the same label TheRock CI uses.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

CAPTURE_LABEL = "ci:hrr-capture"

# A change under any of these runs the generated-source, script and unit jobs.
CPU_PATHS = (
    "projects/hrr/",
    "projects/clr/hipamd/src/hrr/",
    "projects/clr/hipamd/include/hip/amd_detail/hip_api_trace.hpp",
    "projects/hip/include/hip/hip_runtime_api.h",
    ".github/workflows/hrr-ci.yml",
    ".github/workflows/hrr-gpu.yml",
    ".github/scripts/hrr_",
    ".github/scripts/tests/hrr_",
)

# hrr-skills.yml covers these.
EXCLUDED_PATHS = ("projects/hrr/skills/", "projects/hrr/README.md")

# A change under any of these also runs the capture-enabled build and the GPU
# suite. hip_runtime_api.h is deliberately absent: almost every HIP API change
# touches it, and the generated-source job already checks HRR against it.
CAPTURE_PATHS = (
    "projects/clr/hipamd/src/hrr/",
    "projects/clr/hipamd/src/hip_context.cpp",
    "projects/clr/hipamd/src/hip_code_object.cpp",
    "projects/clr/hipamd/src/hip_code_object.hpp",
    "projects/clr/hipamd/include/hip/amd_detail/hip_api_trace.hpp",
    "projects/hrr/CMakeLists.txt",
    "projects/hrr/include/",
    "projects/hrr/playback/",
    "projects/hrr/tests/CMakeLists.txt",
    "projects/hrr/tests/include/",
    "projects/hrr/tests/integration/",
    "projects/hrr/tests/config/integration.yaml",
    "projects/hrr/tests/scripts/run_catch2.py",
    "projects/hrr/tests/scripts/hrr_test_config.py",
    "projects/hrr/tests/scripts/summarize_junit.py",
    ".github/workflows/hrr-ci.yml",
    ".github/workflows/hrr-gpu.yml",
    ".github/scripts/hrr_capture_build.py",
    ".github/scripts/hrr_ci_configure.py",
    ".github/scripts/hrr_ci_report.py",
)

# TheRock family keys the HRR fat binary covers (HRR_ARCHS in hrr-gpu.yml).
# gfx125x is built but only tested on request: its runner pool is not reliably
# online.
HRR_FAMILIES = (
    "gfx90a",
    "gfx94x",
    "gfx950",
    "gfx103x",
    "gfx110x",
    "gfx1150",
    "gfx1151",
    "gfx1153",
    "gfx120x",
    "gfx125x",
)
ON_REQUEST_FAMILIES = ("gfx125x",)
# TheRock tests gfx950 only for submodule bumps; HRR also tests it nightly.
NIGHTLY_EXTRA_FAMILIES = ("gfx950",)
# The T3 tier needs two visible devices, so these families take TheRock's
# multi-GPU runner when it has one.
MULTI_GPU_FAMILIES = ("gfx950",)

TRIGGER_FOR_EVENT = {
    "pull_request": "presubmit",
    "push": "postsubmit",
    "schedule": "nightly",
    "workflow_dispatch": "nightly",
}

# The pull request files API returns at most this many files. A larger change
# is treated as touching everything rather than guessed at.
MAX_LISTED_FILES = 3000


@dataclass
class Decision:
    run_cpu: bool
    run_gpu: bool
    reason: str
    families: list[dict] = field(default_factory=list)


def _matches(path: str, prefixes: Iterable[str]) -> bool:
    return any(path == p or path.startswith(p) for p in prefixes)


def relevant_paths(paths: Iterable[str]) -> list[str]:
    return [p for p in paths if not _matches(p, EXCLUDED_PATHS)]


def touches_cpu(paths: Iterable[str]) -> bool:
    return any(_matches(p, CPU_PATHS) for p in relevant_paths(paths))


def touches_capture(paths: Iterable[str]) -> bool:
    return any(_matches(p, CAPTURE_PATHS) for p in relevant_paths(paths))


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
    trigger: str,
    requested: list[str],
    labels: list[str],
    matrix: dict,
    pick_weighted: Callable[[list, str], str],
) -> list[dict]:
    """GPU families to test, in HRR_FAMILIES order, skipping any without a runner."""
    if requested:
        chosen = set(requested)
    else:
        chosen = {
            key
            for key in HRR_FAMILIES
            if key not in ON_REQUEST_FAMILIES
            and trigger in matrix.get(key, {}).get("linux", {}).get("tests_on_trigger", [])
        }
        if trigger == "nightly":
            chosen |= set(NIGHTLY_EXTRA_FAMILIES)
    chosen |= set(labelled_families(labels, matrix))

    families = []
    for key in HRR_FAMILIES:
        if key not in chosen:
            continue
        linux = matrix.get(key, {}).get("linux", {})
        runner = runner_for(linux, pick_weighted, key in MULTI_GPU_FAMILIES)
        if not runner:
            continue
        families.append(
            {
                "family": key,
                "runner": runner,
                "gfx_targets": ",".join(linux.get("fetch-gfx-targets", [])),
            }
        )
    return families


def decide(
    event: str,
    paths: list[str] | None,
    labels: list[str],
    is_fork: bool,
    requested: list[str],
    matrix: dict,
    pick_weighted: Callable[[list, str], str],
) -> Decision:
    """paths is None when the change set could not be listed in full."""
    trigger = TRIGGER_FOR_EVENT.get(event)
    if trigger is None:
        return Decision(False, False, f"event {event} does not run HRR CI")

    if event in ("schedule", "workflow_dispatch"):
        run_cpu, capture, why = True, True, f"{event} run"
    elif paths is None:
        run_cpu, capture, why = True, True, "change set too large to list"
    else:
        run_cpu = touches_cpu(paths)
        capture = touches_capture(paths)
        why = "the change touches capture or replay" if capture else (
            "the change touches HRR" if run_cpu else "the change does not touch HRR"
        )

    labelled = event == "pull_request" and CAPTURE_LABEL in labels
    if labelled and not capture:
        run_cpu, capture, why = True, True, f"the pull request is labelled {CAPTURE_LABEL}"

    if not capture:
        return Decision(run_cpu, False, why)
    if is_fork:
        return Decision(run_cpu, False, why + "; GPU suite skipped for a fork pull request")

    families = select_families(trigger, requested, labels, matrix, pick_weighted)
    if not families:
        return Decision(run_cpu, False, why + "; no GPU family with a runner was selected")
    return Decision(run_cpu, True, why, families)


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


def write_outputs(decision: Decision, output_path: str | None) -> None:
    lines = [
        f"run_cpu={'true' if decision.run_cpu else 'false'}",
        f"run_gpu={'true' if decision.run_gpu else 'false'}",
        f"families={json.dumps(decision.families, separators=(',', ':'))}",
        f"reason={decision.reason}",
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
    parser.add_argument("--is-fork", default="false")
    parser.add_argument(
        "--families", default="", help="Comma-separated family keys (workflow_dispatch)"
    )
    parser.add_argument("--therock-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    labels = json.loads(args.labels or "[]")
    requested = [f.strip().lower() for f in args.families.split(",") if f.strip()]

    paths: list[str] | None = []
    if args.event == "pull_request" and args.pr_number:
        paths = list_pr_files(args.repo, args.pr_number)
    elif args.event == "push":
        paths = list_push_files(args.repo, args.before, args.after)

    matrix, pick_weighted = load_matrix(args.therock_dir)
    decision = decide(
        args.event, paths, labels, args.is_fork == "true", requested, matrix, pick_weighted
    )
    write_outputs(decision, os.environ.get("GITHUB_OUTPUT"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
