#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Run the skill evaluation datasets through Claude Code.

There is no CI for skills, so this is the manual gate: run it before a release
to check that each skill still routes on the prompts it should and stays quiet
on the ones it should not.

Usage:
    ./skills/run_evals.py                      # routing only, no tools
    ./skills/run_evals.py --mode behavioral    # also grade what the agent did
    ./skills/run_evals.py --skill memory       # one skill
    ./skills/run_evals.py --list               # validate datasets, run nothing

Exits 0 when every case passes, 1 on a failure, and 77 when the `claude` CLI
is missing so a caller can treat that as skipped rather than broken.
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

SKILLS_ROOT = Path(__file__).resolve().parent
EXIT_SKIPPED = 77

# Only these keys are understood. A typo is an error rather than an expectation
# that silently never runs.
KNOWN_KEYS = {
    "id",
    "prompt",
    "skill_should_trigger",
    "note",
    "expected_behavior",
    "unexpected_behavior",
    "logs_contain",
    "files_exist",
}
JUDGED_KEYS = ("expected_behavior", "unexpected_behavior")
BEHAVIORAL_KEYS = (*JUDGED_KEYS, "logs_contain", "files_exist")

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "pass": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["pass", "reason"],
}


def discover_skills(only: Optional[str]) -> list[Path]:
    """Return skill directories that ship an evals dataset."""
    found = sorted(p.parent.parent for p in SKILLS_ROOT.glob("*/evals/evals.json"))
    if only:
        found = [p for p in found if p.name == only]
        if not found:
            sys.exit(f"No skill named {only!r} with an evals dataset.")
    return found


def load_cases(skill: Path) -> list[dict[str, Any]]:
    """Parse and validate one dataset, erroring on anything malformed."""
    dataset = json.loads((skill / "evals" / "evals.json").read_text())
    cases = dataset["evaluations"]

    positive = 0
    negative = 0
    judged = 0
    seen: set[str] = set()

    for case in cases:
        unknown = set(case) - KNOWN_KEYS
        if unknown:
            sys.exit(
                f"{skill.name}: unknown key(s) {sorted(unknown)} in {case.get('id')}"
            )
        if not isinstance(case.get("skill_should_trigger"), bool):
            sys.exit(
                f"{skill.name}: {case.get('id')} needs a boolean skill_should_trigger"
            )
        if case["id"] in seen:
            sys.exit(f"{skill.name}: duplicate case id {case['id']!r}")
        seen.add(case["id"])

        if case["skill_should_trigger"]:
            positive += 1
            judged += any(case.get(key) for key in JUDGED_KEYS)
        else:
            negative += 1
            extra = set(case) - {"id", "prompt", "skill_should_trigger", "note"}
            if extra:
                sys.exit(
                    f"{skill.name}: {case['id']} is negative but sets {sorted(extra)}"
                )

    if positive < 3 or negative < 2:
        sys.exit(f"{skill.name}: needs at least 3 positive and 2 negative cases")
    if not judged:
        sys.exit(f"{skill.name}: needs at least one positive case with judged behavior")

    return cases


def build_workspace(root: Path) -> Path:
    """Copy every skill into a scratch workspace so the agent can discover them."""
    workspace = root / "workspace"
    skills_dir = workspace / ".claude" / "skills"
    skills_dir.mkdir(parents=True)
    for skill in sorted(SKILLS_ROOT.glob("*/SKILL.md")):
        shutil.copytree(skill.parent, skills_dir / skill.parent.name)
    return workspace


def run_agent(prompt: str, workspace: Path, behavioral: bool) -> tuple[str, list[str]]:
    """Run one prompt and return the transcript text and the skills it loaded."""
    command = [
        "claude",
        "-p",
        prompt,
        "--output-format",
        "stream-json",
        "--verbose",
        "--add-dir",
        str(workspace),
    ]
    if behavioral:
        command += ["--permission-mode", "acceptEdits"]
    else:
        # Routing only: the decision to load a skill is all we grade, so deny
        # the agent the tools that would make it start doing the work.
        command += ["--disallowedTools", "Bash", "Edit", "Write", "NotebookEdit"]

    result = subprocess.run(
        command,
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=900 if behavioral else 300,
    )

    transcript: list[str] = []
    loaded: list[str] = []
    for line in result.stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        transcript.append(line)
        loaded.extend(_skills_in(event))

    return "\n".join(transcript), loaded


def _skills_in(event: dict[str, Any]) -> list[str]:
    """Pull skill names out of Skill tool calls and SKILL.md reads."""
    names: list[str] = []
    content = event.get("message", {}).get("content")
    if not isinstance(content, list):
        return names

    for block in content:
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            continue
        payload = block.get("input", {})
        if block.get("name") == "Skill":
            skill = str(payload.get("skill", ""))
            names.append(skill.split(":")[-1])
        elif block.get("name") == "Read":
            path = str(payload.get("file_path", ""))
            if path.endswith("SKILL.md"):
                names.append(Path(path).parent.name)
    return names


def grade_judged(claim: str, transcript: str, should_hold: bool) -> tuple[bool, str]:
    """Ask a second agent whether a plain-language claim holds in the transcript."""
    question = (
        "You are grading an AI agent transcript against one expectation.\n"
        f"Expectation: {claim}\n"
        f"The expectation should be {'met' if should_hold else 'NOT met'}.\n"
        "Answer only with the JSON object.\n\n"
        f"Transcript:\n{transcript[:200000]}"
    )
    result = subprocess.run(
        [
            "claude",
            "-p",
            question,
            "--output-format",
            "json",
            "--json-schema",
            json.dumps(JUDGE_SCHEMA),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    try:
        payload = json.loads(result.stdout)
        verdict = payload if "pass" in payload else json.loads(payload["result"])
    except (json.JSONDecodeError, KeyError, TypeError):
        return False, "judge returned unparseable output"

    held = bool(verdict.get("pass"))
    return held == should_hold, str(verdict.get("reason", ""))


def grade(
    case: dict[str, Any],
    skill: str,
    transcript: str,
    loaded: list[str],
    workspace: Path,
    behavioral: bool,
) -> list[str]:
    """Return the list of failure messages for one case, empty when it passes."""
    failures: list[str] = []

    triggered = skill in loaded
    if triggered != case["skill_should_trigger"]:
        want = "trigger" if case["skill_should_trigger"] else "not trigger"
        failures.append(f"expected {skill} to {want}, loaded={loaded or 'none'}")

    if not behavioral:
        return failures

    for needle in case.get("logs_contain", []):
        if needle not in transcript:
            failures.append(f"logs missing {needle!r}")

    for artifact in case.get("files_exist", []):
        if not any(p.match(f"**/{artifact}") for p in workspace.rglob("*")):
            failures.append(f"missing artifact {artifact!r}")

    for claim in case.get("expected_behavior", []):
        ok, reason = grade_judged(claim, transcript, should_hold=True)
        if not ok:
            failures.append(f"expected behavior not met: {claim} ({reason})")

    for claim in case.get("unexpected_behavior", []):
        ok, reason = grade_judged(claim, transcript, should_hold=False)
        if not ok:
            failures.append(f"unexpected behavior occurred: {claim} ({reason})")

    return failures


def run_case(
    name: str, case: dict[str, Any], tmp: Path, behavioral: bool
) -> tuple[str, str, list[str]]:
    """Run and grade one case. Returns (skill, case id, failure messages)."""
    workspace = build_workspace(tmp / f"{name}-{case['id']}")
    try:
        transcript, loaded = run_agent(case["prompt"], workspace, behavioral)
    except subprocess.TimeoutExpired:
        return name, case["id"], ["timed out"]
    failures = grade(case, name, transcript, loaded, workspace, behavioral)
    return name, case["id"], failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["routing", "behavioral"], default="routing")
    parser.add_argument("--skill", help="Run one skill instead of all of them.")
    parser.add_argument("--list", action="store_true", help="Validate datasets only.")
    parser.add_argument(
        "--jobs",
        type=int,
        default=8,
        help="Cases to run concurrently (DEFAULT: 8). Use 1 to run serially.",
    )
    args = parser.parse_args()

    skills = discover_skills(args.skill)
    datasets = {skill.name: load_cases(skill) for skill in skills}

    if args.list:
        for name, cases in datasets.items():
            positive = sum(case["skill_should_trigger"] for case in cases)
            print(f"{name}: {len(cases)} cases ({positive} positive)")
        return 0

    if not shutil.which("claude"):
        print("SKIP: the claude CLI is not installed; skill evals not run.")
        return EXIT_SKIPPED

    behavioral = args.mode == "behavioral"
    if not behavioral:
        for cases in datasets.values():
            for case in cases:
                for key in BEHAVIORAL_KEYS:
                    case.pop(key, None)

    queued = [(name, case) for name, cases in datasets.items() for case in cases]
    started = time.monotonic()
    results: dict[str, list[tuple[str, list[str]]]] = {name: [] for name in datasets}

    # Each case gets its own workspace and its own agent, so they are
    # independent. The work is waiting on a subprocess, not on Python.
    with tempfile.TemporaryDirectory() as tmp:
        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            futures = {
                pool.submit(run_case, name, case, Path(tmp), behavioral): name
                for name, case in queued
            }
            for done in as_completed(futures):
                name, case_id, failures = done.result()
                results[name].append((case_id, failures))
                print("." if not failures else "x", end="", flush=True)

    print()
    failed = 0
    for name, cases in sorted(results.items()):
        print(f"\n=== {name} ===")
        for case_id, failures in sorted(cases):
            if failures:
                failed += 1
                print(f"FAIL {case_id}")
                for failure in failures:
                    print(f"     {failure}")
            else:
                print(f"PASS {case_id}")

    elapsed = time.monotonic() - started
    total = len(queued)
    print(f"\n{total - failed}/{total} passed in {elapsed:.0f}s ({args.mode} mode)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
