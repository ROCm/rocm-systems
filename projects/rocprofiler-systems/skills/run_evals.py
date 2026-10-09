#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run the skill evaluation datasets through Claude Code.

There is no CI for skills, so this is the manual gate: run it before a release
to check that each skill still routes on the prompts it should and stays quiet
on the ones it should not.

Usage:
    ./skills/run_evals.py                      # routing only, no tools
    ./skills/run_evals.py --mode behavioural    # also grade what the agent did
    ./skills/run_evals.py --skill instrumenting-binaries  # one skill
    ./skills/run_evals.py --model sonnet       # pick the model under test
    ./skills/run_evals.py --save-transcripts /tmp/t   # keep transcripts to debug
    ./skills/run_evals.py --list               # validate datasets, run nothing

Exits 0 when every case passes, 1 on a failure, and 77 when the `claude` CLI
is missing so a caller can treat that as skipped rather than broken.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
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
BEHAVIOURAL_KEYS = (*JUDGED_KEYS, "logs_contain", "files_exist")

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "pass": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["pass", "reason"],
}


@dataclass(frozen=True)
class Options:
    """Settings shared by every agent and judge invocation."""

    model: Optional[str]
    isolate: bool
    behavioural: bool
    transcripts: Optional[Path] = None


@dataclass
class Outcome:
    """The result of one case."""

    skill: str
    case_id: str
    failures: list[str]
    visible: set[str]
    model: str = ""


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
        sys.exit(f"{skill.name}: needs at least one positive case with judged behaviour")

    return cases


def check_unique_ids(datasets: dict[str, list[dict[str, Any]]]) -> None:
    """Error if a case id is used by more than one skill."""
    owners: dict[str, str] = {}
    for name, cases in datasets.items():
        for case in cases:
            other = owners.setdefault(case["id"], name)
            if other != name:
                sys.exit(f"case id {case['id']!r} is used by {other} and {name}")


def build_workspace(root: Path) -> Path:
    """Install every skill into a scratch workspace with install-skills.sh."""
    workspace = root / "workspace"
    workspace.mkdir(parents=True)
    subprocess.run(
        [str(SKILLS_ROOT / "install-skills.sh"), "--agent", "claude"],
        env={**os.environ, "HOME": str(workspace)},
        stdout=subprocess.DEVNULL,
        check=True,
    )
    return workspace


def claude_flags(options: Options) -> list[str]:
    """Flags shared by the agent and the judge."""
    flags = ["--no-session-persistence"]
    if options.model:
        flags += ["--model", options.model]
    if options.isolate:
        # Without this the agent also sees the user's personal skills, which can
        # answer a prompt instead of the skill under test.
        flags += ["--setting-sources", "project"]
    return flags


def run_agent(
    prompt: str, workspace: Path, options: Options
) -> tuple[str, list[str], set[str], str]:
    """Run one prompt.

    Returns the transcript text, the skills it loaded, the skills the agent could
    see, and the model it ran on.
    """
    command = [
        "claude",
        "-p",
        prompt,
        "--output-format",
        "stream-json",
        "--verbose",
        "--add-dir",
        str(workspace),
        *claude_flags(options),
    ]
    if options.behavioural:
        command += ["--permission-mode", "acceptEdits"]
    else:
        # Routing only: the decision to load a skill is all we grade, so deny
        # the agent the tools that would make it start doing the work.
        command += ["--disallowedTools", "Bash", "Edit", "Write", "NotebookEdit"]

    result = subprocess.run(
        command,
        cwd=workspace,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=900 if options.behavioural else 300,
        check=False,
    )

    transcript: list[str] = []
    loaded: list[str] = []
    visible: set[str] = set()
    model = ""
    for line in result.stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        transcript.append(line)
        loaded.extend(_skills_in(event))
        if event.get("type") == "system" and event.get("subtype") == "init":
            visible.update(str(name) for name in event.get("skills", []))
            model = str(event.get("model", ""))

    return "\n".join(transcript), loaded, visible, model


def _skills_in(event: dict[str, Any]) -> list[str]:
    """Pull skill names out of Skill tool calls.

    Only the Skill tool counts as loading a skill. Reading a SKILL.md does not:
    on a vague prompt the agent explores the workspace, finds the installed
    skills and reads them, which would fail a case that should not route.
    """
    names: list[str] = []
    # Not every event carries a message object: a permission denial, for example,
    # has "message" set to a plain string.
    message = event.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, list):
        return names

    for block in content:
        is_skill_call = (
            isinstance(block, dict)
            and block.get("type") == "tool_use"
            and block.get("name") == "Skill"
        )
        if is_skill_call:
            payload = block.get("input")
            skill = str(payload.get("skill", "")) if isinstance(payload, dict) else ""
            if skill:
                names.append(skill.split(":")[-1])
    return names


def personal_skills() -> set[str]:
    """Names of the skills in the user's real personal Claude skills folder."""
    root = Path.home() / ".claude" / "skills"
    return {p.parent.name for p in root.glob("*/SKILL.md")}


def grade_judged(
    claim: str, transcript: str, should_hold: bool, options: Options
) -> tuple[bool, str]:
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
            *claude_flags(options),
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
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
    options: Options,
) -> list[str]:
    """Return the list of failure messages for one case, empty when it passes."""
    failures: list[str] = []

    triggered = skill in loaded
    if triggered != case["skill_should_trigger"]:
        want = "trigger" if case["skill_should_trigger"] else "not trigger"
        failures.append(f"expected {skill} to {want}, loaded={loaded or 'none'}")

    if not options.behavioural:
        return failures

    for needle in case.get("logs_contain", []):
        if needle not in transcript:
            failures.append(f"logs missing {needle!r}")

    for artifact in case.get("files_exist", []):
        if not any(p.match(f"**/{artifact}") for p in workspace.rglob("*")):
            failures.append(f"missing artifact {artifact!r}")

    for claim in case.get("expected_behavior", []):
        ok, reason = grade_judged(claim, transcript, True, options)
        if not ok:
            failures.append(f"expected behaviour not met: {claim} ({reason})")

    for claim in case.get("unexpected_behavior", []):
        ok, reason = grade_judged(claim, transcript, False, options)
        if not ok:
            failures.append(f"unexpected behaviour occurred: {claim} ({reason})")

    return failures


def run_case(name: str, case: dict[str, Any], tmp: Path, options: Options) -> Outcome:
    """Run and grade one case."""
    # The agent sees its working directory, so the folder name must not give away
    # the skill under test or the case: an opaque, unique name avoids that.
    folder = hashlib.sha1(f"{name}/{case['id']}".encode()).hexdigest()[:10]
    workspace = build_workspace(tmp / folder)
    try:
        transcript, loaded, visible, model = run_agent(case["prompt"], workspace, options)
    except subprocess.TimeoutExpired:
        return Outcome(name, case["id"], ["timed out"], set())
    if options.transcripts:
        options.transcripts.mkdir(parents=True, exist_ok=True)
        (options.transcripts / f"{name}--{case['id']}.jsonl").write_text(transcript)
    failures = grade(case, name, transcript, loaded, workspace, options)
    return Outcome(name, case["id"], failures, visible, model)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["routing", "behavioural"], default="routing")
    parser.add_argument("--skill", help="Run one skill instead of all of them.")
    parser.add_argument("--list", action="store_true", help="Validate datasets only.")
    parser.add_argument("--model", help="Model to run the agent and the judge on.")
    parser.add_argument(
        "--keep-user-settings",
        action="store_true",
        help="Let the agent also see your personal skills and settings.",
    )
    parser.add_argument(
        "--save-transcripts",
        type=Path,
        metavar="DIR",
        help="Write each case's agent transcript (stream-json) into DIR.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=8,
        help="Cases to run concurrently (DEFAULT: 8). Use 1 to run serially.",
    )
    args = parser.parse_args()

    skills = discover_skills(args.skill)
    datasets = {skill.name: load_cases(skill) for skill in skills}
    check_unique_ids(datasets)

    if args.list:
        for name, cases in datasets.items():
            positive = sum(case["skill_should_trigger"] for case in cases)
            print(f"{name}: {len(cases)} cases ({positive} positive)")
        return 0

    if not shutil.which("claude"):
        print("SKIP: the claude CLI is not installed; skill evals not run.")
        return EXIT_SKIPPED

    options = Options(
        model=args.model,
        isolate=not args.keep_user_settings,
        behavioural=args.mode == "behavioural",
        transcripts=args.save_transcripts,
    )
    if not options.behavioural:
        for cases in datasets.values():
            for case in cases:
                for key in BEHAVIOURAL_KEYS:
                    case.pop(key, None)

    queued = [(name, case) for name, cases in datasets.items() for case in cases]
    started = time.monotonic()
    outcomes: list[Outcome] = []

    # Each case gets its own workspace and its own agent, so they are
    # independent. The work is waiting on a subprocess, not on Python.
    jobs = max(1, args.jobs)
    with tempfile.TemporaryDirectory() as tmp, ThreadPoolExecutor(jobs) as pool:
        futures = [
            pool.submit(run_case, name, case, Path(tmp), options) for name, case in queued
        ]
        for done in as_completed(futures):
            outcome = done.result()
            outcomes.append(outcome)
            print("." if not outcome.failures else "x", end="", flush=True)

    print()
    failed = 0
    for name in sorted(datasets):
        print(f"\n=== {name} ===")
        for outcome in sorted(
            (o for o in outcomes if o.skill == name), key=lambda o: o.case_id
        ):
            if outcome.failures:
                failed += 1
                print(f"FAIL {outcome.case_id}")
                for failure in outcome.failures:
                    print(f"     {failure}")
            else:
                print(f"PASS {outcome.case_id}")

    installed = {skill.name for skill in skills}
    seen = set().union(*(o.visible for o in outcomes))
    competing = sorted((seen & personal_skills()) - installed)
    if competing:
        print(
            f"\nWARNING: {len(competing)} personal skill(s) were visible to the "
            f"agent and can compete with these skills: {', '.join(competing)}"
        )

    models = sorted({o.model for o in outcomes if o.model})
    elapsed = time.monotonic() - started
    total = len(queued)
    print(
        f"\n{total - failed}/{total} passed in {elapsed:.0f}s "
        f"({args.mode} mode, model {', '.join(models) or 'unknown'})"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
