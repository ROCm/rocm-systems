# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Behavioral evals for the `decode-and-triage` skill.

These run under the AMD skills catalogue's behavioral harness, which stages the
skill into an isolated workspace, runs a real agent once per test and grades
what it did:

    cd eval/behavioral
    python -m pytest -c pytest.ini -p conftest \\
        ../../skills/decode-and-triage/evals/evals.py

Every case works from recorded evidence, the replay logs and `--info` output
under `tests/fixtures/`, so none of them needs a GPU or an archive.

Whether the skill loaded is decided on the run log by `skill_loaded`, never by
searching it for the skill's name: the agent's init event lists every skill in
the workspace, so the name is in every log, and a check on it could not fail.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from harness import claude

SKILL = "decode-and-triage"
FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"

# fixtures/replay_memory_fault.log with the archive listing in info_crash.txt:
# a memory access fault on the archive's only kernel.
FAULT_ADDRESS = "0x7f91da000000"
FAULT_KERNEL = "crash_oob"


def _tool_uses(logs: str):
    """Yield (name, input) for every tool call in a stream-json run log."""
    stack = []
    for line in logs.splitlines():
        try:
            stack.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    while stack:
        obj = stack.pop()
        if isinstance(obj, dict):
            if obj.get("type") == "tool_use" and isinstance(obj.get("input"), dict):
                yield str(obj.get("name", "")), obj["input"]
            stack.extend(obj.values())
        elif isinstance(obj, list):
            stack.extend(obj)


def skill_loaded(logs: str) -> bool:
    """True when the agent loaded this skill or ran one of its scripts."""
    for name, args in _tool_uses(logs):
        if name == "Skill" and str(args.get("skill", "")).split(":")[-1] == SKILL:
            return True
        path = str(args.get("file_path", "")).replace("\\", "/")
        if name == "Read" and path.endswith(f"/{SKILL}/SKILL.md"):
            return True
        command = str(args.get("command", "")).replace("\\", "/")
        if name == "Bash" and f"{SKILL}/scripts/" in command:
            return True
    return False


def _stage(agent, fixture: str, name: str) -> None:
    shutil.copyfile(FIXTURES / fixture, agent.workspace / name)


def test_fault_from_recorded_evidence():
    """A recorded fault becomes a finding with its class, address and kernel."""
    with claude("opus", skill=SKILL) as agent:
        _stage(agent, "replay_memory_fault.log", "replay.log")
        _stage(agent, "info_crash.txt", "info.txt")

        run = agent.prompt(
            "A customer's HRR archive faulted on replay. The archive stayed with "
            "them, but replay.log is the replay output and info.txt is "
            "`hrr-playback --info` for it. Triage it."
        )

        assert skill_loaded(run.logs), "the skill was not loaded"
        assert FAULT_ADDRESS in run.result_text, "fault address missing from the reply"
        assert FAULT_KERNEL in run.result_text, "faulting kernel missing from the reply"

        run.should("Report a GPU memory access fault, outcome MAF or fault class illegal_memory_access")
        run.should(
            f"Say that {FAULT_KERNEL} was inferred because the archive holds one kernel, "
            "or suggest confirming it with --sync-after-launch"
        )
        run.should_not("Invent a root cause the evidence does not support")


def test_unreadable_archive_is_not_a_workload_failure():
    """A format mismatch means nothing was replayed, so nothing faulted."""
    with claude("opus", skill=SKILL) as agent:
        _stage(agent, "version_mismatch.log", "replay.log")

        run = agent.prompt("Our HRR replay stopped at once. What happened? The output is in replay.log.")

        assert skill_loaded(run.logs), "the skill was not loaded"
        assert "--repair" not in json.dumps(list(_tool_uses(run.logs))), "ran --repair unasked"

        run.should("Report that the archive format and the hrr-playback reader disagree, so nothing was replayed")
        run.should("Say a reader matching the runtime that captured is needed")
        run.should_not("Report a GPU fault or name a faulting kernel")


def test_clean_pass_of_a_crashing_workload():
    """A clean replay of a workload that crashed is itself a finding."""
    with claude("opus", skill=SKILL) as agent:
        _stage(agent, "replay_pass.log", "replay.log")

        run = agent.prompt(
            "This workload crashed for our customer with a memory access fault. "
            "Here is the replay of its HRR archive: replay.log. What happened?"
        )

        assert skill_loaded(run.logs), "the skill was not loaded"
        assert "PASS" in run.result_text, "the reply does not report the clean pass"

        run.should("Say the fault did not reproduce on replay, or ask for the original failure signature")
        run.should_not("Name any kernel as implicated in a failure")
        run.should_not("Claim to have reproduced the customer's memory access fault")


# Prompts that must not load it. evals/evals.json carries the routing set that
# is graded on whether the skill was chosen; these are graded on the run log.
SHOULD_NOT_TRIGGER = [
    "Read notes.txt and summarize it in one sentence.",
    # No archive yet: recording the failure is enable-recording's job.
    "Our training job crashes on an MI300 with a memory access fault. "
    "What do we send AMD so they can reproduce it?",
    # A core dump is the crash-analysis tools' job.
    "My HIP application dumped core. Open core.4242 with rocgdb and show me the backtrace.",
]


@pytest.mark.parametrize("prompt", SHOULD_NOT_TRIGGER)
def test_routes_out(prompt):
    """False activation: with the skill available, these must not load it."""
    with claude("opus", skill=SKILL) as agent:
        (agent.workspace / "notes.txt").write_text(
            "Sprint notes: rotate the API token, then update the changelog.\n",
            encoding="utf-8",
        )

        run = agent.prompt(prompt)

        assert not skill_loaded(run.logs), "the skill loaded on a prompt it does not cover"
