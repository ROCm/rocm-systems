# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Fixture staging and deterministic checks for the `decode-and-triage` evals.

Prompts and judged expectations live in ``evals.json``. skillscope, the runner
the AMD skills catalogue grades with, imports this file and calls ``setup``
before each behavioral case and ``check`` after it:

    skillscope behavioral --skill decode-and-triage

Every case works from recorded evidence, the replay logs and ``--info`` output
under ``tests/fixtures/``, so none of them needs a GPU or an archive. ``setup``
copies them into the agent's workspace under the names the prompt uses.

``check`` holds the assertions a dataset field cannot make: what the final
reply says, and what the agent must not have run. Whether the skill loaded is
decided from the agent's own tool calls, never by searching the log for the
skill's name: the init event lists every installed skill, so the name is in
every log and a check on it could not fail.

The false-activation cases are the ``skill_should_trigger: false`` entries in
``evals.json``. ``skillscope routing`` runs them with the skill installed and
grades them on the run log's tool calls, so they need nothing here.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

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


def _check_fault(run) -> None:
    assert FAULT_ADDRESS in run.result_text, "fault address missing from the reply"
    assert FAULT_KERNEL in run.result_text, "faulting kernel missing from the reply"


def _check_unreadable(run) -> None:
    ran = json.dumps([args for _, args in _tool_uses(run.logs)])
    assert "--repair" not in ran, "ran --repair unasked"


def _check_clean_pass(run) -> None:
    assert "PASS" in run.result_text, "the reply does not report the clean pass"


# Case id -> (fixtures staged as {workspace name: fixture}, reply check).
CASES = {
    "triage-recorded-memory-fault": (
        {"replay.log": "replay_memory_fault.log", "info.txt": "info_crash.txt"},
        _check_fault,
    ),
    "triage-unreadable-archive": (
        {"replay.log": "version_mismatch.log"},
        _check_unreadable,
    ),
    "triage-clean-pass-of-crashing-workload": (
        {"replay.log": "replay_pass.log"},
        _check_clean_pass,
    ),
}


def setup(workspace, case, ctx):
    """Copy the case's recorded evidence into the agent's workspace."""
    staged, _ = CASES.get(case.id, ({}, None))
    for name, fixture in staged.items():
        shutil.copyfile(FIXTURES / fixture, Path(workspace) / name)
    return {}


def check(run, case, ctx):
    """Fail the case unless the skill loaded and the reply passes its check."""
    if case.id not in CASES:
        return
    assert skill_loaded(run.logs), "the skill was not loaded"
    CASES[case.id][1](run)
