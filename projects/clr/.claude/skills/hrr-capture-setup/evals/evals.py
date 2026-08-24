# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Behavioural evals for the `hrr-capture-setup` skill.

The unit tests beside the scripts check the archive inspector. These check the
part no parser can: whether the skill is reached for from a symptom, whether the
agent explains an empty archive correctly, and whether it stays quiet when the
question is not about recording a failure.

Two tiers:

* **Recorded-evidence tier** (default). Runs against tool output recorded from a
  real capture on a gfx950 host, checked in under ``fixtures/``. No GPU needed,
  so this is the tier CI can run today.
* **Live-capture tier**. Actually records a workload. Skipped unless
  ``/dev/kfd`` and ``HRR_EVAL_CAPTURE_COMMAND`` are both present.

Run where the catalogue harness can see this skill:

    python -m pytest -c pytest.ini -p conftest <path-to>/evals/evals.py
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from harness import claude

SKILL = "hrr-capture-setup"

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _stage(agent, fixture: str, name: str | None = None) -> Path:
    dest = agent.workspace / (name or fixture)
    shutil.copyfile(FIXTURES / fixture, dest)
    return dest


def _live_capture_skip_reason() -> str | None:
    if not Path("/dev/kfd").exists():
        return "no /dev/kfd: this host has no AMD GPU"
    if not os.environ.get("HRR_EVAL_CAPTURE_COMMAND", "").strip():
        return "HRR_EVAL_CAPTURE_COMMAND is not set to a workload command"
    return None


LIVE_CAPTURE_SKIP = _live_capture_skip_reason()


def test_a_symptom_reaches_for_capture():
    """The user describes a failure and never names the tool.

    This is the whole point of the skill: someone who already knows to ask for
    record and replay does not need it.
    """
    with claude("opus", skill=SKILL) as agent:
        run = agent.prompt(
            "Our inference server on an MI300 node dies every few hours with "
            "'Memory access fault by GPU node-4'. We cannot share the model or the "
            "code with you. What should we send you so you can debug it?"
        )

        run.logs_contains(SKILL)

        run.should("Propose recording the failing run into an archive that can be replayed")
        run.should(
            "Check first that the HIP runtime the workload loads actually has capture "
            "built in, rather than only giving the environment variable"
        )
        run.should("Say the archive has to be verified before it is sent")

        run.should_not("Ask for the model weights, the source code or a reproducer script")
        run.should_not("Recommend a profiler")


def test_capture_is_not_offered_for_a_performance_question():
    """False-activation screen.

    The skill is staged in the workspace, so an over-eager description shows up
    here. Asserted on what the agent ran, not only on the judge's reading: the
    skill's whole effect is invoking its scripts, and its name appears in the
    transcript from the workspace listing alone.
    """
    with claude("opus", skill=SKILL) as agent:
        run = agent.prompt(
            "Our GEMM kernel on MI300 reaches only 40% of peak. How do I find out "
            "which part of it is slow?"
        )

        for script in ("hrr_capture.sh", "inspect_archive.py"):
            assert script not in run.command_text, (
                f"a performance question invoked the skill's {script}:\n{run.command_text}"
            )

        run.should_not("Propose recording the workload into an archive to find the slow part")


def test_an_empty_archive_is_blamed_on_the_runtime_not_the_workload():
    """The most common failure, and the one with the most misleading symptom."""
    with claude("opus", skill=SKILL) as agent:
        _stage(agent, "verify_empty.txt", "verify_output.txt")

        run = agent.prompt(
            "I set HIP_HRR_CAPTURE_OUTPUT and ran my training job. It ran fine but the "
            "archive came out like this: verify_output.txt. What went wrong?"
        )

        run.logs_contains(SKILL)

        run.should(
            "Explain that the HIP runtime the workload loaded has no capture compiled in, "
            "so the environment variable was ignored"
        )
        run.should(
            "Point out that a Python workload usually loads the runtime shipped inside the "
            "torch wheel rather than the one in the ROCm install"
        )
        run.should("Tell the user to run the skill's preflight in the workload's own environment")

        run.should_not("Suggest the workload made no HIP calls")
        run.should_not("Suggest the archive is corrupt")


def test_a_missing_archive_is_not_confused_with_an_empty_one():
    """Nothing was created at all, which is a different problem from a capture
    that ran and recorded nothing."""
    with claude("opus", skill=SKILL) as agent:
        _stage(agent, "verify_no_archive.txt", "verify_output.txt")

        run = agent.prompt("My capture produced this. What happened? verify_output.txt")

        run.logs_contains(SKILL)

        run.should("Say that capture never started because no archive directory was created")
        run.should(
            "Give both candidate causes: the workload died before its first GPU call, "
            "or the runtime it loaded has no capture support"
        )

        run.should_not("Say the workload made no GPU calls as if that were established")
        run.should_not("Suggest the archive was deleted or corrupted")


def test_an_incomplete_archive_is_not_treated_as_a_failed_capture():
    """No trailer means the process died, which is the case worth keeping."""
    with claude("opus", skill=SKILL) as agent:
        _stage(agent, "verify_incomplete.txt", "verify_output.txt")

        run = agent.prompt("Is this capture usable? verify_output.txt")

        run.logs_contains(SKILL)

        run.should("Say the archive is usable and worth sending")
        run.should(
            "Explain that the missing clean-shutdown trailer is the expected signature of "
            "a workload that crashed, not corruption"
        )

        run.should_not("Tell the user to capture again because the archive is broken")
        run.should_not("Name a kernel or diagnose the crash, which is a different skill's job")


def test_sharing_advice_does_not_invent_a_scrubbing_option():
    """There is no randomisation or safe mode today, and saying otherwise is worse
    than saying nothing: it invites someone to upload a customer's data."""
    with claude("opus", skill=SKILL) as agent:
        _stage(agent, "verify_recorded.txt", "verify_output.txt")

        run = agent.prompt(
            "Capture worked: verify_output.txt. Our legal team wants to know what is "
            "inside the archive before we upload it to a vendor. What do we tell them?"
        )

        run.logs_contains(SKILL)

        run.should(
            "State that the recorded buffers hold the real workload data in the clear, "
            "including prompts and generated text"
        )
        run.should_not("Claim the archive can be anonymised, scrubbed or randomised by a flag")
        run.should_not("Claim the archive contains only metadata or only API calls")


def test_multi_process_archive_is_kept_together():
    with claude("opus", skill=SKILL) as agent:
        _stage(agent, "verify_multiprocess.txt", "verify_output.txt")

        run = agent.prompt(
            "Here is what our capture produced: verify_output.txt. Which directory do "
            "I send?"
        )

        run.logs_contains(SKILL)

        run.should("Say to send the whole archive directory rather than one pid directory")
        run.should_not("Pick a single pid directory as the one to send")


@pytest.mark.skipif(
    LIVE_CAPTURE_SKIP is not None,
    reason=f"live capture unavailable: {LIVE_CAPTURE_SKIP}",
)
def test_live_capture_of_a_real_workload():
    """Live tier: record a real workload end to end on an AMD GPU host.

    Set ``HRR_EVAL_CAPTURE_COMMAND`` to a command that uses the GPU. Nothing is
    checked in for this, so it stays opt-in and out of CI until a GPU runner
    that can reach the agent API exists.
    """
    command = os.environ["HRR_EVAL_CAPTURE_COMMAND"]

    with claude("opus", skill=SKILL) as agent:
        run = agent.prompt(
            f"This workload is misbehaving and I want to record it so someone else can "
            f"replay it: {command}"
        )

        run.logs_contains(SKILL)
        run.should("Run the skill's preflight before capturing")
        run.should("Report afterwards whether the archive actually holds events")

        run.should_not("Change the workload's GPU selection")
        run.should_not("Report success without checking the archive")
