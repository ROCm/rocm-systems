# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for skill evaluation grading."""

import argparse
import json
from pathlib import Path
from typing import List

import pytest

from argparser import omniarg_parser
from skills.run_evals import grade

USAGE_ERROR = (
    "usage: rocprof-compute [mode] [options]\n"
    "tool: error: unrecognized arguments: --membw-analysis"
)
COMMAND_FAILURE = ["rocprof-compute command failed with a usage error"]
HELP_OUTPUT = "usage: rocprof-compute [mode] [options]\noptions:\n  -h, --help"
SKILL = "rocprof-compute-memory"


def tool_result(output: str, is_error: bool = False, tool_name: str = "Bash") -> str:
    """Serialize a tool call and its result as stream-json transcript events."""
    command_id = f"test-{tool_name}"
    call = {"type": "tool_use", "name": tool_name, "id": command_id}
    response = {
        "type": "tool_result",
        "tool_use_id": command_id,
        "content": output,
        "is_error": is_error,
    }
    return "\n".join([
        json.dumps({"type": "assistant", "message": {"content": [call]}}),
        json.dumps({"type": "user", "message": {"content": [response]}}),
    ])


@pytest.mark.parametrize(
    ("transcript", "behavioral", "expected_failures"),
    [
        pytest.param(
            tool_result(USAGE_ERROR, is_error=True), False, [], id="routing-mode"
        ),
        pytest.param(
            tool_result(HELP_OUTPUT, is_error=True)
            + "\n"
            + tool_result("tool: error: unrelated failure", is_error=True),
            True,
            [],
            id="usage-and-error-in-separate-results",
        ),
        pytest.param(
            json.dumps({
                "message": {"content": [{"type": "text", "text": USAGE_ERROR}]}
            }),
            True,
            [],
            id="assistant-explanation",
        ),
        pytest.param(
            tool_result(HELP_OUTPUT)
            + "\n"
            + tool_result(USAGE_ERROR, tool_name="Read"),
            True,
            [],
            id="help-then-read-of-error-example",
        ),
        pytest.param(
            tool_result(USAGE_ERROR)
            + "\n"
            + tool_result("3. Memory Chart\n30. Guidance"),
            True,
            COMMAND_FAILURE,
            id="masked-usage-error-then-recovery",
        ),
    ],
)
def test_grade_command_usage_errors(
    transcript: str,
    behavioral: bool,
    expected_failures: List[str],
    tmp_path: Path,
) -> None:
    failures = grade(
        {"skill_should_trigger": True}, SKILL, transcript, [SKILL], tmp_path, behavioral
    )

    assert failures == expected_failures


@pytest.mark.parametrize(
    "arguments",
    [
        pytest.param(
            ["analyze", "--experimental", "--membw-analysis", "-p", "workload/"],
            id="unknown-analyze-option",
        ),
        pytest.param(
            ["analyze", "--decimal", "abc", "-p", "workload/"],
            id="multiline-analyze-usage",
        ),
        pytest.param(["profile", "--name"], id="multiline-profile-usage"),
    ],
)
def test_grade_real_parser_usage_errors(
    arguments: List[str], tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    parser = argparse.ArgumentParser(
        prog="tool", usage="rocprof-compute [mode] [options]"
    )
    omniarg_parser(
        parser, tmp_path, {"gfx950": "MI350"}, {"ver_pretty": "unit test"}, True
    )
    with pytest.raises(SystemExit) as error:
        parser.parse_args(arguments)
    assert error.value.code == 2
    transcript = tool_result(capsys.readouterr().err, is_error=True)

    failures = grade(
        {"skill_should_trigger": True}, SKILL, transcript, [SKILL], tmp_path, True
    )

    assert failures == COMMAND_FAILURE
