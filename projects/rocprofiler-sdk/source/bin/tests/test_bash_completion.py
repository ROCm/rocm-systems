#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""GPU-free tests for rocprofv3 Bash completion."""

import os
import re
import subprocess
import sys

import pytest

OPTION_PATTERN = re.compile(r"(?<!\w)-{1,2}[A-Za-z0-9][A-Za-z0-9-]*")

BASH_COMPLETION_DRIVER = r"""
source "$1"
shift

if [[ ${ROCPROFV3_TEST_STUB_COMMAND_OFFSET:-0} == 1 ]]; then
    _command_offset()
    {
        COMPREPLY=("offset:$1")
    }
fi

COMP_WORDS=("$@")
COMP_CWORD=$((${#COMP_WORDS[@]} - 1))
COMP_TYPE=${ROCPROFV3_TEST_COMP_TYPE:-37}
COLUMNS=${ROCPROFV3_TEST_COLUMNS:-160}
COMP_LINE=${COMP_WORDS[0]}
for ((i = 1; i <= COMP_CWORD; ++i)); do
    COMP_LINE+=" ${COMP_WORDS[i]}"
done
COMP_POINT=${#COMP_LINE}

_rocprofv3
if ((${#COMPREPLY[@]})); then
    printf '%s\n' "${COMPREPLY[@]}"
fi
"""


def _complete(
    completion_path,
    words,
    *,
    cwd=None,
    env=None,
    stub_offset=False,
    comp_type=37,
    columns=160,
):
    process_env = os.environ.copy()
    if env:
        process_env.update(env)
    if stub_offset:
        process_env["ROCPROFV3_TEST_STUB_COMMAND_OFFSET"] = "1"
    process_env["ROCPROFV3_TEST_COMP_TYPE"] = str(comp_type)
    process_env["ROCPROFV3_TEST_COLUMNS"] = str(columns)

    result = subprocess.run(
        [
            "bash",
            "--noprofile",
            "--norc",
            "-c",
            BASH_COMPLETION_DRIVER,
            "bash",
            completion_path,
            *words,
        ],
        cwd=cwd,
        env=process_env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.splitlines()


def _help_text(rocprofv3):
    result = subprocess.run(
        [sys.executable, rocprofv3.__file__, "--help"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def _help_declarations(help_text):
    for line in help_text.splitlines():
        if re.match(r"^\s+-", line):
            yield re.split(r"\s{2,}", line.strip(), maxsplit=1)[0]


def _declaration_options(declaration):
    return OPTION_PATTERN.findall(declaration)


def _display_fields(lines):
    fields = []
    for line in lines:
        fields.append(re.split(r"\s{2,}", line, maxsplit=1))
    return fields


@pytest.fixture(autouse=True)
def _use_configured_rocprofv3(monkeypatch, rocprofv3):
    monkeypatch.setenv("_ROCPROFV3_COMPLETION_EXECUTABLE", rocprofv3.__file__)


def test_completion_script_has_valid_bash_syntax(rocprofv3_completion_path):
    subprocess.run(
        ["bash", "-n", rocprofv3_completion_path],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_public_options_match_generated_help(rocprofv3, rocprofv3_completion_path):
    expected = {
        option
        for declaration in _help_declarations(_help_text(rocprofv3))
        for option in _declaration_options(declaration)
    }
    completed = set(_complete(rocprofv3_completion_path, ["rocprofv3", "-"]))

    assert "--" in completed
    completed.remove("--")
    assert completed == expected, (
        f"missing completion options: {sorted(expected - completed)}; "
        f"stale completion options: {sorted(completed - expected)}"
    )


def test_option_descriptions_come_from_parser(rocprofv3, rocprofv3_completion_path):
    action = next(
        action
        for action in rocprofv3.build_argument_parser()._actions
        if "--att-resource-mode" in action.option_strings
    )
    expected = " ".join(action.help.split())
    fields = dict(
        _display_fields(
            _complete(
                rocprofv3_completion_path,
                ["rocprofv3", "--att-"],
                comp_type=9,
                columns=240,
            )
        )
    )
    assert fields["--att-resource-mode"] == expected


def test_choice_values_match_generated_help(rocprofv3, rocprofv3_completion_path):
    mismatches = []
    for declaration in _help_declarations(_help_text(rocprofv3)):
        choices_match = re.search(r"\{([^{}]+)\}", declaration)
        if not choices_match:
            continue

        expected = set(choices_match.group(1).split(","))
        for option in _declaration_options(declaration):
            completed = set(
                _complete(
                    rocprofv3_completion_path,
                    ["rocprofv3", option, ""],
                )
            )
            if completed != expected:
                mismatches.append(
                    f"{option}: expected {sorted(expected)}, got {sorted(completed)}"
                )

    assert not mismatches, "\n".join(mismatches)


def test_boolean_values_match_generated_help(rocprofv3, rocprofv3_completion_path):
    mismatches = []
    for declaration in _help_declarations(_help_text(rocprofv3)):
        if "[BOOL]" not in declaration:
            continue

        for option in _declaration_options(declaration):
            completed = set(
                _complete(
                    rocprofv3_completion_path,
                    ["rocprofv3", option, ""],
                )
            )
            if completed != {"true", "false"}:
                mismatches.append(
                    f"{option}: expected ['false', 'true'], got {sorted(completed)}"
                )

    assert not mismatches, "\n".join(mismatches)


def test_output_format_equals_and_multiple_values(rocprofv3_completion_path):
    assert _complete(
        rocprofv3_completion_path,
        ["rocprofv3", "--output-format=j"],
    ) == ["--output-format=json"]
    assert _complete(
        rocprofv3_completion_path,
        ["rocprofv3", "--output-format", "json", "p"],
    ) == ["pftrace"]


def test_output_format_descriptions_and_deprecations(rocprofv3_completion_path):
    completed = _complete(
        rocprofv3_completion_path,
        ["rocprofv3", "--output-format", ""],
        comp_type=9,
        columns=200,
    )
    fields = _display_fields(completed)

    assert [field[0] for field in fields] == [
        "rocpd",
        "json",
        "csv",
        "pftrace",
        "otf2",
    ]
    assert fields[0][1] == (
        "Complete SQLite output; all services and future additions (default)"
    )
    assert fields[1][1] == ("JSON for ad hoc programmatic analysis; schema is not stable")
    for description in (field[1] for field in fields[2:]):
        assert description.startswith("Deprecated direct")
        assert "export using rocpd convert" in description
        assert "incomplete coverage" in description
        assert "new services are rocpd-only" in description


def test_descriptions_fit_terminal_width(rocprofv3_completion_path):
    completed = _complete(
        rocprofv3_completion_path,
        ["rocprofv3", "--output-format", ""],
        comp_type=9,
        columns=60,
    )
    assert all(len(line) <= 60 for line in completed)
    assert any(line.endswith("...") for line in completed)


def test_meaningful_defaults_are_annotated(rocprofv3_completion_path):
    cases = (
        ("--output-format", "rocpd"),
        ("--ompt-trace", "all"),
        ("--replay-mode", "application"),
        ("--collection-period-unit", "sec"),
        ("--agent-index", "relative"),
        ("--att-resource-mode", "code-object"),
    )
    for option, expected_default in cases:
        fields = _display_fields(
            _complete(
                rocprofv3_completion_path,
                ["rocprofv3", option, ""],
                comp_type=9,
                columns=200,
            )
        )
        defaults = [value for value, description in fields if "(default)" in description]
        assert defaults == [expected_default]


def test_ompt_categories(rocprofv3, rocprofv3_completion_path):
    assert set(
        _complete(
            rocprofv3_completion_path,
            ["rocprofv3", "--ompt-trace", ""],
        )
    ) == {*rocprofv3.OMPT_TRACE_CATEGORIES, "true", "false"}


def test_file_and_directory_values(rocprofv3_completion_path, tmp_path):
    input_file = tmp_path / "profile.yaml"
    input_file.write_text("jobs: []\n")
    output_directory = tmp_path / "results"
    output_directory.mkdir()

    input_prefix = str(tmp_path / "pro")
    assert _complete(
        rocprofv3_completion_path,
        ["rocprofv3", "--input", input_prefix],
    ) == [str(input_file)]
    assert _complete(
        rocprofv3_completion_path,
        ["rocprofv3", f"--input={input_prefix}"],
    ) == [f"--input={input_file}"]

    directory_prefix = str(tmp_path / "res")
    assert _complete(
        rocprofv3_completion_path,
        ["rocprofv3", "--output-directory", directory_prefix],
    ) == [str(output_directory)]


def test_completion_delegates_after_separator(rocprofv3_completion_path):
    assert _complete(
        rocprofv3_completion_path,
        ["rocprofv3", "--hip-trace", "--", "application", "--flag"],
        stub_offset=True,
    ) == ["offset:3"]


def test_application_completion_fallback(rocprofv3_completion_path, tmp_path):
    application = tmp_path / "completion-test-application"
    application.write_text("#!/usr/bin/env bash\n")
    application.chmod(0o755)
    path = f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}"

    assert "completion-test-application" in _complete(
        rocprofv3_completion_path,
        ["rocprofv3", "--", "completion-test"],
        env={"PATH": path},
    )
