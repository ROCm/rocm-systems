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
if [[ -n ${ROCPROFV3_TEST_CWORD+x} ]]; then
    COMP_CWORD=${ROCPROFV3_TEST_CWORD}
fi
if [[ -n ${ROCPROFV3_TEST_COMP_LINE+x} ]]; then
    COMP_LINE=${ROCPROFV3_TEST_COMP_LINE}
fi
if [[ -n ${ROCPROFV3_TEST_COMP_POINT+x} ]]; then
    COMP_POINT=${ROCPROFV3_TEST_COMP_POINT}
fi

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
    cword=None,
    comp_line=None,
    comp_point=None,
):
    process_env = os.environ.copy()
    if env:
        process_env.update(env)
    if stub_offset:
        process_env["ROCPROFV3_TEST_STUB_COMMAND_OFFSET"] = "1"
    process_env["ROCPROFV3_TEST_COMP_TYPE"] = str(comp_type)
    process_env["ROCPROFV3_TEST_COLUMNS"] = str(columns)
    if cword is not None:
        process_env["ROCPROFV3_TEST_CWORD"] = str(cword)
    if comp_line is not None:
        process_env["ROCPROFV3_TEST_COMP_LINE"] = comp_line
    if comp_point is not None:
        process_env["ROCPROFV3_TEST_COMP_POINT"] = str(comp_point)

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
        match = re.match(r"^(.*?\S)(?:\s{2,}(.*))?$", line)
        assert match
        fields.append([match.group(1), match.group(2) or ""])
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
    help_options = {
        option
        for declaration in _help_declarations(_help_text(rocprofv3))
        for option in _declaration_options(declaration)
    }
    parser = rocprofv3.build_argument_parser()
    public_actions = [
        action
        for action in parser._actions
        if action.help != rocprofv3.argparse.SUPPRESS and action.option_strings
    ]
    parser_options = {
        option for action in public_actions for option in action.option_strings
    }
    assert parser_options == help_options

    for option in parser_options:
        records = rocprofv3._bash_completion_records(parser, ["rocprofv3", option], 1)
        assert option in {record.split("\t", 1)[0] for record in records}

    expected = {"--"}
    for action in public_actions:
        long_options = [
            option for option in action.option_strings if option.startswith("--")
        ]
        expected.add(long_options[0] if long_options else action.option_strings[0])

    completed = set(_complete(rocprofv3_completion_path, ["rocprofv3", "-"]))
    assert completed == expected, (
        f"missing completion options: {sorted(expected - completed)}; "
        f"stale completion options: {sorted(completed - expected)}"
    )


def test_grouped_option_alias_display(rocprofv3_completion_path):
    fields = _display_fields(
        _complete(
            rocprofv3_completion_path,
            ["rocprofv3", "-"],
            comp_type=9,
            columns=240,
        )
    )
    labels = [label for label, _ in fields]

    assert labels.count("-o, --output-file") == 1
    assert labels.count("-p, --pid, --attach") == 1
    assert labels.count("--output-config") == 1
    assert labels.count("--advanced-thread-trace, --att") == 1

    long_labels = [
        label
        for label, _ in _display_fields(
            _complete(
                rocprofv3_completion_path,
                ["rocprofv3", "--out"],
                comp_type=9,
                columns=240,
            )
        )
    ]
    assert "--output-file (-o)" in long_labels
    assert "--output-directory (-d)" in long_labels
    assert "--output-format (-f)" in long_labels
    assert "--output-config" in long_labels


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


def test_value_completion_with_cursor_before_later_words(rocprofv3_completion_path):
    prefix = "rocprofv3 --output-format rocpd --output-directory " "$PWD --replay-mode "
    line = f"{prefix} -- hip-stream"
    completed = _complete(
        rocprofv3_completion_path,
        [
            "rocprofv3",
            "--output-format",
            "rocpd",
            "--output-directory",
            "$PWD",
            "--replay-mode",
            "--",
            "hip-stream",
        ],
        cword=6,
        comp_line=line,
        comp_point=len(prefix),
    )
    assert completed == ["application", "kernel"]


def test_short_option_values_with_cursor_before_later_words(
    rocprofv3_completion_path,
):
    prefix = "rocprofv3 --output-directory $PWD -f "
    line = f"{prefix} -- hip-stream"
    completed = _complete(
        rocprofv3_completion_path,
        [
            "rocprofv3",
            "--output-directory",
            "$PWD",
            "-f",
            "--",
            "hip-stream",
        ],
        cword=4,
        comp_line=line,
        comp_point=len(prefix),
    )
    assert completed == ["rocpd", "json", "csv", "pftrace", "otf2"]


def test_long_option_completion_does_not_insert_display_padding(
    rocprofv3_completion_path,
):
    prefix = "rocprofv3 -f rocpd -o trace-%hostname% " "--replay-mode kernel --kernel-"
    line = f"{prefix} -- hip-stream"
    words = [
        "rocprofv3",
        "-f",
        "rocpd",
        "-o",
        "trace-%hostname%",
        "--replay-mode",
        "kernel",
        "--kernel-",
        "--",
        "hip-stream",
    ]
    completed = _complete(
        rocprofv3_completion_path,
        words,
        cword=7,
        comp_line=line,
        comp_point=len(prefix),
    )
    assert "--kernel-replay-beta-enabled" in completed
    assert all(value.startswith("--kernel-") for value in completed)

    labels = [
        label
        for label, _ in _display_fields(
            _complete(
                rocprofv3_completion_path,
                words,
                comp_type=9,
                columns=200,
                cword=7,
                comp_line=line,
                comp_point=len(prefix),
            )
        )
    ]
    assert labels
    assert all(label.startswith("--kernel-") for label in labels)


def test_output_file_substitutions(rocprofv3, rocprofv3_completion_path):
    expected = [
        long_form for long_form, _, _ in rocprofv3._BASH_COMPLETION_OUTPUT_SUBSTITUTIONS
    ]
    completed = _complete(
        rocprofv3_completion_path,
        ["rocprofv3", "-o", ""],
    )
    assert completed == expected

    fields = dict(
        _display_fields(
            _complete(
                rocprofv3_completion_path,
                ["rocprofv3", "--output-file", ""],
                comp_type=9,
                columns=160,
            )
        )
    )
    assert fields["%p, %pid%"] == "Process identifier"
    assert fields["%r, %rank%"] == "SLURM or MPI rank"
    assert fields["%env{NAME}%"] == "Value of environment variable NAME"

    assert _complete(
        rocprofv3_completion_path,
        ["rocprofv3", "-o", "results_%pi"],
    ) == ["results_%pid%"]
    assert (
        _complete(
            rocprofv3_completion_path,
            ["rocprofv3", "-o", "trace-%pid%"],
        )
        == []
    )

    appended = _complete(
        rocprofv3_completion_path,
        ["rocprofv3", "-o", "trace-%pid%-"],
    )
    assert appended == [f"trace-%pid%-{value}" for value in expected]


def test_output_file_substitutions_with_cursor_before_later_words(
    rocprofv3, rocprofv3_completion_path
):
    prefix = "rocprofv3 --output-format rocpd -o "
    line = f"{prefix} -- hip-stream"
    completed = _complete(
        rocprofv3_completion_path,
        [
            "rocprofv3",
            "--output-format",
            "rocpd",
            "-o",
            "--",
            "hip-stream",
        ],
        cword=4,
        comp_line=line,
        comp_point=len(prefix),
    )
    expected = [
        long_form for long_form, _, _ in rocprofv3._BASH_COMPLETION_OUTPUT_SUBSTITUTIONS
    ]
    assert completed == expected


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
