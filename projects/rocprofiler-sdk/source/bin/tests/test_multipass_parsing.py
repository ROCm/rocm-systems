#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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

"""GPU-free unit tests for the rocprofv3 multi-pass helpers."""

import json
import os
import sys

import pytest


def _write(tmp_path, name, text):
    path = os.path.join(str(tmp_path), name)
    with open(path, "w") as ofs:
        ofs.write(text)
    return path


def _capture_runs(rocprofv3, monkeypatch, argv):
    calls = []

    def capture_run(app_args, args, **kwargs):
        calls.append((list(app_args), rocprofv3.dotdict(dict(args)), dict(kwargs)))
        return 0

    monkeypatch.setattr(rocprofv3, "run", capture_run)
    assert rocprofv3.main(argv) == 0
    return calls


# --pmc with action="append": one flag -> [["A","B"]] (single-pass, later
# flattened by patch_args); two flags -> [["A"],["B"]] (multi-pass).
def test_single_pmc_flag_is_single_pass(rocprofv3):
    args, _ = rocprofv3.parse_arguments(
        ["--pmc", "SQ_WAVES", "GRBM_COUNT", "--", "./app"]
    )
    assert args.pmc == [["SQ_WAVES", "GRBM_COUNT"]]
    assert len(args.pmc) == 1, "one --pmc flag must be a single group (single-pass)"

    rocprofv3.patch_args(args)
    assert args.pmc == ["SQ_WAVES", "GRBM_COUNT"], "single-pass pmc must be flattened"


def test_multiple_pmc_flags_is_multipass(rocprofv3):
    args, _ = rocprofv3.parse_arguments(
        ["--pmc", "SQ_WAVES", "--pmc", "GRBM_COUNT", "--", "./app"]
    )
    assert args.pmc == [["SQ_WAVES"], ["GRBM_COUNT"]]
    assert len(args.pmc) > 1, "two --pmc flags must be two groups (multi-pass)"

    rocprofv3.patch_args(args)
    assert args.pmc == [
        ["SQ_WAVES"],
        ["GRBM_COUNT"],
    ], "multi-pass groups must be preserved"


def test_parse_input_json_three_jobs(rocprofv3, tmp_path):
    path = _write(
        tmp_path,
        "in.json",
        '{"jobs":[{"pmc":["SQ_WAVES"]},{"pmc":["GRBM_COUNT"]},{"pmc":["GRBM_GUI_ACTIVE"]}]}',
    )
    jobs = rocprofv3.parse_input(path)
    assert len(jobs) == 3
    assert [j["sub_directory"] for j in jobs] == ["pass_", "pass_", "pass_"]
    assert [j["pmc"] for j in jobs] == [["SQ_WAVES"], ["GRBM_COUNT"], ["GRBM_GUI_ACTIVE"]]


def test_parse_input_yaml_three_jobs(rocprofv3, tmp_path):
    path = _write(
        tmp_path,
        "in.yml",
        "jobs:\n"
        "  - pmc:\n    - SQ_WAVES\n"
        "  - pmc:\n    - GRBM_COUNT\n"
        "  - pmc:\n    - GRBM_GUI_ACTIVE\n",
    )
    jobs = rocprofv3.parse_input(path)
    assert len(jobs) == 3
    assert [j["sub_directory"] for j in jobs] == ["pass_", "pass_", "pass_"]
    assert [j["pmc"] for j in jobs] == [["SQ_WAVES"], ["GRBM_COUNT"], ["GRBM_GUI_ACTIVE"]]


def test_parse_input_text_uses_pmc_subdir(rocprofv3, tmp_path):
    path = _write(tmp_path, "in.txt", "pmc: SQ_WAVES\npmc: GRBM_COUNT\n")
    jobs = rocprofv3.parse_input(path)
    assert len(jobs) == 2
    assert [j["sub_directory"] for j in jobs] == ["pmc_", "pmc_"]


@pytest.mark.parametrize(
    "jobs,cli_args,expected",
    [
        (
            [{"pmc": ["SQ_WAVES"]}, {"pmc": ["GRBM_COUNT"]}],
            ["--pid", "12345"],
            "[rocprofv3] Fatal error: Multi-pass counter collection "
            "(multiple input-file jobs) is not compatible with attach mode (--pid)\n",
        ),
        (
            [{"pmc": ["SQ_WAVES"]}, {"pmc": ["GRBM_COUNT"]}],
            ["--collection-period", "0:100:1"],
            "[rocprofv3] Fatal error: Multi-pass counter collection "
            "(multiple input-file jobs) is not compatible with --collection-period\n",
        ),
        (
            [{"pmc": ["SQ_WAVES"]}, {"pmc": ["GRBM_COUNT"], "pid": 12345}],
            [],
            "[rocprofv3] Fatal error: Multi-pass counter collection "
            "(multiple input-file jobs) is not compatible with attach mode (--pid)\n",
        ),
        (
            [
                {"pmc": ["SQ_WAVES"]},
                {"pmc": ["GRBM_COUNT"], "collection_period": ["0:100:1"]},
            ],
            [],
            "[rocprofv3] Fatal error: Multi-pass counter collection "
            "(multiple input-file jobs) is not compatible with --collection-period\n",
        ),
        (
            [{"pmc": ["GRBM_COUNT"]}],
            ["--pmc", "SQ_WAVES", "--pid", "12345"],
            "[rocprofv3] Fatal error: Multi-pass counter collection "
            "(--pmc combined with input-file pmc) is not compatible with attach mode "
            "(--pid)\n",
        ),
    ],
)
def test_multipass_incompatible_options_fail_before_launch(
    rocprofv3, tmp_path, capsys, jobs, cli_args, expected
):
    input_path = _write(
        tmp_path, "input.json", json.dumps({"jobs": jobs}, separators=(",", ":"))
    )
    output_path = tmp_path / "output"

    with pytest.raises(SystemExit) as exc_info:
        rocprofv3.main(
            ["-i", input_path, *cli_args, "-d", str(output_path), "--", "/bin/true"]
        )

    assert exc_info.value.code == 1
    assert capsys.readouterr().err == expected
    assert not output_path.exists()


@pytest.mark.parametrize(
    "cli_output_args,expected_cli_file,expected_cli_directory",
    [
        ([], None, None),
        (["--output-directory", "cli_out"], None, "cli_out"),
        (["--output-file", "cli"], "cli", None),
        (
            ["--output-file", "cli", "--output-directory", "cli_out"],
            "cli",
            "cli_out",
        ),
    ],
)
def test_combined_cli_and_input_passes_keep_independent_output_settings(
    rocprofv3,
    tmp_path,
    monkeypatch,
    cli_output_args,
    expected_cli_file,
    expected_cli_directory,
):
    input_path = _write(
        tmp_path,
        "input.json",
        json.dumps(
            {
                "jobs": [
                    {
                        "pmc": ["SQ_WAVES"],
                        "output_file": "job0",
                        "output_directory": "job0_out",
                    }
                ]
            },
            separators=(",", ":"),
        ),
    )

    calls = _capture_runs(
        rocprofv3,
        monkeypatch,
        [
            "--pmc",
            "GRBM_COUNT",
            "-i",
            input_path,
            *cli_output_args,
            "--output-format",
            "csv",
            "--",
            "/bin/true",
        ],
    )

    assert len(calls) == 2

    cli_app, cli_args, cli_kwargs = calls[0]
    assert cli_app == ["/bin/true"]
    assert cli_args.pmc == ["GRBM_COUNT"]
    assert cli_args.output_file == expected_cli_file
    assert cli_args.output_directory == expected_cli_directory
    assert cli_args.output_format == ["csv"]
    assert cli_kwargs == {"pass_id": 1, "use_execv": False}

    input_app, input_args, input_kwargs = calls[1]
    assert input_app == ["/bin/true"]
    assert input_args.pmc == ["SQ_WAVES"]
    assert input_args.output_file == "job0"
    assert input_args.output_directory == "job0_out"
    assert input_args.output_format == ["csv"]
    assert input_kwargs == {"pass_id": 2, "use_execv": False}


def test_combined_cli_output_settings_do_not_fill_input_job_defaults(
    rocprofv3, tmp_path, monkeypatch
):
    input_path = _write(
        tmp_path,
        "input.json",
        json.dumps({"jobs": [{"pmc": ["SQ_WAVES"]}]}, separators=(",", ":")),
    )

    calls = _capture_runs(
        rocprofv3,
        monkeypatch,
        [
            "--pmc",
            "GRBM_COUNT",
            "-i",
            input_path,
            "--output-file",
            "cli",
            "--output-directory",
            "cli_out",
            "--",
            "/bin/true",
        ],
    )

    assert len(calls) == 2
    assert calls[0][1].output_file == "cli"
    assert calls[0][1].output_directory == "cli_out"
    assert calls[1][1].output_file is None
    assert calls[1][1].output_directory is None


@pytest.mark.parametrize("reverse_jobs", [False, True], ids=["forward", "reversed"])
def test_combined_multiple_cli_and_input_passes_preserve_source_order(
    rocprofv3, tmp_path, monkeypatch, reverse_jobs
):
    jobs = [
        {
            "pmc": ["SQ_WAVES"],
            "output_file": "job_a",
            "output_directory": "job_a_out",
        },
        {
            "pmc": ["GRBM_GUI_ACTIVE"],
            "output_directory": "job_b_out",
        },
    ]
    if reverse_jobs:
        jobs.reverse()

    input_path = _write(
        tmp_path,
        "input.json",
        json.dumps({"jobs": jobs}, separators=(",", ":")),
    )
    calls = _capture_runs(
        rocprofv3,
        monkeypatch,
        [
            "--pmc",
            "GRBM_COUNT",
            "--pmc",
            "TCC_HIT",
            "-i",
            input_path,
            "--output-file",
            "cli",
            "--output-directory",
            "cli_out",
            "--",
            "/bin/true",
        ],
    )

    assert [call[2] for call in calls] == [
        {"pass_id": 1, "use_execv": False},
        {"pass_id": 2, "use_execv": False},
        {"pass_id": 3, "use_execv": False},
        {"pass_id": 4, "use_execv": False},
    ]
    assert [
        (call[1].pmc, call[1].output_file, call[1].output_directory) for call in calls[:2]
    ] == [
        (["GRBM_COUNT"], "cli", "cli_out"),
        (["TCC_HIT"], "cli", "cli_out"),
    ]
    assert [
        (call[1].pmc, call[1].output_file, call[1].output_directory) for call in calls[2:]
    ] == [
        (
            job["pmc"],
            job.get("output_file"),
            job.get("output_directory"),
        )
        for job in jobs
    ]


def test_input_only_job_keeps_custom_output_settings(rocprofv3, tmp_path, monkeypatch):
    input_path = _write(
        tmp_path,
        "input.json",
        json.dumps(
            {
                "jobs": [
                    {
                        "pmc": ["SQ_WAVES"],
                        "output_file": "input",
                        "output_directory": "input_out",
                    }
                ]
            },
            separators=(",", ":"),
        ),
    )

    calls = _capture_runs(
        rocprofv3,
        monkeypatch,
        ["-i", input_path, "--", "/bin/true"],
    )

    assert len(calls) == 1
    assert calls[0][1].output_file == "input"
    assert calls[0][1].output_directory == "input_out"
    assert calls[0][2] == {"pass_id": 1}


def test_configuration_conflict_is_reported_without_traceback(
    rocprofv3, tmp_path, capsys
):
    input_path = _write(
        tmp_path,
        "input.json",
        json.dumps(
            {
                "jobs": [
                    {
                        "pmc": ["SQ_WAVES"],
                        "kernel_include_regex": "job_kernel",
                    }
                ]
            },
            separators=(",", ":"),
        ),
    )

    with pytest.raises(SystemExit) as exc_info:
        rocprofv3.cli_main(
            [
                "--pmc",
                "GRBM_COUNT",
                "-i",
                input_path,
                "--kernel-include-regex",
                "cli_kernel",
                "--",
                "/bin/true",
            ]
        )

    output = capsys.readouterr()
    assert exc_info.value.code == 1
    assert output.out == ""
    assert output.err == (
        "[rocprofv3] Fatal error: Option 'kernel_include_regex' has conflicting "
        "values: cli_kernel vs job_kernel\n"
    )
    assert "Traceback" not in output.err


@pytest.mark.parametrize(
    "name,contents,error_fragment",
    [
        ("invalid.json", "{invalid", "line 1 column"),
        ("missing.json", None, "missing.json"),
    ],
    ids=["invalid-json", "missing-file"],
)
def test_input_file_errors_are_fatal_without_traceback(
    rocprofv3, tmp_path, capsys, name, contents, error_fragment
):
    input_path = os.path.join(str(tmp_path), name)
    if contents is not None:
        _write(tmp_path, name, contents)

    with pytest.raises(SystemExit) as exc_info:
        rocprofv3.cli_main(["-i", input_path, "--", "/bin/true"])

    output = capsys.readouterr()
    assert exc_info.value.code == 1
    assert output.out == ""
    assert output.err.startswith("[rocprofv3] Fatal error: ")
    assert error_fragment in output.err
    assert len(output.err.splitlines()) == 1
    assert "Traceback" not in output.err


if __name__ == "__main__":
    sys.exit(pytest.main(["-x", __file__] + sys.argv[1:]))
