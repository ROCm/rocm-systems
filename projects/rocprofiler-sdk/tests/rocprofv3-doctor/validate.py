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

"""Integration tests: run the real rocprofv3-doctor and check its contract.

These assert only things that hold on any machine, with or without a GPU --
exit code, JSON schema, filter behaviour -- so the suite is meaningful in CI
without requiring particular hardware.
"""

import json
import os
import subprocess
import sys

import pytest

TIMEOUT = 120


def _run(path, args):
    return subprocess.run(
        [sys.executable, path] + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        timeout=TIMEOUT,
    )


def test_validate_doctor_text_report(doctor_path):
    """The default report must render without leaking a traceback."""
    result = _run(doctor_path, ["--no-color"])
    assert result.returncode in (0, 1), result.stderr
    assert "Summary:" in result.stdout
    assert "Traceback (most recent call last)" not in result.stdout
    assert "Traceback (most recent call last)" not in result.stderr


def test_validate_doctor_json_schema(doctor_path):
    result = _run(doctor_path, ["--format", "json"])
    assert result.returncode in (0, 1), result.stderr

    data = json.loads(result.stdout)
    assert data["schema_version"] == 1
    assert "checks" in data
    assert "summary" in data

    total = sum(data["summary"][key] for key in ("pass", "warn", "fail", "skip"))
    assert total == data["summary"]["total"]
    assert total == len(data["checks"])


def test_validate_doctor_json_check_fields(doctor_path):
    result = _run(doctor_path, ["--format", "json"])
    data = json.loads(result.stdout)
    for entry in data["checks"]:
        for key in (
            "id",
            "group",
            "title",
            "severity",
            "status",
            "detail",
            "remediation",
            "data",
        ):
            assert key in entry, "{} missing {}".format(entry.get("id"), key)
        assert entry["status"] in ("pass", "warn", "fail", "skip")


def test_validate_doctor_list_checks(doctor_path):
    result = _run(doctor_path, ["--list-checks"])
    assert result.returncode == 0, result.stderr
    for check_id in ("install.rocm-root", "driver.kfd-device", "python.version"):
        assert check_id in result.stdout


def test_validate_doctor_only_filter(doctor_path):
    result = _run(doctor_path, ["--format", "json", "--only", "driver"])
    assert result.returncode in (0, 1), result.stderr
    data = json.loads(result.stdout)
    assert any(entry["group"] == "driver" for entry in data["checks"])
    # anything else in the run is there as a prerequisite of a selected check
    prerequisites = set()
    for entry in data["checks"]:
        prerequisites.update(entry["depends"])
    for entry in data["checks"]:
        assert entry["group"] == "driver" or entry["id"] in prerequisites, entry["id"]


def test_validate_doctor_skip_filter(doctor_path):
    result = _run(doctor_path, ["--format", "json", "--skip", "python"])
    assert result.returncode in (0, 1), result.stderr
    data = json.loads(result.stdout)
    for entry in data["checks"]:
        assert entry["group"] != "python"


def test_validate_doctor_exit_code_matches_failures(doctor_path):
    """Exit 1 if and only if at least one check failed."""
    result = _run(doctor_path, ["--format", "json"])
    data = json.loads(result.stdout)
    expected = 1 if data["summary"]["fail"] > 0 else 0
    assert result.returncode == expected


def test_validate_doctor_no_color_has_no_escapes(doctor_path):
    result = _run(doctor_path, ["--no-color"])
    assert "\033[" not in result.stdout


def test_validate_doctor_quiet_is_shorter_than_default(doctor_path):
    default = _run(doctor_path, ["--no-color"])
    quiet = _run(doctor_path, ["--no-color", "--quiet"])
    assert len(quiet.stdout) <= len(default.stdout)
    assert "[ PASS ]" not in quiet.stdout


def test_validate_doctor_verbose_is_longer_than_default(doctor_path):
    default = _run(doctor_path, ["--no-color"])
    verbose = _run(doctor_path, ["--no-color", "--verbose"])
    assert len(verbose.stdout) >= len(default.stdout)


def test_validate_doctor_output_file(doctor_path, tmp_path):
    report = os.path.join(str(tmp_path), "report.txt")
    result = _run(doctor_path, ["--no-color", "--output", report])
    assert result.returncode in (0, 1), result.stderr
    assert os.path.exists(report)
    with open(report, "r") as handle:
        assert "Summary:" in handle.read()


def test_validate_doctor_rocm_root_override(doctor_path, tmp_path):
    """An empty --rocm-root must still produce a well-formed report."""
    result = _run(
        doctor_path,
        ["--format", "json", "--rocm-root", str(tmp_path), "--only", "install"],
    )
    assert result.returncode in (0, 1), result.stderr
    data = json.loads(result.stdout)
    assert data["rocm_root"] == str(tmp_path)


def test_validate_doctor_help(doctor_path):
    result = _run(doctor_path, ["--help"])
    assert result.returncode == 0
    assert "--list-checks" in result.stdout


def test_validate_rocprofv3_doctor_help_names_the_command(rocprofv3_path):
    """Users only ever type `rocprofv3 --doctor`; help must say so."""
    result = _run(rocprofv3_path, ["--doctor", "--help"])
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("usage: rocprofv3 --doctor")
    assert "--list-checks" in result.stdout


def test_validate_rocprofv3_doctor_passthrough(rocprofv3_path):
    """rocprofv3 --doctor must behave exactly like the standalone tool."""
    result = _run(rocprofv3_path, ["--doctor", "--format", "json"])
    assert result.returncode in (0, 1), result.stderr
    data = json.loads(result.stdout)
    assert data["schema_version"] == 1


def test_validate_rocprofv3_doctor_passthrough_matches_standalone(
    doctor_path, rocprofv3_path
):
    direct = _run(doctor_path, ["--format", "json"])
    through = _run(rocprofv3_path, ["--doctor", "--format", "json"])
    assert direct.returncode == through.returncode

    direct_ids = [entry["id"] for entry in json.loads(direct.stdout)["checks"]]
    through_ids = [entry["id"] for entry in json.loads(through.stdout)["checks"]]
    assert direct_ids == through_ids


def test_validate_doctor_unknown_filter_is_an_error(doctor_path):
    """An --only typo must not pass as a clean diagnostic run (exit 0)."""
    result = _run(doctor_path, ["--format", "json", "--only", "no-such-group"])
    assert result.returncode == 2, result.stdout
    assert "no-such-group" in result.stderr
    assert "--list-checks" in result.stderr
    assert result.stdout == ""
