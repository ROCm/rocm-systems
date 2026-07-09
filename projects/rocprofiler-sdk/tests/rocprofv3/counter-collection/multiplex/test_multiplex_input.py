#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

# GPU-free unit tests of rocprofv3's multiplex input parsing (parse_input only).

import os
import sys
import json
import subprocess
import pytest


def _layout(jobs):
    return [
        ([list(group) for group in job["pmc_groups"]], job.get("pmc_group_interval"))
        for job in jobs
    ]


def _write_yaml(path, layout, interval):
    lines = ["jobs:", f"  - pmc_groups: {json.dumps(layout)}"]
    if interval is not None:
        lines.append(f"    pmc_group_interval: {interval}")
    path.write_text("\n".join(lines) + "\n")


def _write_json(path, layout, interval):
    job = {"pmc_groups": layout}
    if interval is not None:
        job["pmc_group_interval"] = interval
    path.write_text(json.dumps({"jobs": [job]}) + "\n")


def test_yaml_and_json_parse_to_identical_layout(rocprofv3_module, tmp_path):
    """The same multiplex layout in YAML and JSON must parse identically."""
    layout = [["SQ_WAVES", "GRBM_COUNT"], ["GRBM_GUI_ACTIVE"]]
    interval = 2

    yml = tmp_path / "same.yml"
    js = tmp_path / "same.json"
    _write_yaml(yml, layout, interval)
    _write_json(js, layout, interval)

    parsed_yaml = _layout(rocprofv3_module.parse_input(str(yml)))
    parsed_json = _layout(rocprofv3_module.parse_input(str(js)))

    assert parsed_yaml == parsed_json, f"YAML {parsed_yaml} != JSON {parsed_json}"
    assert parsed_yaml == [(layout, interval)]


def test_group_order_is_preserved(rocprofv3_module, tmp_path):
    """Group ordering (which drives round-robin rotation) must be preserved."""
    layout = [["SQ_WAVES"], ["GRBM_COUNT"], ["GRBM_GUI_ACTIVE"]]

    js = tmp_path / "g3.json"
    _write_json(js, layout, 2)

    parsed = _layout(rocprofv3_module.parse_input(str(js)))
    assert parsed == [(layout, 2)]


def test_interval_optional(rocprofv3_module, tmp_path):
    """A layout without pmc_group_interval parses; YAML and JSON agree."""
    layout = [["SQ_WAVES"], ["GRBM_COUNT"]]

    yml = tmp_path / "noint.yml"
    js = tmp_path / "noint.json"
    _write_yaml(yml, layout, None)
    _write_json(js, layout, None)

    parsed_yaml = _layout(rocprofv3_module.parse_input(str(yml)))
    parsed_json = _layout(rocprofv3_module.parse_input(str(js)))

    assert parsed_yaml == parsed_json
    assert parsed_yaml[0][0] == layout


def test_pmc_and_pmc_groups_are_mutually_exclusive(rocprofv3_path, tmp_path):
    """--pmc and pmc_groups cannot be combined; rocprofv3 rejects it up front
    (GPU-free: the launcher fatal-errors before executing the application)."""
    mux = tmp_path / "mux.json"
    _write_json(mux, [["SQ_WAVES"], ["GRBM_COUNT"]], 1)

    proc = subprocess.run(
        [rocprofv3_path, "--pmc", "SQ_WAVES", "-i", str(mux), "--", "/bin/true"],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "ROCR_VISIBLE_DEVICES": ""},
    )

    output = proc.stdout + proc.stderr
    assert proc.returncode != 0, f"expected non-zero exit, got 0.\n{output}"
    assert "Cannot specify both --pmc" in output, f"unexpected error output:\n{output}"


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
