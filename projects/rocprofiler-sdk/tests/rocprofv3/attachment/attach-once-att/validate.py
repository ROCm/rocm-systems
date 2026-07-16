#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

import sys
import json
import re
import pytest
from pathlib import Path

# ---------------------------------------------------------------------------
# Basic sanity checks on the results JSON
# ---------------------------------------------------------------------------


def test_agent_info(json_data):
    """Results JSON must list at least one GPU agent."""
    data = json_data["rocprofiler-sdk-tool"]

    gpu_count = 0
    for agent in data["agents"]:
        # type 1 = CPU, type 2 = GPU
        assert agent["type"] in (1, 2)
        if agent["type"] == 1:
            assert agent["cpu_cores_count"] > 0
            assert agent["simd_count"] == 0
            assert agent["max_waves_per_simd"] == 0
        else:
            gpu_count += 1
            assert agent["cpu_cores_count"] == 0
            assert agent["simd_count"] > 0
            assert agent["max_waves_per_simd"] > 0

    assert gpu_count > 0, "No GPU agents found in results JSON"


def test_att_filenames_in_json(json_data):
    """The results JSON must report that ATT data was produced (att_filenames key)."""
    data = json_data["rocprofiler-sdk-tool"]
    strings = data["strings"]
    assert "att_filenames" in strings.keys(), (
        "att_filenames key missing from results JSON — "
        "ATT data was not captured during attachment"
    )
    att_files = strings["att_filenames"]
    assert len(att_files) > 0, "att_filenames list is empty — no ATT data captured"


# ---------------------------------------------------------------------------
# ATT decoded output checks (ui_output_agent_* directories)
# ---------------------------------------------------------------------------


def test_att_ui_output_dirs_exist(att_output_dir):
    """At least one ui_output_agent_* directory must be present."""
    ui_dirs = [p for p in Path(att_output_dir).glob("ui_output_agent_*") if p.is_dir()]
    assert len(ui_dirs) > 0, (
        f"No ui_output_agent_* directories found in {att_output_dir}. "
        "ATT decoder did not produce output."
    )


def test_att_ui_output_dir_name_pattern(att_output_dir):
    """ui_output_agent_* directory names must match ui_output_agent_<id>_dispatch_<id>."""
    ui_dirs = [p for p in Path(att_output_dir).glob("ui_output_agent_*") if p.is_dir()]
    assert len(ui_dirs) > 0, f"No ui_output_agent_* directories found in {att_output_dir}"

    pattern = re.compile(r"ui_output_agent_(\d+)_dispatch_(\d+)$")
    for ui_dir in ui_dirs:
        m = pattern.search(ui_dir.name)
        assert m, (
            f"ui_output directory '{ui_dir.name}' does not match expected pattern "
            "'ui_output_agent_<id>_dispatch_<id>'"
        )


def _iter_wave_filenames(wave_filenames):
    """Yield (coord_str, filename, begin, end) from the nested wave_filenames dict.

    wave_filenames structure:
      { SE: { SM: { SL: { WV: [filename, begin, end], ... }, ... }, ... }, ... }
    """
    for se, sm_dict in wave_filenames.items():
        for sm, sl_dict in sm_dict.items():
            for sl, wv_dict in sl_dict.items():
                for wv, entry in wv_dict.items():
                    coord = f"se{se}_sm{sm}_sl{sl}_wv{wv}"
                    yield coord, entry[0], entry[1], entry[2]


def test_att_filenames_manifest(att_output_dir):
    """Every ui_output_agent_* dir must have a valid filenames.json with wave files,
    and every listed wave file must exist on disk."""
    ui_dirs = [p for p in Path(att_output_dir).glob("ui_output_agent_*") if p.is_dir()]
    assert len(ui_dirs) > 0, f"No ui_output_agent_* directories found in {att_output_dir}"

    for ui_dir in ui_dirs:
        filenames_path = ui_dir / "filenames.json"
        assert filenames_path.exists(), f"filenames.json missing in {ui_dir}"

        with open(filenames_path) as f:
            filenames_json = json.load(f)

        wave_filenames = filenames_json.get("wave_filenames", {})
        assert (
            len(wave_filenames) > 0
        ), f"wave_filenames is empty in {filenames_path} — no wave data was decoded"

        # Verify every listed wave file exists on disk
        for coord, filename, begin, end in _iter_wave_filenames(wave_filenames):
            wave_file = ui_dir / filename
            assert wave_file.exists(), (
                f"Wave file {filename} listed in filenames.json "
                f"(coord {coord}) not found on disk: {wave_file}"
            )


def test_att_wave_file_content(att_output_dir):
    """Wave JSON files must contain valid instruction trace data."""
    ui_dirs = [p for p in Path(att_output_dir).glob("ui_output_agent_*") if p.is_dir()]
    assert len(ui_dirs) > 0, f"No ui_output_agent_* directories found in {att_output_dir}"

    found_wave_data = False
    for ui_dir in ui_dirs:
        filenames_path = ui_dir / "filenames.json"
        if not filenames_path.exists():
            continue

        with open(filenames_path) as f:
            filenames_json = json.load(f)

        wave_filenames = filenames_json.get("wave_filenames", {})
        for coord, filename, begin, end in _iter_wave_filenames(wave_filenames):
            wave_file = ui_dir / filename
            if not wave_file.exists():
                continue

            with open(wave_file) as f:
                wave_data = json.load(f)

            # Wave files contain a dict with "wave" key holding the trace
            assert isinstance(
                wave_data, dict
            ), f"Wave file {wave_file} should be a JSON object, got {type(wave_data)}"
            assert (
                "wave" in wave_data
            ), f"Wave file {wave_file} missing 'wave' key; keys: {list(wave_data.keys())}"
            wave = wave_data["wave"]
            assert (
                "instructions" in wave
            ), f"Wave file {wave_file}: 'wave' missing 'instructions' key"
            assert (
                len(wave["instructions"]) > 0
            ), f"Wave file {wave_file}: instructions list is empty"
            # Validate begin/end times match the manifest
            assert (
                wave["begin"] == begin
            ), f"Wave file {wave_file}: begin time {wave['begin']} != manifest {begin}"
            assert (
                wave["end"] == end
            ), f"Wave file {wave_file}: end time {wave['end']} != manifest {end}"
            found_wave_data = True

    assert (
        found_wave_data
    ), "No wave files with content found across all ui_output_agent_* directories"


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
