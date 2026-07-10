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
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Validate rocprofv3 stochastic PC sampling output collected under --selected-regions.

import sys

import pytest

from rocprofiler_sdk.pc_sampling.selected_regions import csv as pcs_csv
from rocprofiler_sdk.pc_sampling.selected_regions import json as pcs_json

METHOD = "stochastic"

INSTRUCTION_TYPE_PREFIX = "ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_"
STALL_REASON_PREFIX = "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_"


def _records(json_data):
    tool = json_data["rocprofiler-sdk-tool"]
    tool = tool[0] if isinstance(tool, list) else tool
    return tool["buffer_records"]["pc_sample_stochastic"]


def test_validate_pc_sampling_selected_regions_csv(pc_csv):
    # CSV schema, sample volume, and per-row values
    pcs_csv.validate_columns(pc_csv, METHOD)
    pcs_csv.validate_sample_volume(pc_csv)
    pcs_csv.validate_values(pc_csv)


def test_validate_pc_sampling_selected_regions_json(pc_csv, json_data, request):
    pcs_json.validate_csv_json_parity(pc_csv, json_data, METHOD)
    pcs_json.validate_data_integrity(json_data, METHOD)
    # collection restricted to the resume regions
    if request.config.getoption("--ref-count"):
        pcs_json.validate_selected_regions_ref_count_gating(json_data, METHOD)
    else:
        pcs_json.validate_selected_regions_gating(json_data, METHOD)


def test_validate_pc_sampling_stochastic_specific_csv(pc_csv):
    # stochastic-only columns are well-formed
    assert pc_csv["Wave_Issued_Instruction"].isin([0, 1]).all()
    assert (pc_csv["Wave_Count"] > 0).all()
    assert pc_csv["Instruction_Type"].str.startswith(INSTRUCTION_TYPE_PREFIX).all()
    assert pc_csv["Stall_Reason"].str.startswith(STALL_REASON_PREFIX).all()


def test_validate_pc_sampling_stochastic_specific_json(json_data):
    # stochastic-only fields are well-formed in the JSON records
    for rec in _records(json_data):
        r = rec["record"]
        assert r["wave_issued"] in (0, 1)
        assert r["wave_cnt"] > 0
        assert r["inst_type"].startswith(INSTRUCTION_TYPE_PREFIX)
        assert r["snapshot"]["stall_reason"].startswith(STALL_REASON_PREFIX)


if __name__ == "__main__":
    sys.exit(pytest.main(["-x", __file__] + sys.argv[1:]))
