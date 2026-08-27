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

# Expected VALU issue and stall metadata for v_mov_b32 samples.
VALU_INSTRUCTION_TYPE = "ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_VALU"
ARBITER_WIN_EX_STALL_REASON = (
    "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_WIN_EX_STALL"
)
ALLOWED_V_MOV_B32_STALL_REASONS = {
    ARBITER_WIN_EX_STALL_REASON,
    "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_NO_INSTRUCTION_AVAILABLE",
    "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_NOT_WIN",
    "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ALU_DEPENDENCY",
}


def _has_gfx12_agent(json_data):
    # Architecture-specific arbiter checks are not implemented for gfx12.
    agents = pcs_json.get_tool(json_data)["agents"]
    gpu_agents = [agent for agent in agents if agent["type"] == 2]
    assert gpu_agents, "No GPU agents found"
    return any(agent["gfx_target_version"] // 10000 == 12 for agent in gpu_agents)


def _validate_v_mov_b32_semantics(json_data):
    # Validate issue and stall semantics for decoded v_mov_b32 samples.
    tool = pcs_json.get_tool(json_data)
    records = pcs_json.get_records(json_data, METHOD)
    instructions = tool["strings"]["pc_sample_instructions"]

    v_mov_b32_records = []
    for sample in records:
        inst_index = sample["inst_index"]
        if inst_index == -1:
            continue

        assert inst_index >= 0
        if instructions[inst_index].startswith("v_mov_b32"):
            v_mov_b32_records.append(sample["record"])

    # Prevent this validation from passing without exercising v_mov_b32.
    assert v_mov_b32_records, "no decoded v_mov_b32 records found"

    for record in v_mov_b32_records:
        wave_issued = record["wave_issued"]
        snapshot = record["snapshot"]
        assert wave_issued in (0, 1)

        if wave_issued == 1:
            assert record["inst_type"] == VALU_INSTRUCTION_TYPE
            assert snapshot["arb_state_issue_valu"] == 1
            assert snapshot["arb_state_stall_valu"] == 0
        else:
            stall_reason = snapshot["stall_reason"]
            assert (
                stall_reason in ALLOWED_V_MOV_B32_STALL_REASONS
            ), f"unexpected non-issued v_mov_b32 stall reason: {stall_reason}"
            if stall_reason == ARBITER_WIN_EX_STALL_REASON:
                # The VALU pipe won arbitration but stalled before execution.
                assert snapshot["arb_state_issue_valu"] == 1


def test_validate_pc_sampling_selected_regions_csv(pc_csv):
    # Validate the common CSV schema, volume, and sample fields.
    pcs_csv.validate_columns(pc_csv, METHOD)
    pcs_csv.validate_sample_volume(pc_csv)
    pcs_csv.validate_values(pc_csv)


def test_validate_pc_sampling_selected_regions_json(pc_csv, json_data, request):
    pcs_json.validate_csv_json_parity_num_samples(pc_csv, json_data, METHOD)
    pcs_json.validate_data_integrity(json_data, METHOD)
    # Validate gating according to the selected-region mode.
    if request.config.getoption("--ref-count"):
        pcs_json.validate_selected_regions_ref_count_gating(json_data, METHOD)
    else:
        pcs_json.validate_selected_regions_gating(json_data, METHOD)


def test_validate_pc_sampling_stochastic_specific_csv(pc_csv):
    # Validate fields emitted only by stochastic sampling.
    assert pc_csv["Wave_Issued_Instruction"].isin([0, 1]).all()
    assert (pc_csv["Wave_Count"] > 0).all()
    assert pc_csv["Instruction_Type"].str.startswith(INSTRUCTION_TYPE_PREFIX).all()
    assert pc_csv["Stall_Reason"].str.startswith(STALL_REASON_PREFIX).all()


def test_validate_pc_sampling_stochastic_specific_json(json_data):
    # Validate fields emitted only by stochastic sampling.
    for rec in pcs_json.get_records(json_data, METHOD):
        r = rec["record"]
        assert r["wave_issued"] in (0, 1)
        assert r["wave_cnt"] > 0
        assert r["inst_type"].startswith(INSTRUCTION_TYPE_PREFIX)
        assert r["snapshot"]["stall_reason"].startswith(STALL_REASON_PREFIX)


def test_validate_pc_sampling_stochastic_v_mov_b32_semantics(json_data):
    if _has_gfx12_agent(json_data):
        pytest.skip("v_mov_b32 arbiter semantic checks are not implemented for GFX12")

    _validate_v_mov_b32_semantics(json_data)


if __name__ == "__main__":
    sys.exit(pytest.main(["-x", __file__] + sys.argv[1:]))
