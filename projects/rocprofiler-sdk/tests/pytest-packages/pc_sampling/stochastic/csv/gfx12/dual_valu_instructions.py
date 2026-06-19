# MIT License
#
# Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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


from __future__ import absolute_import


def validate_dual_valu_instructions_issued(samples_issued):
    """Validate issued dual VALU (v_dual_*) instructions.

    Issued v_dual_* instructions should have:
    - Instruction_Type == DUAL_VALU
    """
    # All issued instructions with type DUAL_VALU should start with v_dual_
    issued_type_dual_valu = samples_issued[
        samples_issued["Instruction_Type"]
        == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_DUAL_VALU"
    ]
    assert issued_type_dual_valu["Instruction"].apply(
        lambda x: x.startswith("v_dual_")
    ).all(), "All issued DUAL_VALU type instructions should start with v_dual_"

    # All v_dual_ issued instructions should have DUAL_VALU instruction type
    v_dual_issued = samples_issued[
        samples_issued["Instruction"].apply(lambda x: x.startswith("v_dual_"))
    ]
    assert (
        v_dual_issued["Instruction_Type"]
        == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_DUAL_VALU"
    ).all(), "All issued v_dual_ instructions should have DUAL_VALU instruction type"


def validate_dual_valu_instructions_stalled(samples):
    """Validate not-issued (stalled) dual VALU (v_dual_*) instructions.

    For v_dual_ instructions that were not issued, the stall reason must be one of:
    - ARBITER_NOT_WIN (arbitration loss, dominant reason under contention)
    - NO_INSTRUCTION_AVAILABLE
    - OTHER_WAIT
    - ALU_DEPENDENCY
    """
    v_dual_samples = samples[
        samples["Instruction"].apply(lambda x: x.startswith("v_dual_"))
    ]
    v_dual_stalled = v_dual_samples[
        v_dual_samples["Wave_Issued_Instruction"] == False
    ]

    if v_dual_stalled.empty:
        return

    # Stall reason must be one of ARBITER_NOT_WIN, NO_INSTRUCTION_AVAILABLE,
    # or ARBITER_WIN_EX_STALL
    allowed_stall_reasons = {
        "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_NOT_WIN",
        "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_NO_INSTRUCTION_AVAILABLE",
        'ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_OTHER_WAIT',
        'ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ALU_DEPENDENCY'
    }
    assert v_dual_stalled["Stall_Reason"].apply(
        lambda x: x in allowed_stall_reasons
    ).all(), (
        "All stalled v_dual_ instructions should have an allowed stall reason. "
        f"Unexpected reasons: "
        f"{set(v_dual_stalled['Stall_Reason'].unique()) - allowed_stall_reasons}"
    )


def validate_dual_valu_instructions(samples):
    """Validate dual VALU instructions (v_dual_*) for both issued and stalled samples."""
    samples_issued = samples[samples["Wave_Issued_Instruction"]]
    validate_dual_valu_instructions_issued(samples_issued)
    validate_dual_valu_instructions_stalled(samples)
