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


def validate_matrix_instructions_issued(samples_issued):
    """
    Validate issued matrix (v_wmma) instructions.

    v_wmma instructions should have
    Instruction_Type == MATRIX.
    """
    # All issued instructions with type MATRIX should start with v_wmma
    issued_type_matrix = samples_issued[
        samples_issued["Instruction_Type"]
        == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_MATRIX"
    ]
    assert issued_type_matrix["Instruction"].apply(
        lambda x: x.startswith("v_wmma")
    ).all(), "All issued MATRIX type instructions should start with v_wmma"

    # All v_wmma issued instructions should have MATRIX instruction type
    v_wmma_issued = samples_issued[
        samples_issued["Instruction"].apply(lambda x: x.startswith("v_wmma"))
    ]
    assert (
        v_wmma_issued["Instruction_Type"]
        == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_MATRIX"
    ).all(), "All issued v_wmma instructions should have MATRIX instruction type"


def validate_matrix_instructions_stalled(samples):
    """
    Validate not-issued (stalled) matrix (v_wmma) instructions.

    For v_wmma instructions that were not issued:
    - The dominant stall reason should be ARBITER_NOT_WIN (arbitration loss)
    - Small percentages of NO_INSTRUCTION_AVAILABLE
    """
    v_wmma_samples = samples[
        samples["Instruction"].apply(lambda x: x.startswith("v_wmma"))
    ]
    v_wmma_stalled = v_wmma_samples[
        v_wmma_samples["Wave_Issued_Instruction"] == False
    ]

    if v_wmma_stalled.empty:
        return

    # Stall reason must be one of ARBITER_NOT_WIN, NO_INSTRUCTION_AVAILABLE,
    # or ARBITER_WIN_EX_STALL
    allowed_stall_reasons = {
        "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_NOT_WIN",
        "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_NO_INSTRUCTION_AVAILABLE",
    }
    assert v_wmma_stalled["Stall_Reason"].apply(
        lambda x: x in allowed_stall_reasons
    ).all(), (
        "All stalled v_wmma instructions should have an allowed stall reason. "
        f"Unexpected reasons: "
        f"{set(v_wmma_stalled['Stall_Reason'].unique()) - allowed_stall_reasons}"
    )


def validate_matrix_instructions(samples):
    """Validate matrix instructions (v_wmma) for both issued and stalled samples."""
    samples_issued = samples[samples["Wave_Issued_Instruction"]]
    validate_matrix_instructions_issued(samples_issued)
    validate_matrix_instructions_stalled(samples)
