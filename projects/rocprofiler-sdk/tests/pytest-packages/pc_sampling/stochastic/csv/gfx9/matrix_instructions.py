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
    # issued instruction with type == MATRIX -> instruction starts with v_mfma
    issued_type_matrix = samples_issued[
        samples_issued["Instruction_Type"]
        == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_MATRIX"
    ]
    assert issued_type_matrix["Instruction"].apply(lambda x: x.startswith("v_mfma")).all()
    # v_mfma_f32 goes through Matrix (MAI) arbiter, while v_mfma_f64 goes through the VALU arbiter

    # SGEMM goes through Matrix (MAI arbiter)
    v_mfma_f32_issued = samples_issued[
        samples_issued["Instruction"].apply(lambda x: x.startswith("v_mfma_f32"))
    ]
    assert (
        v_mfma_f32_issued["Instruction_Type"]
        == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_MATRIX"
    ).all()

    # DGEMM goes through VALU arbiter
    v_mfma_f64_issued = samples_issued[
        samples_issued["Instruction"].apply(lambda x: x.startswith("v_mfma_f64"))
    ]
    assert (v_mfma_f64_issued["Instruction_Type"] == "MATRIX").all()
    assert len(issued_type_matrix) == len(v_mfma_f32_issued) + len(v_mfma_f64_issued)

    # TODO: find an example with MAI instructions


def validate_dgemm_matrix_instructions_stalled(samples):
    v_mfma_f64_samples = samples[
        samples["Instruction"].apply(lambda x: x.startswith("v_mfma_f64"))
    ]
    v_mfma_f64_stalled = v_mfma_f64_samples[
        v_mfma_f64_samples["Wave_Issued_Instruction"] == False
    ]

    assert (
        v_mfma_f64_stalled["Stall_Reason"]
        .apply(
            lambda x: x
            == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_WIN_EX_STALL"
            or x
            == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_NO_INSTRUCTION_AVAILABLE"
            or x
            == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_NOT_WIN"
        )
        .all()
    )


def validate_sgemm_matrix_instructions_stalled(samples):
    v_mfma_f32_samples = samples[
        samples["Instruction"].apply(lambda x: x.startswith("v_mfma_f32"))
    ]
    v_mfma_f32_stalled = v_mfma_f32_samples[
        v_mfma_f32_samples["Wave_Issued_Instruction"] == False
    ]
    assert (
        v_mfma_f32_stalled["Stall_Reason"]
        .apply(
            lambda x: x
            == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_WIN_EX_STALL"
            or x
            == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_NO_INSTRUCTION_AVAILABLE"
            or x
            == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_NOT_WIN"
        )
        .all()
    )


def validate_matrix_instructions_stalled(samples):
    validate_dgemm_matrix_instructions_stalled(samples)
    validate_sgemm_matrix_instructions_stalled(samples)
    # TODO" find an example to test this


def validate_matrix_instructions(samples):
    samples_issued = samples[samples["Wave_Issued_Instruction"]]
    validate_matrix_instructions_issued(samples_issued)
    validate_matrix_instructions_stalled(samples)
