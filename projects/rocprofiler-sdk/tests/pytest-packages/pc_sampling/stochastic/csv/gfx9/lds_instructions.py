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


def validate_lds_instructions_issued(samples_issued):
    # issued instruction with type == LDS -> instruction starts with ds_
    issued_type_lds = samples_issued[
        samples_issued["Instruction_Type"]
        == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_LDS"
    ]
    assert issued_type_lds["Instruction"].apply(lambda x: x.startswith("ds_")).all()

    # issued instruction starts with ds_ -> it must be LDS
    issued_ds = samples_issued[
        samples_issued["Instruction"].apply(lambda x: x.startswith("ds_"))
    ]
    assert (
        issued_ds["Instruction_Type"] == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_LDS"
    ).all()


def validate_lds_instructions_stalled(samples):
    lds_samples = samples[samples["Instruction"].apply(lambda x: x.startswith("ds_"))]
    lds_stalled = lds_samples[lds_samples["Wave_Issued_Instruction"] == False]

    # TODO: question - why we observed alu_dependency on matrix_multiply_tile kernel
    assert (
        lds_stalled["Stall_Reason"]
        .apply(
            lambda x: x
            == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_WIN_EX_STALL"
            or x
            == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_NO_INSTRUCTION_AVAILABLE"
            or x
            == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_NOT_WIN"
            or x == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ALU_DEPENDENCY"
        )
        .all()
    )


def validate_lds_instructions(samples):
    samples_issued = samples[samples["Wave_Issued_Instruction"]]
    validate_lds_instructions_issued(samples_issued)
    validate_lds_instructions_stalled(samples)
