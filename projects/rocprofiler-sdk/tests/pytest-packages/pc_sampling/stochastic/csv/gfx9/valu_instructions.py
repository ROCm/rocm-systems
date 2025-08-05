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


def validate_valu_instructions_issued(samples_issued):
    # issued instruction with type == VALU -> instruction starts with v_
    issued_type_valu = samples_issued[
        samples_issued["Instruction_Type"]
        == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_VALU"
    ]
    assert issued_type_valu["Instruction"].apply(lambda x: x.startswith("v_")).all()

    # issued instruction starts with v_ and is not matrix instruction -> it must be VALU
    issued_v = samples_issued[
        samples_issued["Instruction"].apply(
            lambda x: x.startswith("v_") and ("mfma" not in x)
        )
    ]
    assert (
        issued_v["Instruction_Type"] == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_VALU"
    ).all()


def validate_valu_instructions_stalled(samples):
    valu_samples = samples[
        samples["Instruction"].apply(lambda x: x.startswith("v_") and ("mfma" not in x))
    ]
    valu_stalled = valu_samples[valu_samples["Wave_Issued_Instruction"] == False]

    assert (
        valu_stalled["Stall_Reason"]
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


def validate_valu_instructions(samples):
    samples_issued = samples[samples["Wave_Issued_Instruction"]]
    validate_valu_instructions_issued(samples_issued)
    validate_valu_instructions_stalled(samples)
