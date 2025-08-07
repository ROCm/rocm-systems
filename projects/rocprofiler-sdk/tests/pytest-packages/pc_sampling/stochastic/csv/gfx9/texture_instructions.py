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


def validate_texture_instructions_issued(samples_issued):
    # issued instruction with type == TEX -> instruction starts with buffer_
    issued_type_texture = samples_issued[
        samples_issued["Instruction_Type"]
        == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_TEX"
    ]
    assert (
        issued_type_texture["Instruction"].apply(lambda x: x.startswith("buffer_")).all()
    )

    # issued instruction starts with buffer_ -> it must be TEX
    issued_buffer = samples_issued[
        samples_issued["Instruction"].apply(lambda x: x.startswith("buffer_"))
    ]
    assert (
        issued_buffer["Instruction_Type"]
        == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_TEX"
    ).all()
    # TODO: find an example with TEX instructions


def validate_texture_instructions_stalled(samples):
    texture_samples = samples[
        samples["Instruction"].apply(lambda x: x.startswith("buffer"))
    ]
    texture_stalled = texture_samples[texture_samples["Wave_Issued_Instruction"] == False]

    assert (
        texture_stalled["Stall_Reason"]
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

    # TODO: find an example with texture instructions


def validate_texture_instructions(samples):
    samples_issued = samples[samples["Wave_Issued_Instruction"]]
    validate_texture_instructions_issued(samples_issued)
    validate_texture_instructions_stalled(samples)
