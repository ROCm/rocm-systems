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


def validate_jump_instructions_issued(all_samples, jump_samples):
    jump_type_samples_issued = all_samples[
        all_samples["Wave_Issued_Instruction"]
        & (
            all_samples["Instruction_Type"]
            == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_JUMP"
        )
    ]

    jump_samples_issued = jump_samples[jump_samples["Wave_Issued_Instruction"]]
    # sanity check
    assert len(jump_type_samples_issued) == len(jump_samples_issued)
    # repeat checks from above
    assert (
        jump_samples_issued["Instruction_Type"]
        == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_JUMP"
    ).all()


def validate_jump_instructions_stalled(jump_samples):
    jump_samples_stalled = jump_samples[jump_samples["Wave_Issued_Instruction"] == False]
    assert (
        jump_samples_stalled["Stall_Reason"]
        .apply(
            lambda x: x
            == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_NO_INSTRUCTION_AVAILABLE"
        )
        .all()
    )


def validate_jump_instructions(all_samples, jump_samples):
    validate_jump_instructions_issued(all_samples, jump_samples)
    validate_jump_instructions_stalled(jump_samples)
