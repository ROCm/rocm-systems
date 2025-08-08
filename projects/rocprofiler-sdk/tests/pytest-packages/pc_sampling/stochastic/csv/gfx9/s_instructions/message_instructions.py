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


def validate_message_instructions_issued(all_samples, message_samples):
    message_type_samples_issued = all_samples[
        all_samples["Wave_Issued_Instruction"]
        & (
            all_samples["Instruction_Type"]
            == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_MESSAGE"
        )
    ]

    message_samples_issued = message_samples[message_samples["Wave_Issued_Instruction"]]
    # sanity check
    assert len(message_type_samples_issued) == len(message_samples_issued)
    # repeat checks from above
    assert (
        message_samples_issued["Instruction_Type"]
        == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_MESSAGE"
    ).all()
    # TODO: find an example with messages


def validate_message_instructions_stalled(message_samples):
    message_samples_stalled = message_samples[
        message_samples["Wave_Issued_Instruction"] == False
    ]
    assert (
        message_samples_stalled["Stall_Reason"]
        .apply(
            lambda x: x
            == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_NO_INSTRUCTION_AVAILABLE"
            or x == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ALU_DEPENDENCY"
        )
        .all()
    )


def validate_message_instructions(all_samples, message_samples):
    validate_message_instructions_issued(all_samples, message_samples)
    validate_message_instructions_stalled(message_samples)
