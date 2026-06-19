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


def validate_delay_alu_instructions(all_samples, delay_alu_samples):
    """
    Validate s_delay_alu internal instructions.

    s_delay_alu is a pseudo-instruction that is never issued to the execution pipeline.
    The PC sampling correction WA ensures that no skid samples for s_delay_alu reach
    the validator, so all s_delay_alu samples must appear as not-issued.
    """
    total = len(delay_alu_samples)
    if total == 0:
        return

    assert (delay_alu_samples["Wave_Issued_Instruction"] == False).all(), (
        f"s_delay_alu must never be issued to EX, but "
        f"{(delay_alu_samples['Wave_Issued_Instruction'] == True).sum()}/{total} "
        f"samples have Wave_Issued_Instruction == True"
    )

    # Stall reason must be one of the allowed reasons
    allowed_stall_reasons = {
        "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_INTERNAL_INSTRUCTION",
        "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_OTHER_WAIT",
        "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_NO_INSTRUCTION_AVAILABLE"
    }
    assert delay_alu_samples["Stall_Reason"].apply(
        lambda x: x in allowed_stall_reasons
    ).all(), (
        "All s_delay_alu instructions should have an allowed stall reason. "
        f"Unexpected reasons: "
        f"{set(delay_alu_samples['Stall_Reason'].unique()) - allowed_stall_reasons}"
    )
