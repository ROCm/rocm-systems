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


def validate_clause_instructions(all_samples, clause_samples):
    """
    Validate s_clause internal instruction.
    """
    total = len(clause_samples)
    if total == 0:
        return

    # s_clause is never issued
    assert (
        clause_samples["Wave_Issued_Instruction"] == False
    ).all(), (
        f"s_clause should never be issued, but "
        f"{(clause_samples['Wave_Issued_Instruction'] == True).sum()}/{total} "
        f"samples have Wave_Issued_Instruction == True"
    )

    # Stall reason must be INTERNAL_INSTRUCTION or OTHER_WAIT
    allowed_stall_reasons = {
        "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_INTERNAL_INSTRUCTION",
        "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_OTHER_WAIT",
    }
    invalid = ~clause_samples["Stall_Reason"].isin(allowed_stall_reasons)
    assert not invalid.any(), (
        f"Unexpected stall reasons for s_clause: "
        f"{set(clause_samples.loc[invalid, 'Stall_Reason'].unique())}"
    )
