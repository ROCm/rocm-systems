# MIT License
#
# Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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


def validate_issued_instruction_type_branch_taken(samples):
    # issued instruction with type BRANCH_TAKEN -> instruction starts with either s_cbranch or s_branch
    issued_type_branch_taken = samples[
        (
            samples["Instruction_Type"]
            == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_BRANCH_TAKEN"
        )
        & samples["Wave_Issued_Instruction"]
    ]
    assert (
        issued_type_branch_taken["Instruction"]
        .apply(lambda x: x.startswith("s_branch") or x.startswith("s_cbranch"))
        .all()
    )
    assert issued_type_branch_taken["Wave_Issued_Instruction"].all()

    # if issued instruction starts with s_branch (unconditional branch) -> its type must be BRANCH_TAKEN
    issued_s_branch = samples[
        samples["Instruction"].apply(lambda x: x.startswith("s_branch"))
        & samples["Wave_Issued_Instruction"]
    ]
    assert (
        issued_s_branch["Instruction_Type"]
        == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_BRANCH_TAKEN"
    ).all()

    # see `validate_issued_instruction_type_branch_not_taken` for more info about s_cbranch checks


def validate_issued_instruction_type_branch_not_taken(samples):
    # issued instruction with type BRANCH_NOT_TAKEN -> instruction is conditional branch (starts s_cbranch)
    issued_type_branch_not_taken = samples[
        (
            samples["Instruction_Type"]
            == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_BRANCH_NOT_TAKEN"
        )
        & samples["Wave_Issued_Instruction"]
    ]
    assert (
        issued_type_branch_not_taken["Instruction"]
        .apply(lambda x: x.startswith("s_cbranch"))
        .all()
    )
    assert issued_type_branch_not_taken["Wave_Issued_Instruction"].all()

    # if issued instruction starts with s_cbranch -> its type is either BRANCH_TAKEN on BRANCH_NOT_TAKEN
    issued_s_cbranch = samples[
        samples["Instruction"].apply(lambda x: x.startswith("s_cbranch"))
        & samples["Wave_Issued_Instruction"]
    ]
    assert (
        (
            issued_s_cbranch["Instruction_Type"]
            == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_BRANCH_TAKEN"
        )
        | (
            issued_s_cbranch["Instruction_Type"]
            == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_BRANCH_NOT_TAKEN"
        )
    ).all()


def s_branch_not_issued(stalled_samples):
    s_branch_stalled = stalled_samples[
        stalled_samples["Instruction"].apply(lambda x: x.startswith("s_branch"))
    ]

    if len(s_branch_stalled) > 0:
        # No ALUDEP nor ARBWINEXSTALL observed so far for unconditional branches
        assert (
            s_branch_stalled["Stall_Reason"]
            .apply(
                lambda x: x
                != "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ALU_DEPENDENCY"
                and x
                != "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_WIN_EX_STALL"
            )
            .all()
        )


def validate_stalled_branches(samples):
    stalled_samples = samples[samples["Wave_Issued_Instruction"] == False]

    assert (
        stalled_samples["Stall_Reason"]
        .apply(
            lambda x: x
            == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ALU_DEPENDENCY"
            or x
            == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_NO_INSTRUCTION_AVAILABLE"
            or x
            == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_NOT_WIN"
            or x
            == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_WIN_EX_STALL"
        )
        .all()
    )

    # Further constraints for unconditional branches
    s_branch_not_issued(stalled_samples)


def validate_branch_instructions(all_samples, branch_samples):
    """
    Use all_samples to verify the ROCProfV3 determines `Instruction_Type` field properly.

    Use filtered_samples to verify both issued and stalled branch instructions.
    """
    # For the issued branches, use all samples, as the called functions will do
    # separation based on branch type (conditional or unconditional)
    validate_issued_instruction_type_branch_taken(all_samples)
    validate_issued_instruction_type_branch_not_taken(all_samples)

    # stalled branches
    validate_stalled_branches(branch_samples)
