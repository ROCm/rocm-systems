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


def validate_internal_instructions(sample_records):
    allowed_stall_reasons = set(
        [
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_INTERNAL_INSTRUCTION",
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_NO_INSTRUCTION_AVAILABLE",
        ]
    )
    for record in sample_records:
        assert record["inst"].startswith("s_nop"), "New internal instruction observed"
        assert (
            record["wave_issued"] == 0
        ), "Internal instruction should not be issued to EX"
        assert (
            record["snapshot"]["stall_reason"] in allowed_stall_reasons
        ), "Invalid stall reason for internal instruction"


def validate_waitcnt(sample_records):
    allowed_stall_reasons = set(
        [
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_WAITCNT",
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_NO_INSTRUCTION_AVAILABLE",
        ]
    )
    for record in sample_records:
        assert record["inst"].startswith("s_waitcnt"), "Waitcnt must start with s_waitcn"
        assert record["wave_issued"] == 0, "Waitcnt should not be issued to EX"
        assert (
            record["snapshot"]["stall_reason"] in allowed_stall_reasons
        ), "Invalid stall reason for waitcnt"


def validate_branch_instructions(sample_records):
    allowed_stall_reasons = set(
        [
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_NO_INSTRUCTION_AVAILABLE",
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ALU_DEPENDENCY",
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_NOT_WIN",
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_WIN_EX_STALL",
        ]
    )
    allowed_stall_reasons_uncoditional_branches = set(
        [
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_NO_INSTRUCTION_AVAILABLE",
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_NOT_WIN",
        ]
    )
    for record in sample_records:
        inst = record["inst"]
        inst_type = record["inst_type"]
        snapshot = record["snapshot"]
        stall_reason = snapshot["stall_reason"]
        assert inst.startswith("s_cbranch") or inst.startswith(
            "s_branch"
        ), "Branch must start with s_cbranch or s_branch"

        if record["wave_issued"] == 1:
            if inst.startswith("s_branch"):
                # Uncoditional issued branch can only be branch taken
                assert (
                    inst_type == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_BRANCH_TAKEN"
                ), "Unconditional branch must be taken"
            else:
                # Verifying issued branch instructions
                assert (
                    inst_type == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_BRANCH_TAKEN"
                    or inst_type
                    == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_BRANCH_NOT_TAKEN"
                ), "Invalid branch type for conditional branch instruction"

            assert (
                snapshot["arb_state_issue_misc"] == 1
                and snapshot["arb_state_stall_misc"] == 0
            ), "Invalid arb state for issued branch instruction"

        else:
            # verifying not issued branch instructions
            assert (
                stall_reason in allowed_stall_reasons
            ), "Invalid stall reason for branch instruction"

            if (
                stall_reason
                == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_NOT_WIN"
            ):
                assert (
                    snapshot["arb_state_issue_misc"] == 1
                ), "Arbiter must have issued MISC instruction"

            elif (
                stall_reason
                == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_WIN_EX_STALL"
            ):
                assert (
                    snapshot["arb_state_issue_misc"] == 1
                ), "Arbiter must have issued MISC instruction"

                assert (
                    snapshot["arb_state_stall_misc"] == 1
                ), "Arbiter must have stalled MISC instruction"

            # more specific checks for unconditional branches
            if inst.startswith("s_branch"):
                assert (
                    stall_reason in allowed_stall_reasons_uncoditional_branches
                ), "Invalid stall reason for unconditional branch instruction"


def validate_scalar_instructions(sample_records):
    allowed_stall_reasons = set(
        [
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_NO_INSTRUCTION_AVAILABLE",
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ALU_DEPENDENCY",
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_NOT_WIN",
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_WIN_EX_STALL",
        ]
    )

    for record in sample_records:
        snapshot = record["snapshot"]
        if record["wave_issued"] == 1:
            assert (
                record["inst_type"] == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_SCALAR"
            ), "Invalid scalar instruction type"
            assert (
                snapshot["arb_state_issue_scalar"] == 1
            ), "Arbiter must have issued scalar instruction"
            assert (
                snapshot["arb_state_stall_scalar"] == 0
            ), "Arbiter must have stalled scalar instruction"
        else:
            stall_reason = snapshot["stall_reason"]
            assert (
                stall_reason in allowed_stall_reasons
            ), "Invalid stall reason for scalar instruction"

            if (
                stall_reason
                == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_NOT_WIN"
            ):
                assert (
                    snapshot["arb_state_issue_scalar"] == 1
                ), "Arbiter must have issued scalar instruction"

            elif (
                stall_reason
                == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_WIN_EX_STALL"
            ):
                assert (
                    snapshot["arb_state_issue_scalar"] == 1
                ), "Arbiter must have issued scalar instruction"

                assert (
                    snapshot["arb_state_stall_scalar"] == 1
                ), "Arbiter must have stalled scalar instruction"


def validate_barrier_instructions(sample_records):
    allowed_stall_reasons = set(
        [
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_NO_INSTRUCTION_AVAILABLE",
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_NOT_WIN",
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_BARRIER_WAIT",
        ]
    )
    for record in sample_records:
        assert record["inst"].startswith(
            "s_barrier"
        ), "Barrier instruction must start with s_barrier"
        snapshot = record["snapshot"]
        if record["wave_issued"] == 1:
            assert (
                record["inst_type"] == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_BARRIER"
            ), "Invalid barrier instruction type"
            assert (
                snapshot["arb_state_issue_misc"] == 1
            ), "Arbiter must have issued barrier instruction"
            assert (
                snapshot["arb_state_stall_misc"] == 0
            ), "Arbiter must have stalled barrier instruction"
        else:
            stall_reason = snapshot["stall_reason"]
            assert (
                stall_reason in allowed_stall_reasons
            ), "Invalid stall reason for barrier instruction"

            if (
                stall_reason
                == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_NOT_WIN"
            ):
                assert (
                    snapshot["arb_state_issue_misc"] == 1
                ), "Arbiter must have issued misc instruction"


# TODO: cover other types of instructions
