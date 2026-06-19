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


def validate_internal_instructions(sample_records):
    allowed_stall_reasons = set(
        [
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_INTERNAL_INSTRUCTION",
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_NO_INSTRUCTION_AVAILABLE",
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_OTHER_WAIT",
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_SLEEP_WAIT",
        ]
    )

    # The PC sampling correction WA ensures that no skid samples for internal
    # instructions reach the validator. All internal instructions must not be
    # issued to EX, and ARBITER_NOT_WIN skid samples are also corrected away.
    for record in sample_records:
        assert record["inst"].startswith("s_nop") or record["inst"].startswith(
            "s_sleep"
        ), f"Unexpected internal instruction: {record['inst']}"
        assert (
            record["wave_issued"] == 0
        ), f"Internal instruction must not be issued to EX: {record['inst']}"
        stall_reason = record["snapshot"]["stall_reason"]
        assert (
            stall_reason in allowed_stall_reasons
        ), f"Invalid stall reason for internal instruction: {stall_reason}"


def validate_waitcnt(sample_records):
    allowed_stall_reasons = set(
        [
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_WAITCNT",
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_NO_INSTRUCTION_AVAILABLE",
        ]
    )

    # The PC sampling correction WA ensures that no skid samples for s_wait_*
    # instructions reach the validator. s_wait_* instructions are never issued on
    # GFX12, and ARBITER_NOT_WIN skid samples are also corrected away.
    for record in sample_records:
        assert record["inst"].startswith("s_wait"), "Waitcnt must start with s_wait"
        assert (
            record["wave_issued"] == 0
        ), f"s_wait_* must not be issued to EX: {record['inst']}"
        stall_reason = record["snapshot"]["stall_reason"]
        assert (
            stall_reason in allowed_stall_reasons
        ), f"Invalid stall reason for waitcnt: {stall_reason}"


def validate_branch_instructions(sample_records):
    allowed_stall_reasons = set(
        [
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_NO_INSTRUCTION_AVAILABLE",
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ALU_DEPENDENCY",
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_NOT_WIN",
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_OTHER_WAIT",
        ]
    )
    allowed_stall_reasons_uncoditional_branches = set(
        [
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_NO_INSTRUCTION_AVAILABLE",
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_NOT_WIN",
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_OTHER_WAIT",
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
                snapshot["arb_state_issue_brmsg"] == 1
                and snapshot["arb_state_stall_brmsg"] == 0
            ), "Invalid arb state for issued branch instruction"

        else:
            # verifying not issued branch instructions
            assert (
                stall_reason in allowed_stall_reasons
            ), f"Invalid stall reason for branch instruction: {stall_reason}"

            if (
                stall_reason
                == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_NOT_WIN"
            ):
                assert (
                    snapshot["arb_state_issue_brmsg"] == 1
                    or snapshot["arb_state_stall_brmsg"] == 1
                ), "Arbiter must have issued or stalled brmsg instruction"

            # more specific checks for unconditional branches
            if inst.startswith("s_branch"):
                assert (
                    stall_reason in allowed_stall_reasons_uncoditional_branches
                ), "Invalid stall reason for unconditional branch instruction"


def validate_scalar_instructions(sample_records):
    allowed_stall_reasons = set(
        [
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_NO_INSTRUCTION_AVAILABLE",
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_OTHER_WAIT",
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ALU_DEPENDENCY",
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_NOT_WIN",
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
            ), f"Invalid stall reason for scalar instruction: {stall_reason}"

            if (
                stall_reason
                == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_NOT_WIN"
            ):
                assert (
                    snapshot["arb_state_issue_scalar"] == 1
                    or snapshot["arb_state_stall_scalar"] == 1
                ), "Arbiter must have issued or stalled scalar instruction"


def _validate_barrier_signal(sample_records):
    """Validate s_barrier_signal instructions.

    s_barrier_signal can be issued (type=BARRIER, arb brmsg pipe).
    When stalled, NO_INSTRUCTION_AVAILABLE, ARBITER_NOT_WIN, or OTHER_WAIT are allowed.
    """
    allowed_stall_reasons = set(
        [
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_NO_INSTRUCTION_AVAILABLE",
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_NOT_WIN",
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_OTHER_WAIT",
        ]
    )
    for record in sample_records:
        assert record["inst"].startswith(
            "s_barrier_signal"
        ), "Barrier signal instruction must start with s_barrier_signal"
        snapshot = record["snapshot"]
        if record["wave_issued"] == 1:
            assert (
                record["inst_type"] == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_BARRIER"
            ), "Invalid barrier signal instruction type"
            assert (
                snapshot["arb_state_issue_brmsg"] == 1
            ), "Arbiter must have issued brmsg for barrier signal"
            assert (
                snapshot["arb_state_stall_brmsg"] == 0
            ), "Arbiter must not have stalled brmsg for barrier signal"
        else:
            stall_reason = snapshot["stall_reason"]
            assert (
                stall_reason in allowed_stall_reasons
            ), f"Invalid stall reason for barrier signal: {stall_reason}"

            if (
                stall_reason
                == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_NOT_WIN"
            ):
                assert (
                    snapshot["arb_state_issue_brmsg"] == 1
                    or snapshot["arb_state_stall_brmsg"] == 1
                ), "Arbiter must have issued or stalled brmsg for barrier signal"


def _validate_barrier_wait(sample_records):
    """Validate s_barrier_wait instructions.

    s_barrier_wait is never issued.
    Allowed stall reasons: NO_INSTRUCTION_AVAILABLE, BARRIER_WAIT,
    OTHER_WAIT, INTERNAL_INSTRUCTION.
    """
    allowed_stall_reasons = set(
        [
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_NO_INSTRUCTION_AVAILABLE",
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_BARRIER_WAIT",
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_OTHER_WAIT",
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_INTERNAL_INSTRUCTION",
        ]
    )
    for record in sample_records:
        assert record["inst"].startswith(
            "s_barrier_wait"
        ), "Barrier wait instruction must start with s_barrier_wait"
        assert (
            record["wave_issued"] == 0
        ), "s_barrier_wait should never be issued"
        stall_reason = record["snapshot"]["stall_reason"]
        assert (
            stall_reason in allowed_stall_reasons
        ), f"Invalid stall reason for barrier wait: {stall_reason}"


def validate_barrier_instructions(sample_records):
    """Validate barrier instructions by splitting into signal and wait."""
    signal_records = [r for r in sample_records if r["inst"].startswith("s_barrier_signal")]
    wait_records = [r for r in sample_records if r["inst"].startswith("s_barrier_wait")]

    if signal_records:
        _validate_barrier_signal(signal_records)
    if wait_records:
        _validate_barrier_wait(wait_records)


def validate_jump_instructions(sample_records):
    """Validate jump instructions (s_swappc, s_setpc, s_sleep).

    When issued, inst_type must be JUMP and arbiter must have issued on brmsg pipe.
    When stalled, only NO_INSTRUCTION_AVAILABLE is allowed.
    """
    allowed_stall_reasons = set(
        [
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_NO_INSTRUCTION_AVAILABLE",
        ]
    )
    for record in sample_records:
        snapshot = record["snapshot"]
        if record["wave_issued"] == 1:
            assert (
                record["inst_type"]
                == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_JUMP"
            ), "Invalid jump instruction type"
            assert (
                snapshot["arb_state_issue_brmsg"] == 1
            ), "Arbiter must have issued brmsg instruction for jump"
            assert (
                snapshot["arb_state_stall_brmsg"] == 0
            ), "Arbiter must not have stalled brmsg instruction for jump"
        else:
            stall_reason = snapshot["stall_reason"]
            assert (
                stall_reason in allowed_stall_reasons
            ), f"Invalid stall reason for jump instruction: {stall_reason}"


def validate_message_instructions(sample_records):
    """Validate message instructions (s_sendmsg).

    When issued, inst_type must be MESSAGE and arbiter must have issued on brmsg pipe.
    When stalled, NO_INSTRUCTION_AVAILABLE or ALU_DEPENDENCY are allowed.
    """
    allowed_stall_reasons = set(
        [
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_NO_INSTRUCTION_AVAILABLE",
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ALU_DEPENDENCY",
        ]
    )
    for record in sample_records:
        assert record["inst"].startswith(
            "s_sendmsg"
        ), "Message instruction must start with s_sendmsg"
        snapshot = record["snapshot"]
        if record["wave_issued"] == 1:
            assert (
                record["inst_type"]
                == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_MESSAGE"
            ), "Invalid message instruction type"
            assert (
                snapshot["arb_state_issue_brmsg"] == 1
            ), "Arbiter must have issued brmsg instruction for message"
            assert (
                snapshot["arb_state_stall_brmsg"] == 0
            ), "Arbiter must not have stalled brmsg instruction for message"
        else:
            stall_reason = snapshot["stall_reason"]
            assert (
                stall_reason in allowed_stall_reasons
            ), f"Invalid stall reason for message instruction: {stall_reason}"


def validate_other_s_instructions(sample_records):
    """Validate OTHER type scalar instructions.

    When issued, inst_type must be OTHER.
    When stalled, NO_INSTRUCTION_AVAILABLE or ARBITER_NOT_WIN are allowed.
    """
    allowed_stall_reasons = set(
        [
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_NO_INSTRUCTION_AVAILABLE",
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_NOT_WIN",
        ]
    )
    for record in sample_records:
        snapshot = record["snapshot"]
        if record["wave_issued"] == 1:
            assert (
                record["inst_type"]
                == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_OTHER"
            ), "Invalid OTHER instruction type"
        else:
            stall_reason = snapshot["stall_reason"]
            assert (
                stall_reason in allowed_stall_reasons
            ), f"Invalid stall reason for OTHER instruction: {stall_reason}"


def validate_s_wakeup(sample_records):
    """Validate s_wakeup instructions.

    s_wakeup is declared as NO_INST type but goes through the brmsg arbiter pipe.
    When issued, inst_type must be NO_INST and arbiter must have issued on brmsg pipe.
    When stalled, NO_INSTRUCTION_AVAILABLE or ARBITER_NOT_WIN are allowed.
    """
    allowed_stall_reasons = set(
        [
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_NO_INSTRUCTION_AVAILABLE",
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_NOT_WIN",
        ]
    )
    for record in sample_records:
        assert record["inst"].startswith(
            "s_wakeup"
        ), "S_WAKEUP instruction must start with s_wakeup"
        snapshot = record["snapshot"]
        if record["wave_issued"] == 1:
            assert (
                record["inst_type"]
                == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_NO_INST"
            ), "s_wakeup must have NO_INST instruction type"
            assert (
                snapshot["arb_state_issue_brmsg"] == 1
            ), "Arbiter must have issued brmsg for s_wakeup"
            assert (
                snapshot["arb_state_stall_brmsg"] == 0
            ), "Arbiter must not have stalled brmsg for s_wakeup"
        else:
            stall_reason = snapshot["stall_reason"]
            assert (
                stall_reason in allowed_stall_reasons
            ), f"Invalid stall reason for s_wakeup instruction: {stall_reason}"

            if (
                stall_reason
                == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_NOT_WIN"
            ):
                assert (
                    snapshot["arb_state_issue_brmsg"] == 1
                    or snapshot["arb_state_stall_brmsg"] == 1
                ), "Arbiter must have issued or stalled brmsg for s_wakeup"


def validate_delay_alu_instructions(sample_records):
    """Validate s_delay_alu pseudo-instructions.

    s_delay_alu must never be issued on GFX12. The PC sampling correction WA
    ensures no skid samples for s_delay_alu reach the validator.
    Allowed stall reasons: INTERNAL_INSTRUCTION, OTHER_WAIT, NO_INSTRUCTION_AVAILABLE.
    """
    allowed_stall_reasons = set(
        [
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_INTERNAL_INSTRUCTION",
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_OTHER_WAIT",
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_NO_INSTRUCTION_AVAILABLE",
        ]
    )

    for record in sample_records:
        assert record["inst"].startswith(
            "s_delay_alu"
        ), "DELAY_ALU instruction must start with s_delay_alu"
        assert (
            record["wave_issued"] == 0
        ), f"s_delay_alu must not be issued to EX: {record['inst']}"
        stall_reason = record["snapshot"]["stall_reason"]
        assert (
            stall_reason in allowed_stall_reasons
        ), (
            f"Invalid stall reason for s_delay_alu: {stall_reason}"
        )


def validate_clause_instructions(sample_records):
    """Validate s_clause internal instructions.

    s_clause is never issued.
    Allowed stall reasons: INTERNAL_INSTRUCTION, OTHER_WAIT.
    """
    allowed_stall_reasons = set(
        [
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_INTERNAL_INSTRUCTION",
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_OTHER_WAIT",
        ]
    )

    total = len(sample_records)
    if total == 0:
        return

    for record in sample_records:
        assert record["inst"].startswith(
            "s_clause"
        ), "CLAUSE instruction must start with s_clause"
        assert (
            record["wave_issued"] == 0
        ), "s_clause should never be issued"
        stall_reason = record["snapshot"]["stall_reason"]
        assert (
            stall_reason in allowed_stall_reasons
        ), (
            f"Invalid stall reason for s_clause: {stall_reason}"
        )
