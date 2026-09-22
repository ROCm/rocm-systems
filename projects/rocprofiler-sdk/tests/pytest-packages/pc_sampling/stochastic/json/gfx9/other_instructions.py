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


def validate_valu_instructions(sample_records):
    allowed_stall_reasons = set(
        [
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_WIN_EX_STALL",
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_NO_INSTRUCTION_AVAILABLE",
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_NOT_WIN",
        ]
    )
    for record in sample_records:
        assert record["inst"].startswith("v_"), "VALU instruction must start with 'v_'"

        snapshot = record["snapshot"]
        if record["wave_issued"] == 1:
            # wave issued a VALU instruction
            assert record["inst_type"] == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_VALU"
            assert snapshot["arb_state_issue_valu"] == 1
            assert snapshot["arb_state_stall_valu"] == 0
        else:
            # wave did not issue a VALU instruction
            # inst_type is not relevant
            stall_reason = snapshot["stall_reason"]
            assert (
                stall_reason in allowed_stall_reasons
            ), "Invalid stall reason for VALU instruction"

            if (
                stall_reason
                == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_WIN_EX_STALL"
            ):
                assert snapshot["arb_state_issue_valu"] == 1
                # Expectation would be that the `arb_state_stall_valu` is 1, but in some examples,
                # I've observed different behavior.

            if (
                stall_reason
                == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_NOT_WIN"
            ):
                assert (
                    snapshot["arb_state_issue_valu"] == 1
                    or snapshot["arb_state_stall_matrix"] == 1
                ), "VALU or Matrix instruction should be issued"


def validate_flat_instructions(sample_records):
    allowed_stall_reasons = set(
        [
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_WIN_EX_STALL",
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_NO_INSTRUCTION_AVAILABLE",
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_NOT_WIN",
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ALU_DEPENDENCY",
        ]
    )
    for record in sample_records:
        assert record["inst"].startswith("flat_") or record["inst"].startswith(
            "global_"
        ), "Invalid name of FLAT instruction"

        snapshot = record["snapshot"]
        if record["wave_issued"] == 1:
            # wave issued a flat memory instruction
            assert (
                record["inst_type"] == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_FLAT"
            ), "Invalid instruction type for FLAT instruction"
            assert snapshot["arb_state_issue_flat"] == 1, "Arbiter issued flat"
            assert (
                snapshot["arb_state_stall_flat"] == 0
            ), "Arbiter should not stalled flat"

            # TODO: add checks when flat stalls LDS, and vice versa
            # If global_ inst, check ISSUE_FLAT=1, STALL_FLAT=0, ISSUE_LDS=1 -> STALL_LDS = 1
        else:
            # wave did not issue a flat instruction
            # inst_type is not relevant
            stall_reason = snapshot["stall_reason"]
            assert (
                stall_reason in allowed_stall_reasons
            ), "Invalid stall reason for flat instruction"

            if (
                stall_reason
                == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_WIN_EX_STALL"
            ):
                assert snapshot["arb_state_issue_flat"] == 1, "Arbiter issued flat"
                assert snapshot["arb_state_stall_flat"] == 1, "EX stalled flat"

            # In case of flat instructions, ARBITER_NOT_WIN might mean that
            # the FLAT/VMEM pipe was idle, so the flat instruction is issued to the arbiter
            # to wake up the clock in FLAT/VMEM, but cannot be issued to the execution pipeline.
            # Afterwards, the same instruction is reissued to the arbiter that sends it to the execution pipeline.
            # That's why `Arbiter_State_Issue_Flat` is not always true as in some other cases.


def validate_lds_instructions(sample_records):
    allowed_stall_reasons = set(
        [
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_WIN_EX_STALL",
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_NO_INSTRUCTION_AVAILABLE",
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_NOT_WIN",
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ALU_DEPENDENCY",
        ]
    )
    for record in sample_records:
        assert record["inst"].startswith("ds_"), "Invalid name of LDS instruction"

        snapshot = record["snapshot"]
        if record["wave_issued"] == 1:
            # wave issued an LDS memory instruction
            assert (
                record["inst_type"] == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_LDS"
            ), "Invalid instruction type for LDS instruction"
            assert snapshot["arb_state_issue_lds"] == 1, "Arbiter issued lds"
            assert snapshot["arb_state_stall_lds"] == 0, "EX should not stalled lds"

            # TODO: add checks when LDS stalls flat, and vice versa
            # ISSUE_LDS=1, STALL_LDS=0, ISSUE_FLAT=1 -> STALL_FLAT = 1
        else:
            # wave did not issue an LDS instruction
            # inst_type is not relevant
            stall_reason = snapshot["stall_reason"]
            assert (
                stall_reason in allowed_stall_reasons
            ), "Invalid stall reason for LDS instruction"

            if (
                stall_reason
                == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_WIN_EX_STALL"
            ):
                assert snapshot["arb_state_issue_lds"] == 1, "Arbiter issued flat"
                assert snapshot["arb_state_stall_lds"] == 1, "EX stalled flat"
            elif (
                stall_reason
                == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_NOT_WIN"
            ):
                assert snapshot["arb_state_issue_lds"] == 1, "Arbiter issued flat"
