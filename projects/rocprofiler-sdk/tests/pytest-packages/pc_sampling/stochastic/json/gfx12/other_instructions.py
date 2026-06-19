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


def validate_valu_instructions(sample_records):
    allowed_stall_reasons = set(
        [
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_OTHER_WAIT",
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_NO_INSTRUCTION_AVAILABLE",
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_NOT_WIN",
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ALU_DEPENDENCY",
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
            ), f"Invalid stall reason for VALU instruction: {stall_reason}"

            if (
                stall_reason
                == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_NOT_WIN"
            ):
                assert (
                    snapshot["arb_state_issue_valu"] == 1
                    or snapshot["arb_state_stall_valu"] == 1
                ), "VALU pipe must have issued or stalled (at least one must be 1)"


def validate_vmem_instructions(sample_records):
    """Validate vector memory (VMEM/TEX) instructions that use global_ or buffer_ prefix."""
    allowed_stall_reasons = set(
        [
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_NO_INSTRUCTION_AVAILABLE",
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_NOT_WIN",
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_OTHER_WAIT",
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ALU_DEPENDENCY",
        ]
    )
    for record in sample_records:
        assert record["inst"].startswith("global_") or record["inst"].startswith(
            "buffer_"
        ), "TEX/VMEM instruction must start with 'global_' or 'buffer_'"

        snapshot = record["snapshot"]
        if record["wave_issued"] == 1:
            assert (
                record["inst_type"]
                == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_TEX"
            ), "Invalid instruction type for TEX/VMEM instruction"
            assert (
                snapshot["arb_state_issue_vmem_tex"] == 1
            ), "Arbiter must have issued vmem_tex"
            assert (
                snapshot["arb_state_stall_vmem_tex"] == 0
            ), "Arbiter should not have stalled vmem_tex"
        else:
            stall_reason = snapshot["stall_reason"]
            assert (
                stall_reason in allowed_stall_reasons
            ), f"Invalid stall reason for TEX/VMEM instruction: {stall_reason}"

            if (
                stall_reason
                == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_NOT_WIN"
            ):
                assert (
                    snapshot["arb_state_issue_vmem_tex"] == 1
                    or snapshot["arb_state_stall_vmem_tex"] == 1
                ), "VMEM_TEX pipe must have issued or stalled (at least one must be 1)"


def validate_matrix_instructions(sample_records):
    """
    Validate matrix (v_wmma) instructions.

    When issued, inst_type must be MATRIX and arbiter must have issued on valu pipe.
    When stalled, ARBITER_NOT_WIN or NO_INSTRUCTION_AVAILABLE are allowed.
    """
    allowed_stall_reasons = set(
        [
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_NOT_WIN",
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_NO_INSTRUCTION_AVAILABLE",
        ]
    )
    for record in sample_records:
        assert record["inst"].startswith(
            "v_wmma"
        ), "MATRIX instruction must start with v_wmma"

        snapshot = record["snapshot"]
        if record["wave_issued"] == 1:
            assert (
                record["inst_type"]
                == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_MATRIX"
            ), "Invalid instruction type for MATRIX instruction"
            assert (
                snapshot["arb_state_issue_valu"] == 1
            ), "Arbiter must have issued valu for matrix instruction"
            assert (
                snapshot["arb_state_stall_valu"] == 0
            ), "Arbiter should not have stalled valu for matrix instruction"
        else:
            stall_reason = snapshot["stall_reason"]
            assert (
                stall_reason in allowed_stall_reasons
            ), (
                f"Invalid stall reason for MATRIX instruction: {stall_reason}"
            )

            if (
                stall_reason
                == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_NOT_WIN"
            ):
                assert (
                    snapshot["arb_state_issue_valu"] == 1
                    or snapshot["arb_state_stall_valu"] == 1
                ), "Arbiter must have issued or stalled valu for matrix instruction"


def validate_dual_valu_instructions(sample_records):
    """Validate dual VALU (v_dual_*) instructions.

    When issued, inst_type must be DUAL_VALU and arbiter must have dual-issued VALU.
    When stalled, ARBITER_NOT_WIN, NO_INSTRUCTION_AVAILABLE, OTHER_WAIT,
    or ALU_DEPENDENCY are allowed.
    """
    allowed_stall_reasons = set(
        [
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_NOT_WIN",
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_NO_INSTRUCTION_AVAILABLE",
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_OTHER_WAIT",
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ALU_DEPENDENCY",
        ]
    )
    for record in sample_records:
        assert record["inst"].startswith(
            "v_dual_"
        ), "DUAL_VALU instruction must start with v_dual_"

        snapshot = record["snapshot"]
        if record["wave_issued"] == 1:
            assert (
                record["inst_type"]
                == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_DUAL_VALU"
            ), "Invalid instruction type for DUAL_VALU instruction"
            # dual issue VALU implies arb_state_issue_valu == 1 and stall == 0
            assert (
                snapshot["arb_state_issue_valu"] == 1
            ), "Arbiter must have issued VALU for dual VALU"
            assert (
                snapshot["arb_state_stall_valu"] == 0
            ), "Arbiter should not have stalled VALU for dual VALU"
        else:
            stall_reason = snapshot["stall_reason"]
            assert (
                stall_reason in allowed_stall_reasons
            ), (
                f"Invalid stall reason for DUAL_VALU instruction: {stall_reason}"
            )


def validate_flat_instructions(sample_records):
    allowed_stall_reasons = set(
        [
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_OTHER_WAIT",
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_NO_INSTRUCTION_AVAILABLE",
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_NOT_WIN",
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ALU_DEPENDENCY",
        ]
    )
    for record in sample_records:
        assert record["inst"].startswith(
            "flat_"
        ), "FLAT instruction must start with 'flat_'"

        snapshot = record["snapshot"]
        if record["wave_issued"] == 1:
            # wave issued a flat memory instruction
            # flat pipe doesn't exist on GFX12; both lds and vmem_tex pipes are used
            assert (
                record["inst_type"] == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_FLAT"
            ), "Invalid instruction type for FLAT instruction"
            assert snapshot["arb_state_issue_lds"] == 1, "Arbiter must have issued lds"
            assert snapshot["arb_state_stall_lds"] == 0, "Arbiter should not have stalled lds"
            assert snapshot["arb_state_issue_vmem_tex"] == 1, "Arbiter must have issued vmem_tex"
            assert snapshot["arb_state_stall_vmem_tex"] == 0, "Arbiter should not have stalled vmem_tex"
        else:
            stall_reason = snapshot["stall_reason"]
            assert (
                stall_reason in allowed_stall_reasons
            ), f"Invalid stall reason for flat instruction: {stall_reason}"

            if (
                stall_reason
                == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_NOT_WIN"
            ):
                assert (
                    snapshot["arb_state_issue_vmem_tex"] == 1
                    or snapshot["arb_state_stall_vmem_tex"] == 1
                    or snapshot["arb_state_issue_lds"] == 1
                    or snapshot["arb_state_stall_lds"] == 1
                ), "LDS or VMEM_TEX pipe must have issued or stalled for flat ARBITER_NOT_WIN"


def validate_lds_instructions(sample_records):
    allowed_stall_reasons = set(
        [
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_OTHER_WAIT",
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
            ), f"Invalid stall reason for LDS instruction: {stall_reason}"

            if (
                stall_reason
                == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_NOT_WIN"
            ):
                assert (
                    snapshot["arb_state_issue_lds"] == 1
                    or snapshot["arb_state_stall_lds"] == 1
                ), "Arbiter must have issued or stalled LDS"
