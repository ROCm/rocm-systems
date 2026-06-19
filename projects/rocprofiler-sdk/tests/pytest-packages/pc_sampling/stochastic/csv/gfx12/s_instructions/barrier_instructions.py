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


def _split_barrier_samples(barrier_samples):
    """Split barrier samples into s_barrier_signal and s_barrier_wait groups."""
    s_barrier_signal = barrier_samples[
        barrier_samples["Instruction"].str.startswith("s_barrier_signal")
    ]
    s_barrier_wait = barrier_samples[
        barrier_samples["Instruction"].str.startswith("s_barrier_wait")
    ]
    return s_barrier_signal, s_barrier_wait


def validate_barrier_signal(all_samples, s_barrier_signal_samples):
    """Validate s_barrier_signal samples.

    s_barrier_signal can be both issued and stalled:
    - Issued: Instruction_Type must be BARRIER.
    - Stalled: Stall reason must be NO_INSTRUCTION_AVAILABLE, ARBITER_NOT_WIN, or OTHER_WAIT.
    """
    # -- issued --
    issued = s_barrier_signal_samples[
        s_barrier_signal_samples["Wave_Issued_Instruction"]
    ]

    if not issued.empty:
        # Cross-check: issued barrier samples from all_samples must match
        barrier_type_issued = all_samples[
            all_samples["Wave_Issued_Instruction"]
            & (
                all_samples["Instruction_Type"]
                == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_BARRIER"
            )
            & all_samples["Instruction"].str.startswith("s_barrier_signal")
        ]
        assert len(barrier_type_issued) == len(issued), (
            f"Mismatch between barrier-type issued s_barrier_signal samples: "
            f"{len(barrier_type_issued)} vs {len(issued)}"
        )

        assert (
            issued["Instruction_Type"]
            == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_BARRIER"
        ).all(), (
            "Instruction_Type is not BARRIER for all issued s_barrier_signal samples."
        )

    # -- stalled --
    stalled = s_barrier_signal_samples[
        s_barrier_signal_samples["Wave_Issued_Instruction"] == False
    ]

    if not stalled.empty:
        allowed_stall_reasons = {
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_NO_INSTRUCTION_AVAILABLE",
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_NOT_WIN",
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_OTHER_WAIT"
        }
        invalid = ~stalled["Stall_Reason"].isin(allowed_stall_reasons)
        assert not invalid.any(), (
            f"Unexpected stall reasons for s_barrier_signal: "
            f"{set(stalled.loc[invalid, 'Stall_Reason'].unique())}"
        )


def validate_barrier_wait(s_barrier_wait_samples):
    """Validate s_barrier_wait samples.

    s_barrier_wait is always stalled (issued=False).
    Stall reason must be NO_INSTRUCTION_AVAILABLE, BARRIER_WAIT, OTHER_WAIT, or INTERNAL_INSTRUCTION.
    """
    assert (s_barrier_wait_samples["Wave_Issued_Instruction"] == False).all(), (
        "issued is not False for all s_barrier_wait samples."
    )

    if not s_barrier_wait_samples.empty:
        allowed_stall_reasons = {
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_NO_INSTRUCTION_AVAILABLE",
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_BARRIER_WAIT",
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_OTHER_WAIT",
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_INTERNAL_INSTRUCTION"
        }
        invalid = ~s_barrier_wait_samples["Stall_Reason"].isin(allowed_stall_reasons)
        assert not invalid.any(), (
            f"Unexpected stall reasons for s_barrier_wait: "
            f"{set(s_barrier_wait_samples.loc[invalid, 'Stall_Reason'].unique())}"
        )


def validate_barrier_instructions(all_samples, barrier_samples):
    s_barrier_signal, s_barrier_wait = _split_barrier_samples(barrier_samples)

    validate_barrier_signal(all_samples, s_barrier_signal)
    validate_barrier_wait(s_barrier_wait)
