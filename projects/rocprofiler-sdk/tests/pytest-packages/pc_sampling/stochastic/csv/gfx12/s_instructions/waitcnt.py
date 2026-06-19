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


def validate_waitcnt(all_samples, waitcnt_samples):
    s_waitcnt_samples = all_samples[
        all_samples["Instruction"].apply(lambda x: x.startswith("s_wait"))
    ]
    # sanity check
    assert len(s_waitcnt_samples) == len(waitcnt_samples)

    # The PC sampling correction WA ensures that no skid samples for s_wait_*
    # instructions reach the validator. s_wait_* instructions are never issued
    # on GFX12, and ARBITER_NOT_WIN skid samples are also corrected away.
    assert (waitcnt_samples["Wave_Issued_Instruction"] == False).all(), (
        "s_wait_* instructions must never be issued on GFX12"
    )

    # accepted stall reasons are
    assert (
        waitcnt_samples["Stall_Reason"]
        .apply(
            lambda x: x == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_WAITCNT"
            or x
            == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_NO_INSTRUCTION_AVAILABLE"
        )
        .all()
    )
