# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

# Canonical method names used to select the exact CSV schema.
HOST_TRAP_METHOD_STR = "host_trap"
STOCHASTIC_METHOD_STR = "stochastic"

HOST_TRAP_COLUMNS = [
    "Sample_Timestamp",
    "Exec_Mask",
    "Dispatch_Id",
    "Instruction",
    "Instruction_Comment",
    "Correlation_Id",
]

STOCHASTIC_COLUMNS = HOST_TRAP_COLUMNS + [
    "Wave_Issued_Instruction",
    "Instruction_Type",
    "Stall_Reason",
    "Wave_Count",
]

_COLUMNS_BY_METHOD = {
    HOST_TRAP_METHOD_STR: HOST_TRAP_COLUMNS,
    STOCHASTIC_METHOD_STR: STOCHASTIC_COLUMNS,
}

MIN_SAMPLES = 100


def validate_columns(df, method):
    # Require the exact schema for the selected sampling method.
    assert method in _COLUMNS_BY_METHOD, f"unsupported PC sampling method: {method}"
    expected = _COLUMNS_BY_METHOD[method]
    assert list(df.columns) == expected, f"unexpected columns: {list(df.columns)}"


def validate_sample_volume(df):
    # Require a non-trivial sample set.
    assert len(df) >= MIN_SAMPLES, f"too few samples: {len(df)}"


def validate_values(df):
    # Undecoded blit/self-modifying-code samples use zero correlation IDs.
    no_instruction = df[(df["Instruction"] == "") | (df["Instruction"] == "nullptr")]
    assert (
        no_instruction["Correlation_Id"] == 0
    ).all(), "undecoded samples must have Correlation_Id == 0"

    cid_zero = df[df["Correlation_Id"] == 0]
    assert (
        (cid_zero["Instruction"] == "") | (cid_zero["Instruction"] == "nullptr")
    ).all(), "Correlation_Id == 0 samples must have no decoded instruction"
    assert len(no_instruction) == len(cid_zero)
    assert (
        cid_zero["Dispatch_Id"] == 0
    ).all(), "uncorrelated samples must have Dispatch_Id == 0"

    assert (df["Correlation_Id"] >= 0).all(), "Correlation_Id must be >= 0"

    # Correlation and dispatch IDs are independent for decoded samples.
    decoded = df[df["Correlation_Id"] != 0]
    assert (decoded["Dispatch_Id"] > 0).all(), "Dispatch_Id must be > 0"
