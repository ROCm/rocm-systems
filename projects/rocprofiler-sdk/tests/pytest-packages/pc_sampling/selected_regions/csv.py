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

MIN_SAMPLES = 100


def validate_columns(df, method):
    # CSV columns match the exact schema for this sampling method
    expected = STOCHASTIC_COLUMNS if method == "stochastic" else HOST_TRAP_COLUMNS
    assert list(df.columns) == expected, f"unexpected columns: {list(df.columns)}"


def validate_sample_volume(df):
    # enough samples were collected
    assert len(df) >= MIN_SAMPLES, f"too few samples: {len(df)}"


def validate_values(df):
    # per-row fields are within valid ranges
    assert (df["Exec_Mask"] > 0).all(), "Exec_Mask must be > 0"
    assert (df["Dispatch_Id"] > 0).all(), "Dispatch_Id must be > 0"
    assert (df["Correlation_Id"] >= 0).all(), "Correlation_Id must be >= 0"
