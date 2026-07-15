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


def normalize_attach_csv(df):
    # In attach mode the profiler joins mid-run. Early samples that were not yet mapped
    # to a dispatch carry Correlation_Id == 0 -- drop that prefix. The first mapped
    # dispatch is numbered Correlation_Id == 1 instead of its true index, so shift
    # Correlation_Id and Dispatch_Id by that offset so the shared exec-mask validators apply unchanged.
    assert not df.empty, "attach-mode PC sampling CSV is empty"

    df = df.sort_values("Sample_Timestamp").reset_index(drop=True)
    resolved = df[df["Correlation_Id"] != 0]
    assert not resolved.empty, "no resolved samples (all Correlation_Id == 0)"
    first_resolved_idx = int(resolved.index.min())
    # the dropped prefix must be genuinely unresolved (Dispatch_Id == 0)
    prefix = df.iloc[:first_resolved_idx]
    assert (
        prefix["Dispatch_Id"] == 0
    ).all(), "unresolved prefix has a non-zero Dispatch_Id"
    df = df.iloc[first_resolved_idx:].copy()
    assert (
        df["Correlation_Id"] != 0
    ).all(), "Correlation_Id == 0 samples found after the first resolved sample"

    cid1 = df[df["Correlation_Id"] == 1]
    assert not cid1.empty, (
        "no Correlation_Id == 1 samples; the attach window missed the first captured "
        "dispatch and the offset cannot be inferred"
    )
    # loop kernel i launches i active threads, so popcount(Exec_Mask) == true index
    offset = int(cid1["Exec_Mask"].apply(lambda m: bin(int(m)).count("1")).mode()[0]) - 1
    df["Correlation_Id"] = df["Correlation_Id"].astype(int) + offset
    df["Dispatch_Id"] = df["Dispatch_Id"].astype(int) + offset

    # a gap in the dispatch range means a lost correlation-id mapping during attach
    lo, hi = int(df["Dispatch_Id"].min()), int(df["Dispatch_Id"].max())
    assert df["Dispatch_Id"].nunique() == hi - lo + 1, (
        f"gap in Dispatch_Id range [{lo}, {hi}]: "
        f"{df['Dispatch_Id'].nunique()} unique, expected {hi - lo + 1}"
    )
    return df, hi
