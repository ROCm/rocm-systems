#!/usr/bin/env python3

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
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import sys
import pytest
import numpy as np
import pandas as pd

# =========================== Validating CSV output (attach mode, stochastic)


def test_validate_pc_sampling_exec_mask_manipulation_csv(
    input_csv: pd.DataFrame, all_sampled: bool
):
    # In attach mode the first captured dispatch surfaces as Correlation_Id == 1
    # and earlier samples carry CID == 0. Drop the CID == 0 prefix and shift
    # CID/Dispatch_Id by popcount(Exec_Mask @ CID==1) - 1 so wrap-mode csv.py
    # helpers apply verbatim.

    from rocprofiler_sdk.pc_sampling.exec_mask_manipulation.csv import (
        validate_exec_mask_based_on_correlation_id,
        validate_instruction_comment,
        validate_instruction_correlation_id_relation,
        validate_instruction_decoding,
    )

    assert not input_csv.empty, "Attach-mode PC sampling CSV is empty"

    df = input_csv.sort_values("Sample_Timestamp").reset_index(drop=True)
    resolved_rows = df[df["Correlation_Id"] != 0]
    assert not resolved_rows.empty, "No resolved samples (all Correlation_Id == 0)"
    first_resolved_idx = int(resolved_rows.index.min())
    df = df.iloc[first_resolved_idx:].copy()
    assert (
        df["Correlation_Id"] != 0
    ).all(), "Found Correlation_Id == 0 samples after the first resolved sample; "

    cid1 = df[df["Correlation_Id"] == 1]
    assert not cid1.empty, (
        "No Correlation_Id == 1 samples found; attach window missed the first "
        "captured dispatch and the offset cannot be inferred."
    )
    # Assumes the first captured dispatch is a loop kernel (popcount(Exec_Mask) == i);
    # a kernel3-first capture would miscompute the offset.
    cid1_popcount = int(
        cid1["Exec_Mask"].apply(lambda exec_mask: bin(exec_mask).count("1")).mode()[0]
    )
    offset = cid1_popcount - 1

    df["Correlation_Id"] = df["Correlation_Id"].astype(int) + offset
    df["Dispatch_Id"] = df["Dispatch_Id"].astype(int) + offset

    # Every dispatch in [min, max] must be represented; a gap means a lost
    # correlation-id mapping during attach.
    min_dispatch_id = int(df["Dispatch_Id"].min())
    max_dispatch_id = int(df["Dispatch_Id"].max())
    unique_dispatch_ids = df["Dispatch_Id"].nunique()
    expected_unique_dispatch_ids = max_dispatch_id - min_dispatch_id + 1
    assert unique_dispatch_ids == expected_unique_dispatch_ids, (
        f"Gap in Dispatch_Id range: expected {expected_unique_dispatch_ids} unique "
        f"dispatches in [{min_dispatch_id}, {max_dispatch_id}], "
        f"got {unique_dispatch_ids}"
    )

    validate_instruction_comment(df)
    validate_instruction_correlation_id_relation(df)

    if max_dispatch_id in (33, 65):
        # kernel3 has v_rcp on even/odd lanes; validate separately.
        first_kernels_df = df[df["Dispatch_Id"] <= max_dispatch_id - 1]
        validate_exec_mask_based_on_correlation_id(first_kernels_df.copy())

        last_kernel = df[df["Dispatch_Id"] == max_dispatch_id]
        exec_mask_size_hex_digits = max_dispatch_id // 4
        even_simd_threads_active_exec_mask = int("5" * exec_mask_size_hex_digits, 16)
        odd_simd_threads_active_exec_mask = int("A" * exec_mask_size_hex_digits, 16)

        validate_instruction_decoding(
            last_kernel,
            "v_rcp_f64",
            exec_mask_uint64=np.uint64(even_simd_threads_active_exec_mask),
            source_code_lines_range=(288, 387),
        )
        validate_instruction_decoding(
            last_kernel,
            "v_rcp_f32",
            exec_mask_uint64=np.uint64(odd_simd_threads_active_exec_mask),
            source_code_lines_range=(391, 490),
        )
    else:
        # Attach window did not reach kernel3; invariant holds on captured kernels.
        validate_exec_mask_based_on_correlation_id(df.copy())


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
