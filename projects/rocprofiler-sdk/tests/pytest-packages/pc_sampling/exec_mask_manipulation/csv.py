# MIT License
#
# Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

import numpy as np
import pandas as pd


# Keep this in case we decide to revert workgroup_id information
def validate_workgoup_id_x_y_z(df, max_x, max_y, max_z):
    assert (df["Workgroup_Size_X"].astype(int) >= 0).all()
    assert (df["Workgroup_Size_X"].astype(int) <= max_x).all()

    assert (df["Workgroup_Size_Y"].astype(int) >= 0).all()
    assert (df["Workgroup_Size_Y"].astype(int) <= max_y).all()

    assert (df["Workgroup_Size_Z"].astype(int) >= 0).all()
    assert (df["Workgroup_Size_Z"].astype(int) <= max_z).all()


# Keep this in case we decide to revert wave_id information
def validate_wave_id(df, max_wave_id):
    assert (df["Wave_Id"].astype(int) <= max_wave_id).all()


# Keep this in case we decide to revert wave_id information
def validate_chiplet(df, max_chiplet):
    assert (df["Chiplet"].astype(int) <= max_chiplet).all()


def validate_instruction_decoding(
    df,
    inst_str,
    exec_mask_uint64: np.uint64 = None,
    source_code_lines_range: (int, int) = None,
    all_source_lines_samples=False,
):
    # Make a copy, so that we don't work (modify) a view.
    df_inst = df[df["Instruction"].apply(lambda inst: inst.startswith(inst_str))].copy()

    assert not df_inst.empty
    # assert the exec mask if requested
    if exec_mask_uint64 is not None:
        assert (
            df_inst["Exec_Mask"].astype(np.uint64) == exec_mask_uint64
        ).all(), "Exec_Mask mismatch: not all samples have the expected exec mask value"

    # assert whether the samples source code lines belongs to the provided range
    if source_code_lines_range is not None:
        start_range, end_range = source_code_lines_range
        # The instruction comment is isually in the following format: /path/to/source/file.cpp:line_num
        df_inst["source_line_num"] = df_inst["Instruction_Comment"].apply(
            lambda source_line: int(source_line.split(":")[-1])
        )
        assert (df_inst["source_line_num"] >= start_range).all()
        assert (df_inst["source_line_num"] <= end_range).all()
        # if requested, check if all lines from the range are sampled
        if all_source_lines_samples:
            assert len(df_inst["source_line_num"].unique()) == (
                end_range - start_range + 1
            )


def validate_instruction_comment(df):
    # Instruction comment must always be present, since the testing application
    # is built with debug symbols.
    assert (
        (df["Instruction_Comment"] != "") & (df["Instruction_Comment"] != "nullptr")
    ).all()


def validate_instruction_correlation_id_relation(df):
    # Samples with no decoded instructions originates from either
    # blit kernels or self modifying code. The correlation id for this
    # type of samples should always be zero.
    # Thus, Correlation_Id is 0 `iff`` instruction is not decoded.

    # The previous statement has two implications.
    # Implication 1: If the instruction is not decoded, then correlation id is 0.
    samples_no_instruction_df = df[
        (df["Instruction"] == "") | (df["Instruction"] == "nullptr")
    ]
    assert (samples_no_instruction_df["Correlation_Id"] == 0).all()

    # Implication 2: If the correlation id is 0, then the instruction is not decoded.
    samples_cid_zero_df = df[df["Correlation_Id"] == 0]
    assert (
        (samples_cid_zero_df["Instruction"] == "")
        | (samples_cid_zero_df["Instruction"] == "nullptr")
    ).all()

    assert len(samples_no_instruction_df) == len(samples_cid_zero_df)

    # Since we're not enabling any kind of API tracing,
    # internal correlation id should match the dispatch id
    assert all(df["Correlation_Id"] == df["Dispatch_Id"])


def validate_exec_mask_based_on_correlation_id(df):
    # The function assumes that each kernel launches 1024 blocks.
    # Each block contains number of threads that matches correlation ID of the kernel.
    # The exec mask of a sample should contain number of ones equal to
    # the correlation ID of the kernel during which execution the sample was generated.
    df["active_SIMD_threads"] = df["Exec_Mask"].apply(
        lambda exec_mask: bin(exec_mask).count("1")
    )
    assert (
        df["active_SIMD_threads"] == df["Correlation_Id"]
    ).all(), "Active SIMD thread count does not match Correlation_Id for all samples"

    # TODO: Comment out the following code if it causes spurious fails.
    # The more conservative constraint based on the experience follows.
    # The exec mask of sampled instructions of the kernels respect the following pattern:
    # cid -> exec
    # 1 -> 0b1
    # 2 -> 0b11
    # 3 -> 0b111
    # ...
    # 64 -> 0xffffffffffffffff

    df["Exec_Mask2"] = (
        df["Correlation_Id"].astype(int).apply(lambda x: int("0b" + (x * "1"), 2))
    )

    # TODO: exec should be in hex and that will ease the comparison
    assert (
        df["Exec_Mask"].astype(np.uint64) == df["Exec_Mask2"].astype(np.uint64)
    ).all(), "Exec_Mask does not match expected mask derived from Correlation_Id for all samples"


def exec_mask_manipulation_validate_csv(
    df, all_sampled=False, wave_size=None, num_kernels=None
):
    assert not df.empty

    validate_instruction_comment(df)
    validate_instruction_correlation_id_relation(df)

    # Validate samples with non-zero correlation IDs (and with decoded instructions)
    samples_cid_non_zero_df = df[df["Correlation_Id"] != 0]

    # Number of captured kernels (== max correlation id). In wrap mode this is the
    # full run (wave_size + 1 kernels); in attach mode the window may capture fewer.
    max_correlation_id = int(samples_cid_non_zero_df["Correlation_Id"].max())

    if wave_size is None:
        # Backward-compatible path: infer from the data. A full wrap-mode run always
        # has exactly wave_size + 1 kernels, i.e. 33 (wave32) or 65 (wave64).
        assert max_correlation_id in [
            33,
            65,
        ], f"Expected 33 or 65 unique kernels, got {max_correlation_id}"
        wave_size = max_correlation_id - 1

    if num_kernels is None:
        num_kernels = max_correlation_id
    # Never expect more kernels than a full run would produce.
    num_kernels = min(num_kernels, wave_size + 1)

    assert (samples_cid_non_zero_df["Correlation_Id"].astype(int) >= 1).all()
    assert (samples_cid_non_zero_df["Correlation_Id"].astype(int) <= num_kernels).all()
    if all_sampled:
        # all correlation IDs must be sampled
        assert (
            len(samples_cid_non_zero_df["Correlation_Id"].astype(int).unique())
            == num_kernels
        )

    is_full_capture = num_kernels == wave_size + 1

    # The last kernel (kernel3, the v_rcp kernel) is only present on a full capture.
    # On a partial attach capture, validate the exec masks on the captured kernels.
    if not is_full_capture:
        validate_exec_mask_based_on_correlation_id(samples_cid_non_zero_df.copy())
        return

    # all kernels except the last one
    first_kernels_df = samples_cid_non_zero_df[
        samples_cid_non_zero_df["Correlation_Id"] <= num_kernels - 1
    ]

    # Make a copy, so that we don't work (modify) a view.
    validate_exec_mask_based_on_correlation_id(first_kernels_df.copy())

    # validate the last kernel
    last_kernel = df[df["Correlation_Id"] == num_kernels]

    # For 32 wave size, the exec mask is 32 bits or 8 hex digits.
    # For 64 wave size, the exec mask is 64 bits or 16 hex digits.
    exec_mask_size_hex_digits = num_kernels // 4
    even_simd_threads_active_exec_mask = int("5" * exec_mask_size_hex_digits, 16)
    odd_simd_threads_active_exec_mask = int("A" * exec_mask_size_hex_digits, 16)

    # assert that v_rcp instructions are properly decoded
    # the v_rcp is executed by even SIMD threads
    validate_instruction_decoding(
        last_kernel,
        "v_rcp_f64",
        exec_mask_uint64=np.uint64(even_simd_threads_active_exec_mask),
        source_code_lines_range=(301, 400),
        all_source_lines_samples=all_sampled,
    )

    # assert that v_rcp_f32 instructions are properly decoded
    # the v_rcp_f32 is executed by odd SIMD threads
    validate_instruction_decoding(
        last_kernel,
        "v_rcp_f32",
        exec_mask_uint64=np.uint64(odd_simd_threads_active_exec_mask),
        source_code_lines_range=(406, 505),
        all_source_lines_samples=all_sampled,
    )
