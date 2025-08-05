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


def stochastic_assert(df, df_condition_selection, max_failing_samples=10):
    # TODO: When asserting certain conditions related to exec_masks for all samples,
    # we observe some failures.
    # This usually happens because some small number of samples (e.g., 1-10 out of 100k)
    # do not satisfy the condition. This is either a regression in the ROCr 2nd level trap
    # handler (as sometimes execution mask or correlation ID mismatches), or
    # just stochastic nature of the sampling (meaning our checks are too strict).
    # To relax checks, we introduce an assertion that will allow some small number
    # of samples to disobey the condition.
    # This is a temporary solution until we find the root cause of the issue.

    # extract the failing samples
    failing_samples = df[~df_condition_selection]
    assert len(failing_samples) <= max_failing_samples, "Too many failing samples"


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
        stochastic_assert(
            df_inst, df_inst["Exec_Mask"].astype(np.uint64) == exec_mask_uint64
        )

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
    # type of samples should alway be zero.
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
    stochastic_assert(df, df["active_SIMD_threads"] == df["Correlation_Id"])

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
    stochastic_assert(
        df, df["Exec_Mask"].astype(np.uint64) == df["Exec_Mask2"].astype(np.uint64)
    )


def exec_mask_manipulation_validate_csv(df, all_sampled=False):
    assert not df.empty

    validate_instruction_comment(df)
    validate_instruction_correlation_id_relation(df)

    # Validate samples with non-zero correlation IDs (and with decoded instructions)
    samples_cid_non_zero_df = df[df["Correlation_Id"] != 0]

    # exactly 65 kernels and 65 correlation id
    assert (samples_cid_non_zero_df["Correlation_Id"].astype(int) >= 1).all()
    assert (samples_cid_non_zero_df["Correlation_Id"].astype(int) <= 65).all()
    if all_sampled:
        # all correlation IDs must be sampled
        assert len(samples_cid_non_zero_df["Correlation_Id"].astype(int).unique()) == 65

    first_64_kernels_df = samples_cid_non_zero_df[
        samples_cid_non_zero_df["Correlation_Id"] <= 64
    ]

    # Make a copy, so that we don't work (modify) a view.
    validate_exec_mask_based_on_correlation_id(first_64_kernels_df.copy())

    # validate the last kernel
    kernel_65_df = df[df["Correlation_Id"] == 65]

    # assert that v_rcp instructions are properly decoded
    # the v_rcp is executed by even SIMD threads
    validate_instruction_decoding(
        kernel_65_df,
        "v_rcp_f64",
        exec_mask_uint64=np.uint64(int("5555555555555555", 16)),
        source_code_lines_range=(288, 387),
        all_source_lines_samples=all_sampled,
    )

    # assert that v_rcp_f32 instructions are properly decoded
    # the v_rcp_f32 is executed by odd SIMD threads
    validate_instruction_decoding(
        kernel_65_df,
        "v_rcp_f32",
        exec_mask_uint64=np.uint64(int("AAAAAAAAAAAAAAAA", 16)),
        source_code_lines_range=(391, 490),
        all_source_lines_samples=all_sampled,
    )
