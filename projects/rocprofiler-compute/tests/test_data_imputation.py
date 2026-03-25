##############################################################################
# MIT License
#
# Copyright (c) 2021 - 2026 Advanced Micro Devices, Inc. All Rights Reserved.
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

##############################################################################

import pandas as pd
import pytest

import utils.utils_analysis as utils


def make_multilevel_df(data: dict) -> "pd.DataFrame":
    """
    Create a MultiIndex DataFrame for imputation tests.

    Args:
        data: dict of (level, column) -> values tuples.

    Returns:
        pd.DataFrame with MultiIndex columns.
    """
    df = pd.DataFrame(data)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def test_impute_multiplex_kernel_policy():
    """Test imputation with kernel policy on a single kernel."""

    data = {
        ("file1", "Dispatch_ID"): [1, 2, 3],
        ("file1", "GPU_ID"): [0, 0, 0],
        ("file1", "Grid_Size"): [1024, 512, 1024],
        ("file1", "Workgroup_Size"): [64, 64, 64],
        ("file1", "LDS_Per_Workgroup"): [32, 32, 32],
        ("file1", "Scratch_Per_Workitem"): [0, 0, 0],
        ("file1", "Arch_VGPR"): [16, 16, 16],
        ("file1", "Accum_VGPR"): [0, 0, 0],
        ("file1", "SGPR"): [32, 32, 32],
        ("file1", "Kernel_Name"): ["kernel_a", "kernel_a", "kernel_a"],
        ("file1", "Start_Timestamp"): [1000, 1200, 1400],
        ("file1", "End_Timestamp"): [1500, 1700, 1900],
        ("file1", "Kernel_ID"): [1, 1, 1],
        ("file1", "Counter1"): [100, None, None],
        ("file1", "Counter2"): [None, 500, 300],
    }

    df = make_multilevel_df(data)

    result = utils.impute_counters_iteration_multiplex(df, "kernel")

    # Sort by Dispatch_ID to ensure consistent order
    result = result.sort_values(by=("file1", "Dispatch_ID"))
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3  # Ensure same number of rows

    # Assert Counter2 imputed for first dispatch, Counter1 imputed for second dispatch
    assert result[("file1", "Counter2")].iloc[0] == 500
    assert result[("file1", "Counter1")].iloc[1] == 100


def test_impute_multiplex_kernel_launch_params_policy():
    """Test imputation with kernel_launch_params policy on a single kernel."""

    data = {
        ("file1", "Dispatch_ID"): [1, 2, 3],
        ("file1", "GPU_ID"): [0, 0, 0],
        ("file1", "Grid_Size"): [1024, 512, 1024],
        ("file1", "Workgroup_Size"): [64, 64, 64],
        ("file1", "LDS_Per_Workgroup"): [32, 32, 32],
        ("file1", "Scratch_Per_Workitem"): [0, 0, 0],
        ("file1", "Arch_VGPR"): [16, 16, 16],
        ("file1", "Accum_VGPR"): [0, 0, 0],
        ("file1", "SGPR"): [32, 32, 32],
        ("file1", "Kernel_Name"): ["kernel_a", "kernel_a", "kernel_a"],
        ("file1", "Start_Timestamp"): [1000, 1200, 1400],
        ("file1", "End_Timestamp"): [1500, 1700, 1900],
        ("file1", "Kernel_ID"): [1, 1, 1],
        ("file1", "Counter1"): [100, None, None],
        ("file1", "Counter2"): [None, 500, 300],
    }

    df = make_multilevel_df(data)

    result = utils.impute_counters_iteration_multiplex(df, "kernel_launch_params")

    # Sort by Dispatch_ID to ensure consistent order
    result = result.sort_values(by=("file1", "Dispatch_ID"))

    # Assert Counter2 imputed for first dispatch, Counter1 imputed for last dispatch
    assert result[("file1", "Counter2")].iloc[0] == 300
    assert result[("file1", "Counter1")].iloc[2] == 100

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3


def test_impute_multiplex_kernel_launch_params_no_imputation():
    """Test imputation with kernel_launch_params when no imputation is possible."""

    data = {
        ("file1", "Dispatch_ID"): [1, 2, 3],
        ("file1", "GPU_ID"): [0, 0, 0],
        ("file1", "Grid_Size"): [1024, 1024, 1024],
        ("file1", "Workgroup_Size"): [64, 64, 32],
        ("file1", "LDS_Per_Workgroup"): [32, 24, 32],
        ("file1", "Scratch_Per_Workitem"): [0, 0, 0],
        ("file1", "Arch_VGPR"): [16, 16, 16],
        ("file1", "Accum_VGPR"): [0, 0, 0],
        ("file1", "SGPR"): [32, 32, 32],
        ("file1", "Kernel_Name"): ["kernel_a", "kernel_a", "kernel_a"],
        ("file1", "Start_Timestamp"): [1000, 1200, 1400],
        ("file1", "End_Timestamp"): [1500, 1700, 1900],
        ("file1", "Kernel_ID"): [1, 1, 1],
        ("file1", "Counter1"): [100, None, 300],
        ("file1", "Counter2"): [None, 500, None],
    }

    df = make_multilevel_df(data)

    result = utils.impute_counters_iteration_multiplex(df, "kernel_launch_params")

    # Sort by Dispatch_ID to ensure consistent order
    result = result.sort_values(by=("file1", "Dispatch_ID"))

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3  # Ensure same number of rows

    # No imputation possible
    assert pd.isna(result[("file1", "Counter2")].iloc[0])
    assert pd.isna(result[("file1", "Counter1")].iloc[1])
    assert pd.isna(result[("file1", "Counter2")].iloc[2])


def test_impute_multiplex_multi_kernel_kernel_policy():
    """Test imputation with kernel policy on multiple kernels."""

    data = {
        ("file1", "Dispatch_ID"): [1, 2, 3],
        ("file1", "GPU_ID"): [0, 0, 0],
        ("file1", "Grid_Size"): [1024, 1024, 512],
        ("file1", "Workgroup_Size"): [64, 64, 64],
        ("file1", "LDS_Per_Workgroup"): [32, 32, 32],
        ("file1", "Scratch_Per_Workitem"): [0, 0, 0],
        ("file1", "Arch_VGPR"): [16, 16, 16],
        ("file1", "Accum_VGPR"): [0, 0, 0],
        ("file1", "SGPR"): [32, 32, 32],
        ("file1", "Kernel_Name"): ["kernel_a", "kernel_b", "kernel_a"],
        ("file1", "Start_Timestamp"): [1000, 1200, 1400],
        ("file1", "End_Timestamp"): [1500, 1700, 1900],
        ("file1", "Kernel_ID"): [1, 1, 1],
        ("file1", "Counter1"): [100, None, None],
        ("file1", "Counter2"): [None, 500, 300],
    }

    df = make_multilevel_df(data)

    result = utils.impute_counters_iteration_multiplex(df, "kernel")

    # Sort by Dispatch_ID to ensure consistent order
    result = result.sort_values(by=("file1", "Dispatch_ID"))

    # Assert Counter1 and Counter2 imputed for first and last dispatches
    assert result[("file1", "Counter2")].iloc[0] == 300
    assert result[("file1", "Counter1")].iloc[2] == 100

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3  # Ensure same number of rows


def test_impute_multiplex_multi_kernel_kernel_launch_params_no_imputation():
    """Test imputation with kernel_launch_params when no imputation is possible."""

    data = {
        ("file1", "Dispatch_ID"): [1, 2, 3],
        ("file1", "GPU_ID"): [0, 0, 0],
        ("file1", "Grid_Size"): [1024, 1024, 512],
        ("file1", "Workgroup_Size"): [64, 64, 64],
        ("file1", "LDS_Per_Workgroup"): [32, 32, 32],
        ("file1", "Scratch_Per_Workitem"): [0, 0, 0],
        ("file1", "Arch_VGPR"): [16, 16, 16],
        ("file1", "Accum_VGPR"): [0, 0, 0],
        ("file1", "SGPR"): [32, 32, 32],
        ("file1", "Kernel_Name"): ["kernel_a", "kernel_b", "kernel_a"],
        ("file1", "Start_Timestamp"): [1000, 1200, 1400],
        ("file1", "End_Timestamp"): [1500, 1700, 1900],
        ("file1", "Kernel_ID"): [1, 1, 1],
        ("file1", "Counter1"): [100, None, None],
        ("file1", "Counter2"): [None, 500, 300],
    }

    df = make_multilevel_df(data)

    result = utils.impute_counters_iteration_multiplex(df, "kernel_launch_params")

    # Sort by Dispatch_ID to ensure consistent order
    result = result.sort_values(by=("file1", "Dispatch_ID"))

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3  # Ensure same number of rows

    # No imputation possible
    assert pd.isna(result[("file1", "Counter2")].iloc[0])
    assert pd.isna(result[("file1", "Counter1")].iloc[1])
    assert pd.isna(result[("file1", "Counter1")].iloc[2])


def test_fewer_dispatches_single_kernel():
    """
    Test imputation with kernel policy on a single kernel with
    fewer dispatches than buckets.

    1 kernel, 3 counters, only 2 dispatches (missing C3 bucket).
    C3 remains NaN because there are no previous_fill_values.
    """

    data = {
        ("file1", "Dispatch_ID"): [1, 2],
        ("file1", "GPU_ID"): [0, 0],
        ("file1", "Grid_Size"): [1024, 1024],
        ("file1", "Workgroup_Size"): [64, 64],
        ("file1", "LDS_Per_Workgroup"): [32, 32],
        ("file1", "Scratch_Per_Workitem"): [0, 0],
        ("file1", "Arch_VGPR"): [16, 16],
        ("file1", "Accum_VGPR"): [0, 0],
        ("file1", "SGPR"): [32, 32],
        ("file1", "Kernel_Name"): ["kernel_a", "kernel_a"],
        ("file1", "Start_Timestamp"): [1000, 1200],
        ("file1", "End_Timestamp"): [1500, 1700],
        ("file1", "Kernel_ID"): [1, 1],
        ("file1", "C1"): [10, None],
        ("file1", "C2"): [None, 20],
        ("file1", "C3"): [None, None],
    }

    df = make_multilevel_df(data)
    result = utils.impute_counters_iteration_multiplex(df, "kernel")
    result = result.sort_values(by=("file1", "Dispatch_ID"))

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2

    # C1 and C2 are imputed across both dispatches
    assert result[("file1", "C1")].iloc[0] == 10
    assert result[("file1", "C2")].iloc[0] == 20
    assert result[("file1", "C1")].iloc[1] == 10
    assert result[("file1", "C2")].iloc[1] == 20

    # C3 remains NaN (no dispatch provided a value)
    assert pd.isna(result[("file1", "C3")].iloc[0])
    assert pd.isna(result[("file1", "C3")].iloc[1])


def test_fewer_dispatches_multiple_kernels_both_incomplete():
    """
    Test imputation with kernel policy on multiple kernels, both incomplete.

    kernel_a: buckets {C1}, {C2} (missing C3)
    kernel_b: buckets {C1}, {C2} (missing C3)
    """

    data = {
        ("file1", "Dispatch_ID"): [1, 2, 3, 4],
        ("file1", "GPU_ID"): [0, 0, 0, 0],
        ("file1", "Grid_Size"): [1024, 1024, 1024, 1024],
        ("file1", "Workgroup_Size"): [64, 64, 64, 64],
        ("file1", "LDS_Per_Workgroup"): [32, 32, 32, 32],
        ("file1", "Scratch_Per_Workitem"): [0, 0, 0, 0],
        ("file1", "Arch_VGPR"): [16, 16, 16, 16],
        ("file1", "Accum_VGPR"): [0, 0, 0, 0],
        ("file1", "SGPR"): [32, 32, 32, 32],
        ("file1", "Kernel_Name"): ["kernel_a", "kernel_a", "kernel_b", "kernel_b"],
        ("file1", "Start_Timestamp"): [1000, 1200, 1400, 1600],
        ("file1", "End_Timestamp"): [1500, 1700, 1900, 2100],
        ("file1", "Kernel_ID"): [1, 1, 2, 2],
        ("file1", "C1"): [10, None, 40, None],
        ("file1", "C2"): [None, 20, None, 60],
        ("file1", "C3"): [None, None, None, None],
    }

    df = make_multilevel_df(data)
    result = utils.impute_counters_iteration_multiplex(df, "kernel")
    result = result.sort_values(by=("file1", "Dispatch_ID"))

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 4

    # kernel_a (dispatches 1-2): C1 and C2 imputed, C3 remains NaN
    assert result[("file1", "C1")].iloc[0] == 10
    assert result[("file1", "C2")].iloc[0] == 20
    assert pd.isna(result[("file1", "C3")].iloc[0])
    assert result[("file1", "C1")].iloc[1] == 10
    assert result[("file1", "C2")].iloc[1] == 20
    assert pd.isna(result[("file1", "C3")].iloc[1])

    # kernel_b (dispatches 3-4): C1 and C2 imputed, C3 remains NaN
    assert result[("file1", "C1")].iloc[2] == 40
    assert result[("file1", "C2")].iloc[2] == 60
    assert pd.isna(result[("file1", "C3")].iloc[2])
    assert result[("file1", "C1")].iloc[3] == 40
    assert result[("file1", "C2")].iloc[3] == 60
    assert pd.isna(result[("file1", "C3")].iloc[3])


def test_fewer_dispatches_one_incomplete_one_complete():
    """
    Test imputation with kernel policy on one kernel incomplete, second complete.

    kernel_a: 2 dispatches (missing C3 bucket)
    kernel_b: 3 dispatches covering all 3 buckets
    """

    data = {
        ("file1", "Dispatch_ID"): [1, 2, 3, 4, 5],
        ("file1", "GPU_ID"): [0, 0, 0, 0, 0],
        ("file1", "Grid_Size"): [1024, 1024, 1024, 1024, 1024],
        ("file1", "Workgroup_Size"): [64, 64, 64, 64, 64],
        ("file1", "LDS_Per_Workgroup"): [32, 32, 32, 32, 32],
        ("file1", "Scratch_Per_Workitem"): [0, 0, 0, 0, 0],
        ("file1", "Arch_VGPR"): [16, 16, 16, 16, 16],
        ("file1", "Accum_VGPR"): [0, 0, 0, 0, 0],
        ("file1", "SGPR"): [32, 32, 32, 32, 32],
        ("file1", "Kernel_Name"): [
            "kernel_a",
            "kernel_a",
            "kernel_b",
            "kernel_b",
            "kernel_b",
        ],
        ("file1", "Start_Timestamp"): [1000, 1200, 1400, 1600, 1800],
        ("file1", "End_Timestamp"): [1500, 1700, 1900, 2100, 2300],
        ("file1", "Kernel_ID"): [1, 1, 2, 2, 2],
        ("file1", "C1"): [10, None, 50, None, None],
        ("file1", "C2"): [None, 20, None, 60, None],
        ("file1", "C3"): [None, None, None, None, 70],
    }

    df = make_multilevel_df(data)
    result = utils.impute_counters_iteration_multiplex(df, "kernel")
    result = result.sort_values(by=("file1", "Dispatch_ID"))

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 5

    # kernel_a (dispatches 1-2): C1 and C2 imputed, C3 remains NaN
    assert result[("file1", "C1")].iloc[0] == 10
    assert result[("file1", "C2")].iloc[0] == 20
    assert pd.isna(result[("file1", "C3")].iloc[0])
    assert result[("file1", "C1")].iloc[1] == 10
    assert result[("file1", "C2")].iloc[1] == 20
    assert pd.isna(result[("file1", "C3")].iloc[1])

    # kernel_b (dispatches 3-5): all 3 counters fully imputed
    assert result[("file1", "C1")].iloc[2] == 50
    assert result[("file1", "C2")].iloc[2] == 60
    assert result[("file1", "C3")].iloc[2] == 70
    assert result[("file1", "C1")].iloc[3] == 50
    assert result[("file1", "C2")].iloc[3] == 60
    assert result[("file1", "C3")].iloc[3] == 70
    assert result[("file1", "C1")].iloc[4] == 50
    assert result[("file1", "C2")].iloc[4] == 60
    assert result[("file1", "C3")].iloc[4] == 70


def test_fewer_dispatches_same_kernel_different_launch_params():
    """
    Test imputation with kernel_launch_params on the same kernel
    with different launch params.

    kernel_launch_params policy splits into 2 groups, each incomplete.
    Config 1 (Grid=1024, WG=64, LDS=32): buckets {C1}, {C2}
    Config 2 (Grid=512,  WG=32, LDS=16): buckets {C1}, {C2}
    """

    data = {
        ("file1", "Dispatch_ID"): [1, 2, 3, 4],
        ("file1", "GPU_ID"): [0, 0, 0, 0],
        ("file1", "Grid_Size"): [1024, 1024, 512, 512],
        ("file1", "Workgroup_Size"): [64, 64, 32, 32],
        ("file1", "LDS_Per_Workgroup"): [32, 32, 16, 16],
        ("file1", "Scratch_Per_Workitem"): [0, 0, 0, 0],
        ("file1", "Arch_VGPR"): [16, 16, 16, 16],
        ("file1", "Accum_VGPR"): [0, 0, 0, 0],
        ("file1", "SGPR"): [32, 32, 32, 32],
        ("file1", "Kernel_Name"): [
            "kernel_a",
            "kernel_a",
            "kernel_a",
            "kernel_a",
        ],
        ("file1", "Start_Timestamp"): [1000, 1200, 1400, 1600],
        ("file1", "End_Timestamp"): [1500, 1700, 1900, 2100],
        ("file1", "Kernel_ID"): [1, 1, 1, 1],
        ("file1", "C1"): [10, None, 30, None],
        ("file1", "C2"): [None, 20, None, 40],
        ("file1", "C3"): [None, None, None, None],
    }

    df = make_multilevel_df(data)
    result = utils.impute_counters_iteration_multiplex(df, "kernel_launch_params")
    result = result.sort_values(by=("file1", "Dispatch_ID"))

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 4

    # Config 1 (dispatches 1-2): C1 and C2 imputed, C3 remains NaN
    assert result[("file1", "C1")].iloc[0] == 10
    assert result[("file1", "C2")].iloc[0] == 20
    assert pd.isna(result[("file1", "C3")].iloc[0])
    assert result[("file1", "C1")].iloc[1] == 10
    assert result[("file1", "C2")].iloc[1] == 20
    assert pd.isna(result[("file1", "C3")].iloc[1])

    # Config 2 (dispatches 3-4): C1 and C2 imputed, C3 remains NaN
    assert result[("file1", "C1")].iloc[2] == 30
    assert result[("file1", "C2")].iloc[2] == 40
    assert pd.isna(result[("file1", "C3")].iloc[2])
    assert result[("file1", "C1")].iloc[3] == 30
    assert result[("file1", "C2")].iloc[3] == 40
    assert pd.isna(result[("file1", "C3")].iloc[3])


def test_fewer_dispatches_same_kernel_one_incomplete_one_complete():
    """
    Test imputation with kernel_launch_params on one config incomplete, other complete.

    kernel_launch_params policy:
    Config 1 (Grid=1024, WG=64, LDS=32): 2 dispatches (missing C3 bucket)
    Config 2 (Grid=512,  WG=32, LDS=16): 3 dispatches (all 3 buckets)
    """

    data = {
        ("file1", "Dispatch_ID"): [1, 2, 3, 4, 5],
        ("file1", "GPU_ID"): [0, 0, 0, 0, 0],
        ("file1", "Grid_Size"): [1024, 1024, 512, 512, 512],
        ("file1", "Workgroup_Size"): [64, 64, 32, 32, 32],
        ("file1", "LDS_Per_Workgroup"): [32, 32, 16, 16, 16],
        ("file1", "Scratch_Per_Workitem"): [0, 0, 0, 0, 0],
        ("file1", "Arch_VGPR"): [16, 16, 16, 16, 16],
        ("file1", "Accum_VGPR"): [0, 0, 0, 0, 0],
        ("file1", "SGPR"): [32, 32, 32, 32, 32],
        ("file1", "Kernel_Name"): [
            "kernel_a",
            "kernel_a",
            "kernel_a",
            "kernel_a",
            "kernel_a",
        ],
        ("file1", "Start_Timestamp"): [1000, 1200, 1400, 1600, 1800],
        ("file1", "End_Timestamp"): [1500, 1700, 1900, 2100, 2300],
        ("file1", "Kernel_ID"): [1, 1, 1, 1, 1],
        ("file1", "C1"): [10, None, 50, None, None],
        ("file1", "C2"): [None, 20, None, 60, None],
        ("file1", "C3"): [None, None, None, None, 70],
    }

    df = make_multilevel_df(data)
    result = utils.impute_counters_iteration_multiplex(df, "kernel_launch_params")
    result = result.sort_values(by=("file1", "Dispatch_ID"))

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 5

    # Config 1 (dispatches 1-2): C1 and C2 imputed, C3 remains NaN
    assert result[("file1", "C1")].iloc[0] == 10
    assert result[("file1", "C2")].iloc[0] == 20
    assert pd.isna(result[("file1", "C3")].iloc[0])
    assert result[("file1", "C1")].iloc[1] == 10
    assert result[("file1", "C2")].iloc[1] == 20
    assert pd.isna(result[("file1", "C3")].iloc[1])

    # Config 2 (dispatches 3-5): all 3 counters fully imputed
    assert result[("file1", "C1")].iloc[2] == 50
    assert result[("file1", "C2")].iloc[2] == 60
    assert result[("file1", "C3")].iloc[2] == 70
    assert result[("file1", "C1")].iloc[3] == 50
    assert result[("file1", "C2")].iloc[3] == 60
    assert result[("file1", "C3")].iloc[3] == 70
    assert result[("file1", "C1")].iloc[4] == 50
    assert result[("file1", "C2")].iloc[4] == 60
    assert result[("file1", "C3")].iloc[4] == 70


def test_incomplete_last_group_single_kernel():
    """
    Test imputation with kernel policy on a single kernel with incomplete last group.

    1 kernel, 2 counters, 3 dispatches (2 buckets, 1 full round + 1 trailing).
    The trailing subgroup uses previous_fill_values to fill its gaps.
    """

    data = {
        ("file1", "Dispatch_ID"): [1, 2, 3],
        ("file1", "GPU_ID"): [0, 0, 0],
        ("file1", "Grid_Size"): [1024, 1024, 1024],
        ("file1", "Workgroup_Size"): [64, 64, 64],
        ("file1", "LDS_Per_Workgroup"): [32, 32, 32],
        ("file1", "Scratch_Per_Workitem"): [0, 0, 0],
        ("file1", "Arch_VGPR"): [16, 16, 16],
        ("file1", "Accum_VGPR"): [0, 0, 0],
        ("file1", "SGPR"): [32, 32, 32],
        ("file1", "Kernel_Name"): ["kernel_a", "kernel_a", "kernel_a"],
        ("file1", "Start_Timestamp"): [1000, 1200, 1400],
        ("file1", "End_Timestamp"): [1500, 1700, 1900],
        ("file1", "Kernel_ID"): [1, 1, 1],
        ("file1", "C1"): [10, None, 30],
        ("file1", "C2"): [None, 20, None],
    }

    df = make_multilevel_df(data)
    result = utils.impute_counters_iteration_multiplex(df, "kernel")
    result = result.sort_values(by=("file1", "Dispatch_ID"))

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3

    # Subgroup 1 (dispatches 1-2): C1 and C2 imputed within the subgroup
    assert result[("file1", "C1")].iloc[0] == 10
    assert result[("file1", "C2")].iloc[0] == 20
    assert result[("file1", "C1")].iloc[1] == 10
    assert result[("file1", "C2")].iloc[1] == 20

    # Subgroup 2 (dispatch 3, incomplete): C2 filled from previous_fill_values
    assert result[("file1", "C1")].iloc[2] == 30
    assert result[("file1", "C2")].iloc[2] == 20


def test_incomplete_last_group_multiple_kernels_both_incomplete():
    """
    Test imputation with kernel policy on multiple kernels,
    both with incomplete last groups.

    kernel_a: 4 dispatches, 3 buckets {C1},{C2},{C3} (incomplete last)
    kernel_b: 5 dispatches, 3 buckets {C1},{C2},{C3} (incomplete last)
    """

    data = {
        ("file1", "Dispatch_ID"): [1, 2, 3, 4, 5, 6, 7, 8, 9],
        ("file1", "GPU_ID"): [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ("file1", "Grid_Size"): [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024],
        ("file1", "Workgroup_Size"): [64, 64, 64, 64, 64, 64, 64, 64, 64],
        ("file1", "LDS_Per_Workgroup"): [32, 32, 32, 32, 32, 32, 32, 32, 32],
        ("file1", "Scratch_Per_Workitem"): [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ("file1", "Arch_VGPR"): [16, 16, 16, 16, 16, 16, 16, 16, 16],
        ("file1", "Accum_VGPR"): [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ("file1", "SGPR"): [32, 32, 32, 32, 32, 32, 32, 32, 32],
        ("file1", "Kernel_Name"): [
            "kernel_a",
            "kernel_a",
            "kernel_a",
            "kernel_a",
            "kernel_b",
            "kernel_b",
            "kernel_b",
            "kernel_b",
            "kernel_b",
        ],
        ("file1", "Start_Timestamp"): [
            1000,
            1200,
            1400,
            1600,
            1800,
            2000,
            2200,
            2400,
            2600,
        ],
        ("file1", "End_Timestamp"): [
            1500,
            1700,
            1900,
            2100,
            2300,
            2500,
            2700,
            2900,
            3100,
        ],
        ("file1", "Kernel_ID"): [1, 1, 1, 1, 2, 2, 2, 2, 2],
        ("file1", "C1"): [10, None, None, 40, 50, None, None, 80, None],
        ("file1", "C2"): [None, 20, None, None, None, 60, None, None, 90],
        ("file1", "C3"): [None, None, 30, None, None, None, 70, None, None],
    }

    df = make_multilevel_df(data)
    result = utils.impute_counters_iteration_multiplex(df, "kernel")
    result = result.sort_values(by=("file1", "Dispatch_ID"))

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 9

    # kernel_a subgroup 1 (dispatches 1-3): all 3 counters imputed within subgroup
    assert result[("file1", "C1")].iloc[0] == 10
    assert result[("file1", "C2")].iloc[0] == 20
    assert result[("file1", "C3")].iloc[0] == 30
    assert result[("file1", "C1")].iloc[1] == 10
    assert result[("file1", "C2")].iloc[1] == 20
    assert result[("file1", "C3")].iloc[1] == 30
    assert result[("file1", "C1")].iloc[2] == 10
    assert result[("file1", "C2")].iloc[2] == 20
    assert result[("file1", "C3")].iloc[2] == 30

    # kernel_a subgroup 2 (dispatch 4, incomplete): C2 and C3 from previous_fill_values
    assert result[("file1", "C1")].iloc[3] == 40
    assert result[("file1", "C2")].iloc[3] == 20
    assert result[("file1", "C3")].iloc[3] == 30

    # kernel_b subgroup 1 (dispatches 5-7): all 3 counters imputed within subgroup
    assert result[("file1", "C1")].iloc[4] == 50
    assert result[("file1", "C2")].iloc[4] == 60
    assert result[("file1", "C3")].iloc[4] == 70
    assert result[("file1", "C1")].iloc[5] == 50
    assert result[("file1", "C2")].iloc[5] == 60
    assert result[("file1", "C3")].iloc[5] == 70
    assert result[("file1", "C1")].iloc[6] == 50
    assert result[("file1", "C2")].iloc[6] == 60
    assert result[("file1", "C3")].iloc[6] == 70

    # kernel_b subgroup 2 (dispatches 8-9, incomplete): C3 from previous_fill_values
    assert result[("file1", "C1")].iloc[7] == 80
    assert result[("file1", "C2")].iloc[7] == 90
    assert result[("file1", "C3")].iloc[7] == 70
    assert result[("file1", "C1")].iloc[8] == 80
    assert result[("file1", "C2")].iloc[8] == 90
    assert result[("file1", "C3")].iloc[8] == 70


def test_incomplete_last_group_one_incomplete_other_complete():
    """
    Test imputation with kernel policy on one kernel incomplete, second kernel complete.

    kernel_a: 4 dispatches, 3 buckets {C1},{C2},{C3} (incomplete last)
    kernel_b: 6 dispatches, 3 buckets {C1},{C2},{C3} (2 complete rounds)
    """

    data = {
        ("file1", "Dispatch_ID"): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        ("file1", "GPU_ID"): [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ("file1", "Grid_Size"): [
            1024,
            1024,
            1024,
            1024,
            1024,
            1024,
            1024,
            1024,
            1024,
            1024,
        ],
        ("file1", "Workgroup_Size"): [64, 64, 64, 64, 64, 64, 64, 64, 64, 64],
        ("file1", "LDS_Per_Workgroup"): [32, 32, 32, 32, 32, 32, 32, 32, 32, 32],
        ("file1", "Scratch_Per_Workitem"): [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ("file1", "Arch_VGPR"): [16, 16, 16, 16, 16, 16, 16, 16, 16, 16],
        ("file1", "Accum_VGPR"): [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ("file1", "SGPR"): [32, 32, 32, 32, 32, 32, 32, 32, 32, 32],
        ("file1", "Kernel_Name"): [
            "kernel_a",
            "kernel_a",
            "kernel_a",
            "kernel_a",
            "kernel_b",
            "kernel_b",
            "kernel_b",
            "kernel_b",
            "kernel_b",
            "kernel_b",
        ],
        ("file1", "Start_Timestamp"): [
            1000,
            1200,
            1400,
            1600,
            1800,
            2000,
            2200,
            2400,
            2600,
            2800,
        ],
        ("file1", "End_Timestamp"): [
            1500,
            1700,
            1900,
            2100,
            2300,
            2500,
            2700,
            2900,
            3100,
            3300,
        ],
        ("file1", "Kernel_ID"): [1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
        ("file1", "C1"): [10, None, None, 40, 50, None, None, 80, None, None],
        ("file1", "C2"): [None, 20, None, None, None, 60, None, None, 90, None],
        ("file1", "C3"): [None, None, 30, None, None, None, 70, None, None, 100],
    }

    df = make_multilevel_df(data)
    result = utils.impute_counters_iteration_multiplex(df, "kernel")
    result = result.sort_values(by=("file1", "Dispatch_ID"))

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 10

    # kernel_a subgroup 1 (dispatches 1-3): all 3 counters imputed within subgroup
    assert result[("file1", "C1")].iloc[0] == 10
    assert result[("file1", "C2")].iloc[0] == 20
    assert result[("file1", "C3")].iloc[0] == 30
    assert result[("file1", "C1")].iloc[1] == 10
    assert result[("file1", "C2")].iloc[1] == 20
    assert result[("file1", "C3")].iloc[1] == 30
    assert result[("file1", "C1")].iloc[2] == 10
    assert result[("file1", "C2")].iloc[2] == 20
    assert result[("file1", "C3")].iloc[2] == 30

    # kernel_a subgroup 2 (dispatch 4, incomplete): C2 and C3 from previous_fill_values
    assert result[("file1", "C1")].iloc[3] == 40
    assert result[("file1", "C2")].iloc[3] == 20
    assert result[("file1", "C3")].iloc[3] == 30

    # kernel_b subgroup 1 (dispatches 5-7): all 3 counters imputed within subgroup
    assert result[("file1", "C1")].iloc[4] == 50
    assert result[("file1", "C2")].iloc[4] == 60
    assert result[("file1", "C3")].iloc[4] == 70
    assert result[("file1", "C1")].iloc[5] == 50
    assert result[("file1", "C2")].iloc[5] == 60
    assert result[("file1", "C3")].iloc[5] == 70
    assert result[("file1", "C1")].iloc[6] == 50
    assert result[("file1", "C2")].iloc[6] == 60
    assert result[("file1", "C3")].iloc[6] == 70

    # kernel_b subgroup 2 (dispatches 8-10): complete round, no fallback needed
    assert result[("file1", "C1")].iloc[7] == 80
    assert result[("file1", "C2")].iloc[7] == 90
    assert result[("file1", "C3")].iloc[7] == 100
    assert result[("file1", "C1")].iloc[8] == 80
    assert result[("file1", "C2")].iloc[8] == 90
    assert result[("file1", "C3")].iloc[8] == 100
    assert result[("file1", "C1")].iloc[9] == 80
    assert result[("file1", "C2")].iloc[9] == 90
    assert result[("file1", "C3")].iloc[9] == 100


def test_incomplete_last_group_same_kernel_different_launch_params():
    """
    Test imputation with kernel_launch_params on the same kernel
    with different launch params.

    kernel_launch_params policy, both configs have incomplete last subgroups.
    Config 1 (Grid=1024, WG=64, LDS=32): 3 dispatches, 2 buckets
    Config 2 (Grid=512,  WG=32, LDS=16): 3 dispatches, 2 buckets
    """

    data = {
        ("file1", "Dispatch_ID"): [1, 2, 3, 4, 5, 6],
        ("file1", "GPU_ID"): [0, 0, 0, 0, 0, 0],
        ("file1", "Grid_Size"): [1024, 1024, 1024, 512, 512, 512],
        ("file1", "Workgroup_Size"): [64, 64, 64, 32, 32, 32],
        ("file1", "LDS_Per_Workgroup"): [32, 32, 32, 16, 16, 16],
        ("file1", "Scratch_Per_Workitem"): [0, 0, 0, 0, 0, 0],
        ("file1", "Arch_VGPR"): [16, 16, 16, 16, 16, 16],
        ("file1", "Accum_VGPR"): [0, 0, 0, 0, 0, 0],
        ("file1", "SGPR"): [32, 32, 32, 32, 32, 32],
        ("file1", "Kernel_Name"): [
            "kernel_a",
            "kernel_a",
            "kernel_a",
            "kernel_a",
            "kernel_a",
            "kernel_a",
        ],
        ("file1", "Start_Timestamp"): [1000, 1200, 1400, 1600, 1800, 2000],
        ("file1", "End_Timestamp"): [1500, 1700, 1900, 2100, 2300, 2500],
        ("file1", "Kernel_ID"): [1, 1, 1, 1, 1, 1],
        ("file1", "C1"): [10, None, 30, 50, None, 70],
        ("file1", "C2"): [None, 20, None, None, 60, None],
    }

    df = make_multilevel_df(data)
    result = utils.impute_counters_iteration_multiplex(df, "kernel_launch_params")
    result = result.sort_values(by=("file1", "Dispatch_ID"))

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 6

    # Config 1 (dispatches 1-3): incomplete last, C2 from previous_fill_values
    assert result[("file1", "C1")].iloc[0] == 10
    assert result[("file1", "C2")].iloc[0] == 20
    assert result[("file1", "C1")].iloc[1] == 10
    assert result[("file1", "C2")].iloc[1] == 20
    assert result[("file1", "C1")].iloc[2] == 30
    assert result[("file1", "C2")].iloc[2] == 20

    # Config 2 (dispatches 4-6): incomplete last, C1 from previous_fill_values
    assert result[("file1", "C1")].iloc[3] == 50
    assert result[("file1", "C2")].iloc[3] == 60
    assert result[("file1", "C1")].iloc[4] == 50
    assert result[("file1", "C2")].iloc[4] == 60
    assert result[("file1", "C1")].iloc[5] == 70
    assert result[("file1", "C2")].iloc[5] == 60


def test_incomplete_last_group_same_kernel_one_incomplete_one_complete():
    """
    Test imputation with kernel_launch_params on the same kernel
    with one config incomplete, other complete.

    kernel_launch_params policy:
    Config 1 (Grid=1024, WG=64, LDS=32): 3 dispatches, incomplete last
    Config 2 (Grid=512,  WG=32, LDS=16): 4 dispatches, 2 complete rounds
    """

    data = {
        ("file1", "Dispatch_ID"): [1, 2, 3, 4, 5, 6, 7],
        ("file1", "GPU_ID"): [0, 0, 0, 0, 0, 0, 0],
        ("file1", "Grid_Size"): [1024, 1024, 1024, 512, 512, 512, 512],
        ("file1", "Workgroup_Size"): [64, 64, 64, 32, 32, 32, 32],
        ("file1", "LDS_Per_Workgroup"): [32, 32, 32, 16, 16, 16, 16],
        ("file1", "Scratch_Per_Workitem"): [0, 0, 0, 0, 0, 0, 0],
        ("file1", "Arch_VGPR"): [16, 16, 16, 16, 16, 16, 16],
        ("file1", "Accum_VGPR"): [0, 0, 0, 0, 0, 0, 0],
        ("file1", "SGPR"): [32, 32, 32, 32, 32, 32, 32],
        ("file1", "Kernel_Name"): [
            "kernel_a",
            "kernel_a",
            "kernel_a",
            "kernel_a",
            "kernel_a",
            "kernel_a",
            "kernel_a",
        ],
        ("file1", "Start_Timestamp"): [1000, 1200, 1400, 1600, 1800, 2000, 2200],
        ("file1", "End_Timestamp"): [1500, 1700, 1900, 2100, 2300, 2500, 2700],
        ("file1", "Kernel_ID"): [1, 1, 1, 1, 1, 1, 1],
        ("file1", "C1"): [10, None, 30, 50, None, 70, None],
        ("file1", "C2"): [None, 20, None, None, 60, None, 80],
    }

    df = make_multilevel_df(data)
    result = utils.impute_counters_iteration_multiplex(df, "kernel_launch_params")
    result = result.sort_values(by=("file1", "Dispatch_ID"))

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 7

    # Config 1 (dispatches 1-3): incomplete last, C2 from previous_fill_values
    assert result[("file1", "C1")].iloc[0] == 10
    assert result[("file1", "C2")].iloc[0] == 20
    assert result[("file1", "C1")].iloc[1] == 10
    assert result[("file1", "C2")].iloc[1] == 20
    assert result[("file1", "C1")].iloc[2] == 30
    assert result[("file1", "C2")].iloc[2] == 20

    # Config 2 (dispatches 4-7): 2 complete rounds, no fallback needed
    assert result[("file1", "C1")].iloc[3] == 50
    assert result[("file1", "C2")].iloc[3] == 60
    assert result[("file1", "C1")].iloc[4] == 50
    assert result[("file1", "C2")].iloc[4] == 60
    assert result[("file1", "C1")].iloc[5] == 70
    assert result[("file1", "C2")].iloc[5] == 80
    assert result[("file1", "C1")].iloc[6] == 70
    assert result[("file1", "C2")].iloc[6] == 80


def test_complete_last_group_single_kernel():
    """
    Test imputation with kernel policy on a single kernel with complete last group.

    1 kernel, 2 counters, 4 dispatches (2 complete rounds of 2 buckets).
    All imputation happens within subgroups; previous_fill_values fallback
    is never needed.
    """

    data = {
        ("file1", "Dispatch_ID"): [1, 2, 3, 4],
        ("file1", "GPU_ID"): [0, 0, 0, 0],
        ("file1", "Grid_Size"): [1024, 1024, 1024, 1024],
        ("file1", "Workgroup_Size"): [64, 64, 64, 64],
        ("file1", "LDS_Per_Workgroup"): [32, 32, 32, 32],
        ("file1", "Scratch_Per_Workitem"): [0, 0, 0, 0],
        ("file1", "Arch_VGPR"): [16, 16, 16, 16],
        ("file1", "Accum_VGPR"): [0, 0, 0, 0],
        ("file1", "SGPR"): [32, 32, 32, 32],
        ("file1", "Kernel_Name"): ["kernel_a", "kernel_a", "kernel_a", "kernel_a"],
        ("file1", "Start_Timestamp"): [1000, 1200, 1400, 1600],
        ("file1", "End_Timestamp"): [1500, 1700, 1900, 2100],
        ("file1", "Kernel_ID"): [1, 1, 1, 1],
        ("file1", "C1"): [10, None, 30, None],
        ("file1", "C2"): [None, 20, None, 40],
    }

    df = make_multilevel_df(data)
    result = utils.impute_counters_iteration_multiplex(df, "kernel")
    result = result.sort_values(by=("file1", "Dispatch_ID"))

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 4

    # Subgroup 1 (dispatches 1-2): self-contained imputation
    assert result[("file1", "C1")].iloc[0] == 10
    assert result[("file1", "C2")].iloc[0] == 20
    assert result[("file1", "C1")].iloc[1] == 10
    assert result[("file1", "C2")].iloc[1] == 20

    # Subgroup 2 (dispatches 3-4): self-contained, values don't bleed from subgroup 1
    assert result[("file1", "C1")].iloc[2] == 30
    assert result[("file1", "C2")].iloc[2] == 40
    assert result[("file1", "C1")].iloc[3] == 30
    assert result[("file1", "C2")].iloc[3] == 40


def test_complete_last_group_multiple_kernels_both_complete():
    """
    Test imputation with kernel policy on multiple kernels, both complete.

    kernel_a: 6 dispatches, 3 buckets {C1},{C2},{C3}, 2 complete rounds
    kernel_b: 6 dispatches, 3 buckets {C1},{C2},{C3}, 2 complete rounds
    """

    data = {
        ("file1", "Dispatch_ID"): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        ("file1", "GPU_ID"): [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ("file1", "Grid_Size"): [
            1024,
            1024,
            1024,
            1024,
            1024,
            1024,
            1024,
            1024,
            1024,
            1024,
            1024,
            1024,
        ],
        ("file1", "Workgroup_Size"): [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64],
        ("file1", "LDS_Per_Workgroup"): [
            32,
            32,
            32,
            32,
            32,
            32,
            32,
            32,
            32,
            32,
            32,
            32,
        ],
        ("file1", "Scratch_Per_Workitem"): [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ("file1", "Arch_VGPR"): [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16],
        ("file1", "Accum_VGPR"): [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ("file1", "SGPR"): [32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32],
        ("file1", "Kernel_Name"): [
            "kernel_a",
            "kernel_a",
            "kernel_a",
            "kernel_a",
            "kernel_a",
            "kernel_a",
            "kernel_b",
            "kernel_b",
            "kernel_b",
            "kernel_b",
            "kernel_b",
            "kernel_b",
        ],
        ("file1", "Start_Timestamp"): [
            1000,
            1200,
            1400,
            1600,
            1800,
            2000,
            2200,
            2400,
            2600,
            2800,
            3000,
            3200,
        ],
        ("file1", "End_Timestamp"): [
            1500,
            1700,
            1900,
            2100,
            2300,
            2500,
            2700,
            2900,
            3100,
            3300,
            3500,
            3700,
        ],
        ("file1", "Kernel_ID"): [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
        ("file1", "C1"): [
            10,
            None,
            None,
            40,
            None,
            None,
            70,
            None,
            None,
            100,
            None,
            None,
        ],
        ("file1", "C2"): [
            None,
            20,
            None,
            None,
            50,
            None,
            None,
            80,
            None,
            None,
            110,
            None,
        ],
        ("file1", "C3"): [
            None,
            None,
            30,
            None,
            None,
            60,
            None,
            None,
            90,
            None,
            None,
            120,
        ],
    }

    df = make_multilevel_df(data)
    result = utils.impute_counters_iteration_multiplex(df, "kernel")
    result = result.sort_values(by=("file1", "Dispatch_ID"))

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 12

    # kernel_a round 1 (dispatches 1-3)
    assert result[("file1", "C1")].iloc[0] == 10
    assert result[("file1", "C2")].iloc[0] == 20
    assert result[("file1", "C3")].iloc[0] == 30
    assert result[("file1", "C1")].iloc[1] == 10
    assert result[("file1", "C2")].iloc[1] == 20
    assert result[("file1", "C3")].iloc[1] == 30
    assert result[("file1", "C1")].iloc[2] == 10
    assert result[("file1", "C2")].iloc[2] == 20
    assert result[("file1", "C3")].iloc[2] == 30

    # kernel_a round 2 (dispatches 4-6)
    assert result[("file1", "C1")].iloc[3] == 40
    assert result[("file1", "C2")].iloc[3] == 50
    assert result[("file1", "C3")].iloc[3] == 60
    assert result[("file1", "C1")].iloc[4] == 40
    assert result[("file1", "C2")].iloc[4] == 50
    assert result[("file1", "C3")].iloc[4] == 60
    assert result[("file1", "C1")].iloc[5] == 40
    assert result[("file1", "C2")].iloc[5] == 50
    assert result[("file1", "C3")].iloc[5] == 60

    # kernel_b round 1 (dispatches 7-9)
    assert result[("file1", "C1")].iloc[6] == 70
    assert result[("file1", "C2")].iloc[6] == 80
    assert result[("file1", "C3")].iloc[6] == 90
    assert result[("file1", "C1")].iloc[7] == 70
    assert result[("file1", "C2")].iloc[7] == 80
    assert result[("file1", "C3")].iloc[7] == 90
    assert result[("file1", "C1")].iloc[8] == 70
    assert result[("file1", "C2")].iloc[8] == 80
    assert result[("file1", "C3")].iloc[8] == 90

    # kernel_b round 2 (dispatches 10-12)
    assert result[("file1", "C1")].iloc[9] == 100
    assert result[("file1", "C2")].iloc[9] == 110
    assert result[("file1", "C3")].iloc[9] == 120
    assert result[("file1", "C1")].iloc[10] == 100
    assert result[("file1", "C2")].iloc[10] == 110
    assert result[("file1", "C3")].iloc[10] == 120
    assert result[("file1", "C1")].iloc[11] == 100
    assert result[("file1", "C2")].iloc[11] == 110
    assert result[("file1", "C3")].iloc[11] == 120


def test_complete_last_group_same_kernel_different_launch_params():
    """
    Test imputation with kernel_launch_params on the same kernel
    with different launch params.

    kernel_launch_params policy, both configs have 2 complete rounds.
    Config 1 (Grid=1024, WG=64, LDS=32): 4 dispatches
    Config 2 (Grid=512,  WG=32, LDS=16): 4 dispatches
    """

    data = {
        ("file1", "Dispatch_ID"): [1, 2, 3, 4, 5, 6, 7, 8],
        ("file1", "GPU_ID"): [0, 0, 0, 0, 0, 0, 0, 0],
        ("file1", "Grid_Size"): [1024, 1024, 1024, 1024, 512, 512, 512, 512],
        ("file1", "Workgroup_Size"): [64, 64, 64, 64, 32, 32, 32, 32],
        ("file1", "LDS_Per_Workgroup"): [32, 32, 32, 32, 16, 16, 16, 16],
        ("file1", "Scratch_Per_Workitem"): [0, 0, 0, 0, 0, 0, 0, 0],
        ("file1", "Arch_VGPR"): [16, 16, 16, 16, 16, 16, 16, 16],
        ("file1", "Accum_VGPR"): [0, 0, 0, 0, 0, 0, 0, 0],
        ("file1", "SGPR"): [32, 32, 32, 32, 32, 32, 32, 32],
        ("file1", "Kernel_Name"): [
            "kernel_a",
            "kernel_a",
            "kernel_a",
            "kernel_a",
            "kernel_a",
            "kernel_a",
            "kernel_a",
            "kernel_a",
        ],
        ("file1", "Start_Timestamp"): [1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400],
        ("file1", "End_Timestamp"): [1500, 1700, 1900, 2100, 2300, 2500, 2700, 2900],
        ("file1", "Kernel_ID"): [1, 1, 1, 1, 1, 1, 1, 1],
        ("file1", "C1"): [10, None, 30, None, 50, None, 70, None],
        ("file1", "C2"): [None, 20, None, 40, None, 60, None, 80],
    }

    df = make_multilevel_df(data)
    result = utils.impute_counters_iteration_multiplex(df, "kernel_launch_params")
    result = result.sort_values(by=("file1", "Dispatch_ID"))

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 8

    # Config 1 round 1 (dispatches 1-2)
    assert result[("file1", "C1")].iloc[0] == 10
    assert result[("file1", "C2")].iloc[0] == 20
    assert result[("file1", "C1")].iloc[1] == 10
    assert result[("file1", "C2")].iloc[1] == 20

    # Config 1 round 2 (dispatches 3-4)
    assert result[("file1", "C1")].iloc[2] == 30
    assert result[("file1", "C2")].iloc[2] == 40
    assert result[("file1", "C1")].iloc[3] == 30
    assert result[("file1", "C2")].iloc[3] == 40

    # Config 2 round 1 (dispatches 5-6)
    assert result[("file1", "C1")].iloc[4] == 50
    assert result[("file1", "C2")].iloc[4] == 60
    assert result[("file1", "C1")].iloc[5] == 50
    assert result[("file1", "C2")].iloc[5] == 60

    # Config 2 round 2 (dispatches 7-8)
    assert result[("file1", "C1")].iloc[6] == 70
    assert result[("file1", "C2")].iloc[6] == 80
    assert result[("file1", "C1")].iloc[7] == 70
    assert result[("file1", "C2")].iloc[7] == 80


def test_impute_counters_iteration_multiplex_incorrect_structure():
    """Test imputation when the DataFrame is not a MultiIndex."""

    flat_df = pd.DataFrame({
        "Dispatch_ID": [1],
        "Kernel_Name": ["kernel_a"],
        "C1": [10],
    })
    with pytest.raises(ValueError, match="multi-index"):
        utils.impute_counters_iteration_multiplex(flat_df, "kernel")


def test_impute_counters_iteration_multiplex_single_level_multiindex():
    """Test imputation when the DataFrame is a Single-level MultiIndex."""

    single_level_df = pd.DataFrame({
        "Dispatch_ID": [1],
        "Kernel_Name": ["kernel_a"],
        "C1": [10],
    })
    single_level_df.columns = pd.MultiIndex.from_arrays([
        ["Dispatch_ID", "Kernel_Name", "C1"]
    ])
    with pytest.raises(ValueError, match="multi-index"):
        utils.impute_counters_iteration_multiplex(single_level_df, "kernel")


def test_impute_counters_iteration_multiplex_missing_kernel_name():
    """
    Test imputation when the DataFrame is a valid 2-level MultiIndex
    but without the Kernel_Name column raises a KeyError.
    """

    data_no_kernel_name = {
        ("file1", "Dispatch_ID"): [1, 2],
        ("file1", "GPU_ID"): [0, 0],
        ("file1", "Grid_Size"): [1024, 1024],
        ("file1", "Workgroup_Size"): [64, 64],
        ("file1", "LDS_Per_Workgroup"): [32, 32],
        ("file1", "Scratch_Per_Workitem"): [0, 0],
        ("file1", "Arch_VGPR"): [16, 16],
        ("file1", "Accum_VGPR"): [0, 0],
        ("file1", "SGPR"): [32, 32],
        ("file1", "Start_Timestamp"): [1000, 1200],
        ("file1", "End_Timestamp"): [1500, 1700],
        ("file1", "Kernel_ID"): [1, 1],
        ("file1", "C1"): [10, None],
        ("file1", "C2"): [None, 20],
    }
    df_no_kn = make_multilevel_df(data_no_kernel_name)
    with pytest.raises(KeyError):
        utils.impute_counters_iteration_multiplex(df_no_kn, "kernel")


def test_impute_counters_iteration_multiplex_empty_dataframe():
    """Test imputation when the DataFrame is a valid MultiIndex but has no data rows."""

    data_empty = {
        ("file1", "Dispatch_ID"): [],
        ("file1", "GPU_ID"): [],
        ("file1", "Grid_Size"): [],
        ("file1", "Workgroup_Size"): [],
        ("file1", "LDS_Per_Workgroup"): [],
        ("file1", "Scratch_Per_Workitem"): [],
        ("file1", "Arch_VGPR"): [],
        ("file1", "Accum_VGPR"): [],
        ("file1", "SGPR"): [],
        ("file1", "Kernel_Name"): [],
        ("file1", "Start_Timestamp"): [],
        ("file1", "End_Timestamp"): [],
        ("file1", "Kernel_ID"): [],
        ("file1", "C1"): [],
        ("file1", "C2"): [],
    }
    df_empty = make_multilevel_df(data_empty)
    result = utils.impute_counters_iteration_multiplex(df_empty, "kernel")

    # Result is a fallback DataFrame, not actual data.
    # It has a single column ("file1", 0) containing 15 column name strings.
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == [("file1", 0)]
    assert len(result) == 15
    assert "Dispatch_ID" in result[("file1", 0)].values
    assert "C1" in result[("file1", 0)].values


def test_impute_counters_iteration_multiplex_all_counters_nan():
    """
    Test imputation when all counter values are NaN.

    The bucket-identification loop finds no non-empty frozensets, so
    counter_groups stays empty and the group is skipped entirely.
    """

    data_all_nan = {
        ("file1", "Dispatch_ID"): [1, 2, 3],
        ("file1", "GPU_ID"): [0, 0, 0],
        ("file1", "Grid_Size"): [1024, 1024, 1024],
        ("file1", "Workgroup_Size"): [64, 64, 64],
        ("file1", "LDS_Per_Workgroup"): [32, 32, 32],
        ("file1", "Scratch_Per_Workitem"): [0, 0, 0],
        ("file1", "Arch_VGPR"): [16, 16, 16],
        ("file1", "Accum_VGPR"): [0, 0, 0],
        ("file1", "SGPR"): [32, 32, 32],
        ("file1", "Kernel_Name"): ["kernel_a", "kernel_a", "kernel_a"],
        ("file1", "Start_Timestamp"): [1000, 1200, 1400],
        ("file1", "End_Timestamp"): [1500, 1700, 1900],
        ("file1", "Kernel_ID"): [1, 1, 1],
        ("file1", "C1"): [None, None, None],
        ("file1", "C2"): [None, None, None],
    }
    df_all_nan = make_multilevel_df(data_all_nan)
    result = utils.impute_counters_iteration_multiplex(df_all_nan, "kernel")

    # Group was dropped (no valid counters) -- fallback DataFrame returned.
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == [("file1", 0)]
    assert len(result) == 15

    # Original dispatch data is absent from the result
    assert 1 not in result[("file1", 0)].values
    assert 2 not in result[("file1", 0)].values
    assert 3 not in result[("file1", 0)].values


def test_impute_counters_iteration_multiplex_no_counter_columns():
    """
    Test imputation when the DataFrame contains only the 13 non-counter columns.

    counter_columns is empty, so every row yields an empty frozenset
    and the group is skipped.
    """

    data_no_counters = {
        ("file1", "Dispatch_ID"): [1, 2],
        ("file1", "GPU_ID"): [0, 0],
        ("file1", "Grid_Size"): [1024, 1024],
        ("file1", "Workgroup_Size"): [64, 64],
        ("file1", "LDS_Per_Workgroup"): [32, 32],
        ("file1", "Scratch_Per_Workitem"): [0, 0],
        ("file1", "Arch_VGPR"): [16, 16],
        ("file1", "Accum_VGPR"): [0, 0],
        ("file1", "SGPR"): [32, 32],
        ("file1", "Kernel_Name"): ["kernel_a", "kernel_a"],
        ("file1", "Start_Timestamp"): [1000, 1200],
        ("file1", "End_Timestamp"): [1500, 1700],
        ("file1", "Kernel_ID"): [1, 1],
    }
    df_no_counters = make_multilevel_df(data_no_counters)
    result = utils.impute_counters_iteration_multiplex(df_no_counters, "kernel")

    # Group was dropped (no counter columns exist) -- fallback DataFrame returned.
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == [("file1", 0)]
    assert len(result) == 13

    # No counter column names in the fallback values
    assert "C1" not in result[("file1", 0)].values
    assert "C2" not in result[("file1", 0)].values


def test_impute_counters_iteration_multiplex_unrecognized_policy():
    """
    Test imputation when the policy is unrecognized.
    Any policy other than "kernel" falls through to the else branch
    (same as "kernel_launch_params"). The output must match exactly.
    """

    data_policy = {
        ("file1", "Dispatch_ID"): [1, 2, 3],
        ("file1", "GPU_ID"): [0, 0, 0],
        ("file1", "Grid_Size"): [1024, 1024, 1024],
        ("file1", "Workgroup_Size"): [64, 64, 64],
        ("file1", "LDS_Per_Workgroup"): [32, 32, 32],
        ("file1", "Scratch_Per_Workitem"): [0, 0, 0],
        ("file1", "Arch_VGPR"): [16, 16, 16],
        ("file1", "Accum_VGPR"): [0, 0, 0],
        ("file1", "SGPR"): [32, 32, 32],
        ("file1", "Kernel_Name"): ["kernel_a", "kernel_a", "kernel_a"],
        ("file1", "Start_Timestamp"): [1000, 1200, 1400],
        ("file1", "End_Timestamp"): [1500, 1700, 1900],
        ("file1", "Kernel_ID"): [1, 1, 1],
        ("file1", "C1"): [100, None, None],
        ("file1", "C2"): [None, 500, 300],
    }
    df_policy = make_multilevel_df(data_policy)
    result_invalid = utils.impute_counters_iteration_multiplex(
        df_policy, "invalid_policy"
    )
    result_klp = utils.impute_counters_iteration_multiplex(
        df_policy, "kernel_launch_params"
    )
    assert isinstance(result_invalid, pd.DataFrame)
    pd.testing.assert_frame_equal(
        result_invalid.sort_values(by=("file1", "Dispatch_ID")).reset_index(drop=True),
        result_klp.sort_values(by=("file1", "Dispatch_ID")).reset_index(drop=True),
    )


def test_impute_counters_iteration_multiplex_multi_file():
    """
    Test imputation when the DataFrame has multiple collection levels (multi-file).

    Two file levels ("file1", "file2") each with independent dispatches.
    Tests the outer for-loop and final pd.concat across levels.
    """

    data_multi_file = {
        ("file1", "Dispatch_ID"): [1, 2],
        ("file1", "GPU_ID"): [0, 0],
        ("file1", "Grid_Size"): [1024, 1024],
        ("file1", "Workgroup_Size"): [64, 64],
        ("file1", "LDS_Per_Workgroup"): [32, 32],
        ("file1", "Scratch_Per_Workitem"): [0, 0],
        ("file1", "Arch_VGPR"): [16, 16],
        ("file1", "Accum_VGPR"): [0, 0],
        ("file1", "SGPR"): [32, 32],
        ("file1", "Kernel_Name"): ["kernel_a", "kernel_a"],
        ("file1", "Start_Timestamp"): [1000, 1200],
        ("file1", "End_Timestamp"): [1500, 1700],
        ("file1", "Kernel_ID"): [1, 1],
        ("file1", "C1"): [10, None],
        ("file1", "C2"): [None, 20],
        ("file2", "Dispatch_ID"): [1, 2],
        ("file2", "GPU_ID"): [0, 0],
        ("file2", "Grid_Size"): [512, 512],
        ("file2", "Workgroup_Size"): [32, 32],
        ("file2", "LDS_Per_Workgroup"): [16, 16],
        ("file2", "Scratch_Per_Workitem"): [0, 0],
        ("file2", "Arch_VGPR"): [8, 8],
        ("file2", "Accum_VGPR"): [0, 0],
        ("file2", "SGPR"): [16, 16],
        ("file2", "Kernel_Name"): ["kernel_b", "kernel_b"],
        ("file2", "Start_Timestamp"): [2000, 2200],
        ("file2", "End_Timestamp"): [2500, 2700],
        ("file2", "Kernel_ID"): [2, 2],
        ("file2", "C1"): [50, None],
        ("file2", "C2"): [None, 60],
    }
    df_multi_file = make_multilevel_df(data_multi_file)
    result = utils.impute_counters_iteration_multiplex(df_multi_file, "kernel")
    result = result.sort_values(by=("file1", "Dispatch_ID"))

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2

    # Verify both file levels are present in the output
    top_levels = result.columns.get_level_values(0).unique().tolist()
    assert "file1" in top_levels
    assert "file2" in top_levels

    # File1: C1 imputed into row 2, C2 imputed into row 1
    assert result[("file1", "C1")].iloc[0] == 10
    assert result[("file1", "C2")].iloc[0] == 20
    assert result[("file1", "C1")].iloc[1] == 10
    assert result[("file1", "C2")].iloc[1] == 20

    # File2: independent imputation (values must not bleed from file1)
    assert result[("file2", "C1")].iloc[0] == 50
    assert result[("file2", "C2")].iloc[0] == 60
    assert result[("file2", "C1")].iloc[1] == 50
    assert result[("file2", "C2")].iloc[1] == 60
