# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

from pathlib import Path

import pandas as pd
import pytest

import utils.utils_analysis as utils


def seed_perfmon_files(tmp_path: Path, count: int) -> None:
    """Create empty pmc_perf_*.yaml files so the imputation function sees the
    expected number of counter buckets. Clears any existing perfmon files
    first so the helper is safe to call multiple times in one test."""
    perfmon = tmp_path / "perfmon"
    perfmon.mkdir(exist_ok=True)
    for stale in perfmon.glob("pmc_perf_*.yaml"):
        stale.unlink()
    for stale in perfmon.glob("*.txt"):
        stale.unlink()
    for i in range(count):
        (perfmon / f"pmc_perf_{i}.yaml").touch()


def test_impute_multiplex_kernel_policy(tmp_path: Path) -> None:
    """Test imputation with kernel policy on a single kernel."""

    data = {
        "Dispatch_ID": [1, 2, 3],
        "GPU_ID": [0, 0, 0],
        "Grid_Size": [1024, 512, 1024],
        "Workgroup_Size": [64, 64, 64],
        "LDS_Per_Workgroup": [32, 32, 32],
        "Scratch_Per_Workitem": [0, 0, 0],
        "Arch_VGPR": [16, 16, 16],
        "Accum_VGPR": [0, 0, 0],
        "SGPR": [32, 32, 32],
        "Kernel_Name": ["kernel_a", "kernel_a", "kernel_a"],
        "Start_Timestamp": [1000, 1200, 1400],
        "End_Timestamp": [1500, 1700, 1900],
        "Kernel_ID": [1, 1, 1],
        "Counter1": [100, None, None],
        "Counter2": [None, 500, 300],
    }

    df = pd.DataFrame(data)

    result = utils.impute_counters_iteration_multiplex(df, "kernel", tmp_path)

    # Sort by Dispatch_ID to ensure consistent order
    result = result.sort_values(by="Dispatch_ID")
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3  # Ensure same number of rows

    # Assert Counter2 imputed for first dispatch, Counter1 imputed for second dispatch
    assert result["Counter2"].iloc[0] == 500
    assert result["Counter1"].iloc[1] == 100


def test_impute_multiplex_kernel_launch_params_policy(tmp_path: Path) -> None:
    """Test imputation with kernel_launch_params policy on a single kernel."""

    data = {
        "Dispatch_ID": [1, 2, 3],
        "GPU_ID": [0, 0, 0],
        "Grid_Size": [1024, 512, 1024],
        "Workgroup_Size": [64, 64, 64],
        "LDS_Per_Workgroup": [32, 32, 32],
        "Scratch_Per_Workitem": [0, 0, 0],
        "Arch_VGPR": [16, 16, 16],
        "Accum_VGPR": [0, 0, 0],
        "SGPR": [32, 32, 32],
        "Kernel_Name": ["kernel_a", "kernel_a", "kernel_a"],
        "Start_Timestamp": [1000, 1200, 1400],
        "End_Timestamp": [1500, 1700, 1900],
        "Kernel_ID": [1, 1, 1],
        "Counter1": [100, None, None],
        "Counter2": [None, 500, 300],
    }

    df = pd.DataFrame(data)

    result = utils.impute_counters_iteration_multiplex(
        df, "kernel_launch_params", tmp_path
    )

    # Sort by Dispatch_ID to ensure consistent order
    result = result.sort_values(by="Dispatch_ID")

    # Assert Counter2 imputed for first dispatch, Counter1 imputed for last dispatch
    assert result["Counter2"].iloc[0] == 300
    assert result["Counter1"].iloc[2] == 100

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3


def test_impute_multiplex_kernel_launch_params_no_imputation(tmp_path: Path) -> None:
    """Test imputation with kernel_launch_params when no imputation is possible."""

    data = {
        "Dispatch_ID": [1, 2, 3],
        "GPU_ID": [0, 0, 0],
        "Grid_Size": [1024, 1024, 1024],
        "Workgroup_Size": [64, 64, 32],
        "LDS_Per_Workgroup": [32, 24, 32],
        "Scratch_Per_Workitem": [0, 0, 0],
        "Arch_VGPR": [16, 16, 16],
        "Accum_VGPR": [0, 0, 0],
        "SGPR": [32, 32, 32],
        "Kernel_Name": ["kernel_a", "kernel_a", "kernel_a"],
        "Start_Timestamp": [1000, 1200, 1400],
        "End_Timestamp": [1500, 1700, 1900],
        "Kernel_ID": [1, 1, 1],
        "Counter1": [100, None, 300],
        "Counter2": [None, 500, None],
    }

    df = pd.DataFrame(data)
    # Counter1 and Counter2 form 2 round-robin buckets.
    num_counter_bucket = 2
    seed_perfmon_files(tmp_path, count=num_counter_bucket)

    result = utils.impute_counters_iteration_multiplex(
        df, "kernel_launch_params", tmp_path
    )

    # Sort by Dispatch_ID to ensure consistent order
    result = result.sort_values(by="Dispatch_ID")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3  # Ensure same number of rows

    # Each dispatch still has NaN in the other counter after imputation,
    # so all counter columns are nullified for all dispatches.
    assert pd.isna(result["Counter1"].iloc[0])
    assert pd.isna(result["Counter2"].iloc[0])
    assert pd.isna(result["Counter1"].iloc[1])
    assert pd.isna(result["Counter2"].iloc[1])
    assert pd.isna(result["Counter1"].iloc[2])
    assert pd.isna(result["Counter2"].iloc[2])


def test_impute_multiplex_multi_kernel_kernel_policy(tmp_path: Path) -> None:
    """Test imputation with kernel policy on multiple kernels."""

    data = {
        "Dispatch_ID": [1, 2, 3],
        "GPU_ID": [0, 0, 0],
        "Grid_Size": [1024, 1024, 512],
        "Workgroup_Size": [64, 64, 64],
        "LDS_Per_Workgroup": [32, 32, 32],
        "Scratch_Per_Workitem": [0, 0, 0],
        "Arch_VGPR": [16, 16, 16],
        "Accum_VGPR": [0, 0, 0],
        "SGPR": [32, 32, 32],
        "Kernel_Name": ["kernel_a", "kernel_b", "kernel_a"],
        "Start_Timestamp": [1000, 1200, 1400],
        "End_Timestamp": [1500, 1700, 1900],
        "Kernel_ID": [1, 1, 1],
        "Counter1": [100, None, None],
        "Counter2": [None, 500, 300],
    }

    df = pd.DataFrame(data)

    result = utils.impute_counters_iteration_multiplex(df, "kernel", tmp_path)

    # Sort by Dispatch_ID to ensure consistent order
    result = result.sort_values(by="Dispatch_ID")

    # Assert Counter1 and Counter2 imputed for first and last dispatches
    assert result["Counter2"].iloc[0] == 300
    assert result["Counter1"].iloc[2] == 100

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3  # Ensure same number of rows


def test_impute_multiplex_multi_kernel_kernel_launch_params_no_imputation(
    tmp_path: Path,
) -> None:
    """Test imputation with kernel_launch_params when no imputation is possible."""

    data = {
        "Dispatch_ID": [1, 2, 3],
        "GPU_ID": [0, 0, 0],
        "Grid_Size": [1024, 1024, 512],
        "Workgroup_Size": [64, 64, 64],
        "LDS_Per_Workgroup": [32, 32, 32],
        "Scratch_Per_Workitem": [0, 0, 0],
        "Arch_VGPR": [16, 16, 16],
        "Accum_VGPR": [0, 0, 0],
        "SGPR": [32, 32, 32],
        "Kernel_Name": ["kernel_a", "kernel_b", "kernel_a"],
        "Start_Timestamp": [1000, 1200, 1400],
        "End_Timestamp": [1500, 1700, 1900],
        "Kernel_ID": [1, 1, 1],
        "Counter1": [100, None, None],
        "Counter2": [None, 500, 300],
    }

    df = pd.DataFrame(data)
    # Counter1 and Counter2 form 2 round-robin buckets.
    num_counter_bucket = 2
    seed_perfmon_files(tmp_path, count=num_counter_bucket)

    result = utils.impute_counters_iteration_multiplex(
        df, "kernel_launch_params", tmp_path
    )

    # Sort by Dispatch_ID to ensure consistent order
    result = result.sort_values(by="Dispatch_ID")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3  # Ensure same number of rows

    # Each dispatch still has NaN in the other counter after imputation,
    # so all counter columns are nullified for all dispatches.
    assert pd.isna(result["Counter1"].iloc[0])
    assert pd.isna(result["Counter2"].iloc[0])
    assert pd.isna(result["Counter1"].iloc[1])
    assert pd.isna(result["Counter2"].iloc[1])
    assert pd.isna(result["Counter1"].iloc[2])
    assert pd.isna(result["Counter2"].iloc[2])


def test_fewer_dispatches_single_kernel(tmp_path: Path) -> None:
    """
    Test imputation with kernel policy on a single kernel with
    fewer dispatches than buckets.

    1 kernel, 3 counters, only 2 dispatches (missing C3 bucket).
    C3 remains NaN because there are no previous_fill_values.
    """

    data = {
        "Dispatch_ID": [1, 2],
        "GPU_ID": [0, 0],
        "Grid_Size": [1024, 1024],
        "Workgroup_Size": [64, 64],
        "LDS_Per_Workgroup": [32, 32],
        "Scratch_Per_Workitem": [0, 0],
        "Arch_VGPR": [16, 16],
        "Accum_VGPR": [0, 0],
        "SGPR": [32, 32],
        "Kernel_Name": ["kernel_a", "kernel_a"],
        "Start_Timestamp": [1000, 1200],
        "End_Timestamp": [1500, 1700],
        "Kernel_ID": [1, 1],
        "C1": [10, None],
        "C2": [None, 20],
        "C3": [None, None],
    }

    df = pd.DataFrame(data)
    # C1, C2, C3 form 3 round-robin buckets but the kernel only had 2 dispatches.
    num_counter_bucket = 3
    seed_perfmon_files(tmp_path, count=num_counter_bucket)
    result = utils.impute_counters_iteration_multiplex(df, "kernel", tmp_path)
    result = result.sort_values(by="Dispatch_ID")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2

    # C3 was never collected (NaN on all rows after imputation), so all rows are
    # nullified — C1 and C2 are also set to NaN to fully exclude these dispatches.
    assert pd.isna(result["C1"].iloc[0])
    assert pd.isna(result["C2"].iloc[0])
    assert pd.isna(result["C3"].iloc[0])
    assert pd.isna(result["C1"].iloc[1])
    assert pd.isna(result["C2"].iloc[1])
    assert pd.isna(result["C3"].iloc[1])


def test_fewer_dispatches_multiple_kernels_both_incomplete(tmp_path: Path) -> None:
    """
    Test imputation with kernel policy on multiple kernels, both incomplete.

    kernel_a: buckets {C1}, {C2} (missing C3)
    kernel_b: buckets {C1}, {C2} (missing C3)
    """

    data = {
        "Dispatch_ID": [1, 2, 3, 4],
        "GPU_ID": [0, 0, 0, 0],
        "Grid_Size": [1024, 1024, 1024, 1024],
        "Workgroup_Size": [64, 64, 64, 64],
        "LDS_Per_Workgroup": [32, 32, 32, 32],
        "Scratch_Per_Workitem": [0, 0, 0, 0],
        "Arch_VGPR": [16, 16, 16, 16],
        "Accum_VGPR": [0, 0, 0, 0],
        "SGPR": [32, 32, 32, 32],
        "Kernel_Name": ["kernel_a", "kernel_a", "kernel_b", "kernel_b"],
        "Start_Timestamp": [1000, 1200, 1400, 1600],
        "End_Timestamp": [1500, 1700, 1900, 2100],
        "Kernel_ID": [1, 1, 2, 2],
        "C1": [10, None, 40, None],
        "C2": [None, 20, None, 60],
        "C3": [None, None, None, None],
    }

    df = pd.DataFrame(data)
    # C1, C2, C3 form 3 round-robin buckets but each kernel has only 2 dispatches.
    num_counter_bucket = 3
    seed_perfmon_files(tmp_path, count=num_counter_bucket)
    result = utils.impute_counters_iteration_multiplex(df, "kernel", tmp_path)
    result = result.sort_values(by="Dispatch_ID")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 4

    # kernel_a (dispatches 1-2): C3 never collected → all rows nullified
    assert pd.isna(result["C1"].iloc[0])
    assert pd.isna(result["C2"].iloc[0])
    assert pd.isna(result["C3"].iloc[0])
    assert pd.isna(result["C1"].iloc[1])
    assert pd.isna(result["C2"].iloc[1])
    assert pd.isna(result["C3"].iloc[1])

    # kernel_b (dispatches 3-4): C3 never collected → all rows nullified
    assert pd.isna(result["C1"].iloc[2])
    assert pd.isna(result["C2"].iloc[2])
    assert pd.isna(result["C3"].iloc[2])
    assert pd.isna(result["C1"].iloc[3])
    assert pd.isna(result["C2"].iloc[3])
    assert pd.isna(result["C3"].iloc[3])


def test_fewer_dispatches_one_incomplete_one_complete(tmp_path: Path) -> None:
    """
    Test imputation with kernel policy on one kernel incomplete, second complete.

    kernel_a: 2 dispatches (missing C3 bucket)
    kernel_b: 3 dispatches covering all 3 buckets
    """

    data = {
        "Dispatch_ID": [1, 2, 3, 4, 5],
        "GPU_ID": [0, 0, 0, 0, 0],
        "Grid_Size": [1024, 1024, 1024, 1024, 1024],
        "Workgroup_Size": [64, 64, 64, 64, 64],
        "LDS_Per_Workgroup": [32, 32, 32, 32, 32],
        "Scratch_Per_Workitem": [0, 0, 0, 0, 0],
        "Arch_VGPR": [16, 16, 16, 16, 16],
        "Accum_VGPR": [0, 0, 0, 0, 0],
        "SGPR": [32, 32, 32, 32, 32],
        "Kernel_Name": [
            "kernel_a",
            "kernel_a",
            "kernel_b",
            "kernel_b",
            "kernel_b",
        ],
        "Start_Timestamp": [1000, 1200, 1400, 1600, 1800],
        "End_Timestamp": [1500, 1700, 1900, 2100, 2300],
        "Kernel_ID": [1, 1, 2, 2, 2],
        "C1": [10, None, 50, None, None],
        "C2": [None, 20, None, 60, None],
        "C3": [None, None, None, None, 70],
    }

    df = pd.DataFrame(data)
    # C1, C2, C3 form 3 round-robin buckets; kernel_a has only 2 dispatches.
    num_counter_bucket = 3
    seed_perfmon_files(tmp_path, count=num_counter_bucket)
    result = utils.impute_counters_iteration_multiplex(df, "kernel", tmp_path)
    result = result.sort_values(by="Dispatch_ID")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 5

    # kernel_a (dispatches 1-2): C3 never collected → all rows nullified
    assert pd.isna(result["C1"].iloc[0])
    assert pd.isna(result["C2"].iloc[0])
    assert pd.isna(result["C3"].iloc[0])
    assert pd.isna(result["C1"].iloc[1])
    assert pd.isna(result["C2"].iloc[1])
    assert pd.isna(result["C3"].iloc[1])

    # kernel_b (dispatches 3-5): all 3 counters fully imputed, no NaN → not nullified
    assert result["C1"].iloc[2] == 50
    assert result["C2"].iloc[2] == 60
    assert result["C3"].iloc[2] == 70
    assert result["C1"].iloc[3] == 50
    assert result["C2"].iloc[3] == 60
    assert result["C3"].iloc[3] == 70
    assert result["C1"].iloc[4] == 50
    assert result["C2"].iloc[4] == 60
    assert result["C3"].iloc[4] == 70


def test_fewer_dispatches_same_kernel_different_launch_params(tmp_path: Path) -> None:
    """
    Test imputation with kernel_launch_params on the same kernel
    with different launch params.

    kernel_launch_params policy splits into 2 groups, each incomplete.
    Config 1 (Grid=1024, WG=64, LDS=32): buckets {C1}, {C2}
    Config 2 (Grid=512,  WG=32, LDS=16): buckets {C1}, {C2}
    """

    data = {
        "Dispatch_ID": [1, 2, 3, 4],
        "GPU_ID": [0, 0, 0, 0],
        "Grid_Size": [1024, 1024, 512, 512],
        "Workgroup_Size": [64, 64, 32, 32],
        "LDS_Per_Workgroup": [32, 32, 16, 16],
        "Scratch_Per_Workitem": [0, 0, 0, 0],
        "Arch_VGPR": [16, 16, 16, 16],
        "Accum_VGPR": [0, 0, 0, 0],
        "SGPR": [32, 32, 32, 32],
        "Kernel_Name": [
            "kernel_a",
            "kernel_a",
            "kernel_a",
            "kernel_a",
        ],
        "Start_Timestamp": [1000, 1200, 1400, 1600],
        "End_Timestamp": [1500, 1700, 1900, 2100],
        "Kernel_ID": [1, 1, 1, 1],
        "C1": [10, None, 30, None],
        "C2": [None, 20, None, 40],
        "C3": [None, None, None, None],
    }

    df = pd.DataFrame(data)
    # C1, C2, C3 form 3 round-robin buckets but each launch config has 2 dispatches.
    num_counter_bucket = 3
    seed_perfmon_files(tmp_path, count=num_counter_bucket)
    result = utils.impute_counters_iteration_multiplex(
        df, "kernel_launch_params", tmp_path
    )
    result = result.sort_values(by="Dispatch_ID")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 4

    # Config 1 (dispatches 1-2): C3 never collected → all rows nullified
    assert pd.isna(result["C1"].iloc[0])
    assert pd.isna(result["C2"].iloc[0])
    assert pd.isna(result["C3"].iloc[0])
    assert pd.isna(result["C1"].iloc[1])
    assert pd.isna(result["C2"].iloc[1])
    assert pd.isna(result["C3"].iloc[1])

    # Config 2 (dispatches 3-4): C3 never collected → all rows nullified
    assert pd.isna(result["C1"].iloc[2])
    assert pd.isna(result["C2"].iloc[2])
    assert pd.isna(result["C3"].iloc[2])
    assert pd.isna(result["C1"].iloc[3])
    assert pd.isna(result["C2"].iloc[3])
    assert pd.isna(result["C3"].iloc[3])


def test_fewer_dispatches_same_kernel_one_incomplete_one_complete(
    tmp_path: Path,
) -> None:
    """
    Test imputation with kernel_launch_params on one config incomplete, other complete.

    kernel_launch_params policy:
    Config 1 (Grid=1024, WG=64, LDS=32): 2 dispatches (missing C3 bucket)
    Config 2 (Grid=512,  WG=32, LDS=16): 3 dispatches (all 3 buckets)
    """

    data = {
        "Dispatch_ID": [1, 2, 3, 4, 5],
        "GPU_ID": [0, 0, 0, 0, 0],
        "Grid_Size": [1024, 1024, 512, 512, 512],
        "Workgroup_Size": [64, 64, 32, 32, 32],
        "LDS_Per_Workgroup": [32, 32, 16, 16, 16],
        "Scratch_Per_Workitem": [0, 0, 0, 0, 0],
        "Arch_VGPR": [16, 16, 16, 16, 16],
        "Accum_VGPR": [0, 0, 0, 0, 0],
        "SGPR": [32, 32, 32, 32, 32],
        "Kernel_Name": [
            "kernel_a",
            "kernel_a",
            "kernel_a",
            "kernel_a",
            "kernel_a",
        ],
        "Start_Timestamp": [1000, 1200, 1400, 1600, 1800],
        "End_Timestamp": [1500, 1700, 1900, 2100, 2300],
        "Kernel_ID": [1, 1, 1, 1, 1],
        "C1": [10, None, 50, None, None],
        "C2": [None, 20, None, 60, None],
        "C3": [None, None, None, None, 70],
    }

    df = pd.DataFrame(data)
    # C1, C2, C3 form 3 round-robin buckets; the first launch config has 2 dispatches.
    num_counter_bucket = 3
    seed_perfmon_files(tmp_path, count=num_counter_bucket)
    result = utils.impute_counters_iteration_multiplex(
        df, "kernel_launch_params", tmp_path
    )
    result = result.sort_values(by="Dispatch_ID")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 5

    # Config 1 (dispatches 1-2): C3 never collected → all rows nullified
    assert pd.isna(result["C1"].iloc[0])
    assert pd.isna(result["C2"].iloc[0])
    assert pd.isna(result["C3"].iloc[0])
    assert pd.isna(result["C1"].iloc[1])
    assert pd.isna(result["C2"].iloc[1])
    assert pd.isna(result["C3"].iloc[1])

    # Config 2 (dispatches 3-5): all 3 counters fully imputed, no NaN → not nullified
    assert result["C1"].iloc[2] == 50
    assert result["C2"].iloc[2] == 60
    assert result["C3"].iloc[2] == 70
    assert result["C1"].iloc[3] == 50
    assert result["C2"].iloc[3] == 60
    assert result["C3"].iloc[3] == 70
    assert result["C1"].iloc[4] == 50
    assert result["C2"].iloc[4] == 60
    assert result["C3"].iloc[4] == 70


def test_incomplete_last_group_single_kernel(tmp_path: Path) -> None:
    """
    Test imputation with kernel policy on a single kernel with incomplete last group.

    1 kernel, 2 counters, 3 dispatches (2 buckets, 1 full round + 1 trailing).
    The trailing subgroup uses previous_fill_values to fill its gaps.
    """

    data = {
        "Dispatch_ID": [1, 2, 3],
        "GPU_ID": [0, 0, 0],
        "Grid_Size": [1024, 1024, 1024],
        "Workgroup_Size": [64, 64, 64],
        "LDS_Per_Workgroup": [32, 32, 32],
        "Scratch_Per_Workitem": [0, 0, 0],
        "Arch_VGPR": [16, 16, 16],
        "Accum_VGPR": [0, 0, 0],
        "SGPR": [32, 32, 32],
        "Kernel_Name": ["kernel_a", "kernel_a", "kernel_a"],
        "Start_Timestamp": [1000, 1200, 1400],
        "End_Timestamp": [1500, 1700, 1900],
        "Kernel_ID": [1, 1, 1],
        "C1": [10, None, 30],
        "C2": [None, 20, None],
    }

    df = pd.DataFrame(data)
    result = utils.impute_counters_iteration_multiplex(df, "kernel", tmp_path)
    result = result.sort_values(by="Dispatch_ID")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3

    # Subgroup 1 (dispatches 1-2): C1 and C2 imputed within the subgroup
    assert result["C1"].iloc[0] == 10
    assert result["C2"].iloc[0] == 20
    assert result["C1"].iloc[1] == 10
    assert result["C2"].iloc[1] == 20

    # Subgroup 2 (dispatch 3, incomplete): C2 filled from previous subgroup
    # via cross-subgroup ffill; no NaN remains so the row is kept as valid.
    assert result["C1"].iloc[2] == 30
    assert result["C2"].iloc[2] == 20


def test_incomplete_last_group_multiple_kernels_both_incomplete(tmp_path: Path) -> None:
    """
    Test imputation with kernel policy on multiple kernels,
    both with incomplete last groups.

    kernel_a: 4 dispatches, 3 buckets {C1},{C2},{C3} (incomplete last)
    kernel_b: 5 dispatches, 3 buckets {C1},{C2},{C3} (incomplete last)
    """

    data = {
        "Dispatch_ID": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "GPU_ID": [0, 0, 0, 0, 0, 0, 0, 0, 0],
        "Grid_Size": [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024],
        "Workgroup_Size": [64, 64, 64, 64, 64, 64, 64, 64, 64],
        "LDS_Per_Workgroup": [32, 32, 32, 32, 32, 32, 32, 32, 32],
        "Scratch_Per_Workitem": [0, 0, 0, 0, 0, 0, 0, 0, 0],
        "Arch_VGPR": [16, 16, 16, 16, 16, 16, 16, 16, 16],
        "Accum_VGPR": [0, 0, 0, 0, 0, 0, 0, 0, 0],
        "SGPR": [32, 32, 32, 32, 32, 32, 32, 32, 32],
        "Kernel_Name": [
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
        "Start_Timestamp": [
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
        "End_Timestamp": [
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
        "Kernel_ID": [1, 1, 1, 1, 2, 2, 2, 2, 2],
        "C1": [10, None, None, 40, 50, None, None, 80, None],
        "C2": [None, 20, None, None, None, 60, None, None, 90],
        "C3": [None, None, 30, None, None, None, 70, None, None],
    }

    df = pd.DataFrame(data)
    result = utils.impute_counters_iteration_multiplex(df, "kernel", tmp_path)
    result = result.sort_values(by="Dispatch_ID")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 9

    # kernel_a subgroup 1 (dispatches 1-3): all 3 counters imputed within subgroup
    assert result["C1"].iloc[0] == 10
    assert result["C2"].iloc[0] == 20
    assert result["C3"].iloc[0] == 30
    assert result["C1"].iloc[1] == 10
    assert result["C2"].iloc[1] == 20
    assert result["C3"].iloc[1] == 30
    assert result["C1"].iloc[2] == 10
    assert result["C2"].iloc[2] == 20
    assert result["C3"].iloc[2] == 30

    # kernel_a subgroup 2 (dispatch 4, incomplete): filled via cross-subgroup ffill
    assert result["C1"].iloc[3] == 40
    assert result["C2"].iloc[3] == 20
    assert result["C3"].iloc[3] == 30

    # kernel_b subgroup 1 (dispatches 5-7): all 3 counters imputed within subgroup
    assert result["C1"].iloc[4] == 50
    assert result["C2"].iloc[4] == 60
    assert result["C3"].iloc[4] == 70
    assert result["C1"].iloc[5] == 50
    assert result["C2"].iloc[5] == 60
    assert result["C3"].iloc[5] == 70
    assert result["C1"].iloc[6] == 50
    assert result["C2"].iloc[6] == 60
    assert result["C3"].iloc[6] == 70

    # kernel_b subgroup 2 (dispatches 8-9, incomplete): filled via cross-subgroup ffill
    assert result["C1"].iloc[7] == 80
    assert result["C2"].iloc[7] == 90
    assert result["C3"].iloc[7] == 70
    assert result["C1"].iloc[8] == 80
    assert result["C2"].iloc[8] == 90
    assert result["C3"].iloc[8] == 70


def test_incomplete_last_group_one_incomplete_other_complete(tmp_path: Path) -> None:
    """
    Test imputation with kernel policy on one kernel incomplete, second kernel complete.

    kernel_a: 4 dispatches, 3 buckets {C1},{C2},{C3} (incomplete last)
    kernel_b: 6 dispatches, 3 buckets {C1},{C2},{C3} (2 complete rounds)
    """

    data = {
        "Dispatch_ID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "GPU_ID": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "Grid_Size": [
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
        "Workgroup_Size": [64, 64, 64, 64, 64, 64, 64, 64, 64, 64],
        "LDS_Per_Workgroup": [32, 32, 32, 32, 32, 32, 32, 32, 32, 32],
        "Scratch_Per_Workitem": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "Arch_VGPR": [16, 16, 16, 16, 16, 16, 16, 16, 16, 16],
        "Accum_VGPR": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "SGPR": [32, 32, 32, 32, 32, 32, 32, 32, 32, 32],
        "Kernel_Name": [
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
        "Start_Timestamp": [
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
        "End_Timestamp": [
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
        "Kernel_ID": [1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
        "C1": [10, None, None, 40, 50, None, None, 80, None, None],
        "C2": [None, 20, None, None, None, 60, None, None, 90, None],
        "C3": [None, None, 30, None, None, None, 70, None, None, 100],
    }

    df = pd.DataFrame(data)
    result = utils.impute_counters_iteration_multiplex(df, "kernel", tmp_path)
    result = result.sort_values(by="Dispatch_ID")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 10

    # kernel_a subgroup 1 (dispatches 1-3): all 3 counters imputed within subgroup
    assert result["C1"].iloc[0] == 10
    assert result["C2"].iloc[0] == 20
    assert result["C3"].iloc[0] == 30
    assert result["C1"].iloc[1] == 10
    assert result["C2"].iloc[1] == 20
    assert result["C3"].iloc[1] == 30
    assert result["C1"].iloc[2] == 10
    assert result["C2"].iloc[2] == 20
    assert result["C3"].iloc[2] == 30

    # kernel_a subgroup 2 (dispatch 4, incomplete): filled via cross-subgroup ffill
    assert result["C1"].iloc[3] == 40
    assert result["C2"].iloc[3] == 20
    assert result["C3"].iloc[3] == 30

    # kernel_b subgroup 1 (dispatches 5-7): all 3 counters imputed within subgroup
    assert result["C1"].iloc[4] == 50
    assert result["C2"].iloc[4] == 60
    assert result["C3"].iloc[4] == 70
    assert result["C1"].iloc[5] == 50
    assert result["C2"].iloc[5] == 60
    assert result["C3"].iloc[5] == 70
    assert result["C1"].iloc[6] == 50
    assert result["C2"].iloc[6] == 60
    assert result["C3"].iloc[6] == 70

    # kernel_b subgroup 2 (dispatches 8-10): complete round, no nullification
    assert result["C1"].iloc[7] == 80
    assert result["C2"].iloc[7] == 90
    assert result["C3"].iloc[7] == 100
    assert result["C1"].iloc[8] == 80
    assert result["C2"].iloc[8] == 90
    assert result["C3"].iloc[8] == 100
    assert result["C1"].iloc[9] == 80
    assert result["C2"].iloc[9] == 90
    assert result["C3"].iloc[9] == 100


def test_incomplete_last_group_same_kernel_different_launch_params(
    tmp_path: Path,
) -> None:
    """
    Test imputation with kernel_launch_params on the same kernel
    with different launch params.

    kernel_launch_params policy, both configs have incomplete last subgroups.
    Config 1 (Grid=1024, WG=64, LDS=32): 3 dispatches, 2 buckets
    Config 2 (Grid=512,  WG=32, LDS=16): 3 dispatches, 2 buckets
    """

    data = {
        "Dispatch_ID": [1, 2, 3, 4, 5, 6],
        "GPU_ID": [0, 0, 0, 0, 0, 0],
        "Grid_Size": [1024, 1024, 1024, 512, 512, 512],
        "Workgroup_Size": [64, 64, 64, 32, 32, 32],
        "LDS_Per_Workgroup": [32, 32, 32, 16, 16, 16],
        "Scratch_Per_Workitem": [0, 0, 0, 0, 0, 0],
        "Arch_VGPR": [16, 16, 16, 16, 16, 16],
        "Accum_VGPR": [0, 0, 0, 0, 0, 0],
        "SGPR": [32, 32, 32, 32, 32, 32],
        "Kernel_Name": [
            "kernel_a",
            "kernel_a",
            "kernel_a",
            "kernel_a",
            "kernel_a",
            "kernel_a",
        ],
        "Start_Timestamp": [1000, 1200, 1400, 1600, 1800, 2000],
        "End_Timestamp": [1500, 1700, 1900, 2100, 2300, 2500],
        "Kernel_ID": [1, 1, 1, 1, 1, 1],
        "C1": [10, None, 30, 50, None, 70],
        "C2": [None, 20, None, None, 60, None],
    }

    df = pd.DataFrame(data)
    result = utils.impute_counters_iteration_multiplex(
        df, "kernel_launch_params", tmp_path
    )
    result = result.sort_values(by="Dispatch_ID")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 6

    # Config 1 (dispatches 1-3): subgroup 0 (1-2) complete, subgroup 1 (3) filled
    # via cross-subgroup ffill; no NaN remains so dispatch 3 is kept as valid.
    assert result["C1"].iloc[0] == 10
    assert result["C2"].iloc[0] == 20
    assert result["C1"].iloc[1] == 10
    assert result["C2"].iloc[1] == 20
    assert result["C1"].iloc[2] == 30
    assert result["C2"].iloc[2] == 20

    # Config 2 (dispatches 4-6): subgroup 0 (4-5) complete, subgroup 1 (6) filled
    assert result["C1"].iloc[3] == 50
    assert result["C2"].iloc[3] == 60
    assert result["C1"].iloc[4] == 50
    assert result["C2"].iloc[4] == 60
    assert result["C1"].iloc[5] == 70
    assert result["C2"].iloc[5] == 60


def test_incomplete_last_group_same_kernel_one_incomplete_one_complete(
    tmp_path: Path,
) -> None:
    """
    Test imputation with kernel_launch_params on the same kernel
    with one config incomplete, other complete.

    kernel_launch_params policy:
    Config 1 (Grid=1024, WG=64, LDS=32): 3 dispatches, incomplete last
    Config 2 (Grid=512,  WG=32, LDS=16): 4 dispatches, 2 complete rounds
    """

    data = {
        "Dispatch_ID": [1, 2, 3, 4, 5, 6, 7],
        "GPU_ID": [0, 0, 0, 0, 0, 0, 0],
        "Grid_Size": [1024, 1024, 1024, 512, 512, 512, 512],
        "Workgroup_Size": [64, 64, 64, 32, 32, 32, 32],
        "LDS_Per_Workgroup": [32, 32, 32, 16, 16, 16, 16],
        "Scratch_Per_Workitem": [0, 0, 0, 0, 0, 0, 0],
        "Arch_VGPR": [16, 16, 16, 16, 16, 16, 16],
        "Accum_VGPR": [0, 0, 0, 0, 0, 0, 0],
        "SGPR": [32, 32, 32, 32, 32, 32, 32],
        "Kernel_Name": [
            "kernel_a",
            "kernel_a",
            "kernel_a",
            "kernel_a",
            "kernel_a",
            "kernel_a",
            "kernel_a",
        ],
        "Start_Timestamp": [1000, 1200, 1400, 1600, 1800, 2000, 2200],
        "End_Timestamp": [1500, 1700, 1900, 2100, 2300, 2500, 2700],
        "Kernel_ID": [1, 1, 1, 1, 1, 1, 1],
        "C1": [10, None, 30, 50, None, 70, None],
        "C2": [None, 20, None, None, 60, None, 80],
    }

    df = pd.DataFrame(data)
    result = utils.impute_counters_iteration_multiplex(
        df, "kernel_launch_params", tmp_path
    )
    result = result.sort_values(by="Dispatch_ID")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 7

    # Config 1 (dispatches 1-3): subgroup 0 (1-2) complete, subgroup 1 (3) filled
    # via cross-subgroup ffill; no NaN remains so dispatch 3 is kept as valid.
    assert result["C1"].iloc[0] == 10
    assert result["C2"].iloc[0] == 20
    assert result["C1"].iloc[1] == 10
    assert result["C2"].iloc[1] == 20
    assert result["C1"].iloc[2] == 30
    assert result["C2"].iloc[2] == 20

    # Config 2 (dispatches 4-7): 2 complete rounds, no nullification
    assert result["C1"].iloc[3] == 50
    assert result["C2"].iloc[3] == 60
    assert result["C1"].iloc[4] == 50
    assert result["C2"].iloc[4] == 60
    assert result["C1"].iloc[5] == 70
    assert result["C2"].iloc[5] == 80
    assert result["C1"].iloc[6] == 70
    assert result["C2"].iloc[6] == 80


def test_complete_last_group_single_kernel(tmp_path: Path) -> None:
    """
    Test imputation with kernel policy on a single kernel with complete last group.

    1 kernel, 2 counters, 4 dispatches (2 complete rounds of 2 buckets).
    All imputation happens within subgroups; previous_fill_values fallback
    is never needed.
    """

    data = {
        "Dispatch_ID": [1, 2, 3, 4],
        "GPU_ID": [0, 0, 0, 0],
        "Grid_Size": [1024, 1024, 1024, 1024],
        "Workgroup_Size": [64, 64, 64, 64],
        "LDS_Per_Workgroup": [32, 32, 32, 32],
        "Scratch_Per_Workitem": [0, 0, 0, 0],
        "Arch_VGPR": [16, 16, 16, 16],
        "Accum_VGPR": [0, 0, 0, 0],
        "SGPR": [32, 32, 32, 32],
        "Kernel_Name": ["kernel_a", "kernel_a", "kernel_a", "kernel_a"],
        "Start_Timestamp": [1000, 1200, 1400, 1600],
        "End_Timestamp": [1500, 1700, 1900, 2100],
        "Kernel_ID": [1, 1, 1, 1],
        "C1": [10, None, 30, None],
        "C2": [None, 20, None, 40],
    }

    df = pd.DataFrame(data)
    result = utils.impute_counters_iteration_multiplex(df, "kernel", tmp_path)
    result = result.sort_values(by="Dispatch_ID")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 4

    # Subgroup 1 (dispatches 1-2): self-contained imputation
    assert result["C1"].iloc[0] == 10
    assert result["C2"].iloc[0] == 20
    assert result["C1"].iloc[1] == 10
    assert result["C2"].iloc[1] == 20

    # Subgroup 2 (dispatches 3-4): self-contained, values don't bleed from subgroup 1
    assert result["C1"].iloc[2] == 30
    assert result["C2"].iloc[2] == 40
    assert result["C1"].iloc[3] == 30
    assert result["C2"].iloc[3] == 40


def test_complete_last_group_multiple_kernels_both_complete(tmp_path: Path) -> None:
    """
    Test imputation with kernel policy on multiple kernels, both complete.

    kernel_a: 6 dispatches, 3 buckets {C1},{C2},{C3}, 2 complete rounds
    kernel_b: 6 dispatches, 3 buckets {C1},{C2},{C3}, 2 complete rounds
    """

    data = {
        "Dispatch_ID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "GPU_ID": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "Grid_Size": [
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
        "Workgroup_Size": [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64],
        "LDS_Per_Workgroup": [
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
        "Scratch_Per_Workitem": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "Arch_VGPR": [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16],
        "Accum_VGPR": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "SGPR": [32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32],
        "Kernel_Name": [
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
        "Start_Timestamp": [
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
        "End_Timestamp": [
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
        "Kernel_ID": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
        "C1": [
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
        "C2": [
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
        "C3": [
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

    df = pd.DataFrame(data)
    result = utils.impute_counters_iteration_multiplex(df, "kernel", tmp_path)
    result = result.sort_values(by="Dispatch_ID")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 12

    # kernel_a round 1 (dispatches 1-3)
    assert result["C1"].iloc[0] == 10
    assert result["C2"].iloc[0] == 20
    assert result["C3"].iloc[0] == 30
    assert result["C1"].iloc[1] == 10
    assert result["C2"].iloc[1] == 20
    assert result["C3"].iloc[1] == 30
    assert result["C1"].iloc[2] == 10
    assert result["C2"].iloc[2] == 20
    assert result["C3"].iloc[2] == 30

    # kernel_a round 2 (dispatches 4-6)
    assert result["C1"].iloc[3] == 40
    assert result["C2"].iloc[3] == 50
    assert result["C3"].iloc[3] == 60
    assert result["C1"].iloc[4] == 40
    assert result["C2"].iloc[4] == 50
    assert result["C3"].iloc[4] == 60
    assert result["C1"].iloc[5] == 40
    assert result["C2"].iloc[5] == 50
    assert result["C3"].iloc[5] == 60

    # kernel_b round 1 (dispatches 7-9)
    assert result["C1"].iloc[6] == 70
    assert result["C2"].iloc[6] == 80
    assert result["C3"].iloc[6] == 90
    assert result["C1"].iloc[7] == 70
    assert result["C2"].iloc[7] == 80
    assert result["C3"].iloc[7] == 90
    assert result["C1"].iloc[8] == 70
    assert result["C2"].iloc[8] == 80
    assert result["C3"].iloc[8] == 90

    # kernel_b round 2 (dispatches 10-12)
    assert result["C1"].iloc[9] == 100
    assert result["C2"].iloc[9] == 110
    assert result["C3"].iloc[9] == 120
    assert result["C1"].iloc[10] == 100
    assert result["C2"].iloc[10] == 110
    assert result["C3"].iloc[10] == 120
    assert result["C1"].iloc[11] == 100
    assert result["C2"].iloc[11] == 110
    assert result["C3"].iloc[11] == 120


def test_complete_last_group_same_kernel_different_launch_params(
    tmp_path: Path,
) -> None:
    """
    Test imputation with kernel_launch_params on the same kernel
    with different launch params.

    kernel_launch_params policy, both configs have 2 complete rounds.
    Config 1 (Grid=1024, WG=64, LDS=32): 4 dispatches
    Config 2 (Grid=512,  WG=32, LDS=16): 4 dispatches
    """

    data = {
        "Dispatch_ID": [1, 2, 3, 4, 5, 6, 7, 8],
        "GPU_ID": [0, 0, 0, 0, 0, 0, 0, 0],
        "Grid_Size": [1024, 1024, 1024, 1024, 512, 512, 512, 512],
        "Workgroup_Size": [64, 64, 64, 64, 32, 32, 32, 32],
        "LDS_Per_Workgroup": [32, 32, 32, 32, 16, 16, 16, 16],
        "Scratch_Per_Workitem": [0, 0, 0, 0, 0, 0, 0, 0],
        "Arch_VGPR": [16, 16, 16, 16, 16, 16, 16, 16],
        "Accum_VGPR": [0, 0, 0, 0, 0, 0, 0, 0],
        "SGPR": [32, 32, 32, 32, 32, 32, 32, 32],
        "Kernel_Name": [
            "kernel_a",
            "kernel_a",
            "kernel_a",
            "kernel_a",
            "kernel_a",
            "kernel_a",
            "kernel_a",
            "kernel_a",
        ],
        "Start_Timestamp": [1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400],
        "End_Timestamp": [1500, 1700, 1900, 2100, 2300, 2500, 2700, 2900],
        "Kernel_ID": [1, 1, 1, 1, 1, 1, 1, 1],
        "C1": [10, None, 30, None, 50, None, 70, None],
        "C2": [None, 20, None, 40, None, 60, None, 80],
    }

    df = pd.DataFrame(data)
    result = utils.impute_counters_iteration_multiplex(
        df, "kernel_launch_params", tmp_path
    )
    result = result.sort_values(by="Dispatch_ID")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 8

    # Config 1 round 1 (dispatches 1-2)
    assert result["C1"].iloc[0] == 10
    assert result["C2"].iloc[0] == 20
    assert result["C1"].iloc[1] == 10
    assert result["C2"].iloc[1] == 20

    # Config 1 round 2 (dispatches 3-4)
    assert result["C1"].iloc[2] == 30
    assert result["C2"].iloc[2] == 40
    assert result["C1"].iloc[3] == 30
    assert result["C2"].iloc[3] == 40

    # Config 2 round 1 (dispatches 5-6)
    assert result["C1"].iloc[4] == 50
    assert result["C2"].iloc[4] == 60
    assert result["C1"].iloc[5] == 50
    assert result["C2"].iloc[5] == 60

    # Config 2 round 2 (dispatches 7-8)
    assert result["C1"].iloc[6] == 70
    assert result["C2"].iloc[6] == 80
    assert result["C1"].iloc[7] == 70
    assert result["C2"].iloc[7] == 80


def test_impute_counters_iteration_multiplex_missing_kernel_name(
    tmp_path: Path,
) -> None:
    """
    Test imputation when the DataFrame is a valid 2-level MultiIndex
    but without the Kernel_Name column raises a KeyError.
    """

    data_no_kernel_name = {
        "Dispatch_ID": [1, 2],
        "GPU_ID": [0, 0],
        "Grid_Size": [1024, 1024],
        "Workgroup_Size": [64, 64],
        "LDS_Per_Workgroup": [32, 32],
        "Scratch_Per_Workitem": [0, 0],
        "Arch_VGPR": [16, 16],
        "Accum_VGPR": [0, 0],
        "SGPR": [32, 32],
        "Start_Timestamp": [1000, 1200],
        "End_Timestamp": [1500, 1700],
        "Kernel_ID": [1, 1],
        "C1": [10, None],
        "C2": [None, 20],
    }
    df_no_kn = pd.DataFrame(data_no_kernel_name)
    with pytest.raises(KeyError):
        utils.impute_counters_iteration_multiplex(df_no_kn, "kernel", tmp_path)


def test_impute_counters_iteration_multiplex_empty_dataframe(tmp_path: Path) -> None:
    """Test imputation when the DataFrame is a valid MultiIndex but has no data rows."""

    data_empty = {
        "Dispatch_ID": [],
        "GPU_ID": [],
        "Grid_Size": [],
        "Workgroup_Size": [],
        "LDS_Per_Workgroup": [],
        "Scratch_Per_Workitem": [],
        "Arch_VGPR": [],
        "Accum_VGPR": [],
        "SGPR": [],
        "Kernel_Name": [],
        "Start_Timestamp": [],
        "End_Timestamp": [],
        "Kernel_ID": [],
        "C1": [],
        "C2": [],
    }
    df_empty = pd.DataFrame(data_empty)
    result = utils.impute_counters_iteration_multiplex(df_empty, "kernel", tmp_path)

    # Empty-group fallback preserves the input schema with zero rows.
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == list(df_empty.columns)
    assert len(result) == 0


def test_impute_counters_iteration_multiplex_all_counters_nan(tmp_path: Path) -> None:
    """
    Test imputation when all counter values are NaN.

    The bucket-identification loop finds no non-empty frozensets, so
    counter_groups stays empty and the group is skipped entirely.
    """

    data_all_nan = {
        "Dispatch_ID": [1, 2, 3],
        "GPU_ID": [0, 0, 0],
        "Grid_Size": [1024, 1024, 1024],
        "Workgroup_Size": [64, 64, 64],
        "LDS_Per_Workgroup": [32, 32, 32],
        "Scratch_Per_Workitem": [0, 0, 0],
        "Arch_VGPR": [16, 16, 16],
        "Accum_VGPR": [0, 0, 0],
        "SGPR": [32, 32, 32],
        "Kernel_Name": ["kernel_a", "kernel_a", "kernel_a"],
        "Start_Timestamp": [1000, 1200, 1400],
        "End_Timestamp": [1500, 1700, 1900],
        "Kernel_ID": [1, 1, 1],
        "C1": [None, None, None],
        "C2": [None, None, None],
    }
    df_all_nan = pd.DataFrame(data_all_nan)
    result = utils.impute_counters_iteration_multiplex(df_all_nan, "kernel", tmp_path)

    # Group was dropped (no valid counters) -- empty schema-aligned frame.
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == list(df_all_nan.columns)
    assert len(result) == 0


def test_impute_counters_iteration_multiplex_no_counter_columns(tmp_path: Path) -> None:
    """
    Test imputation when the DataFrame contains only the 13 non-counter columns.

    counter_columns is empty, so every row yields an empty frozenset
    and the group is skipped.
    """

    data_no_counters = {
        "Dispatch_ID": [1, 2],
        "GPU_ID": [0, 0],
        "Grid_Size": [1024, 1024],
        "Workgroup_Size": [64, 64],
        "LDS_Per_Workgroup": [32, 32],
        "Scratch_Per_Workitem": [0, 0],
        "Arch_VGPR": [16, 16],
        "Accum_VGPR": [0, 0],
        "SGPR": [32, 32],
        "Kernel_Name": ["kernel_a", "kernel_a"],
        "Start_Timestamp": [1000, 1200],
        "End_Timestamp": [1500, 1700],
        "Kernel_ID": [1, 1],
    }
    df_no_counters = pd.DataFrame(data_no_counters)
    result = utils.impute_counters_iteration_multiplex(
        df_no_counters, "kernel", tmp_path
    )

    # Group was dropped (no counter columns exist) -- empty schema-aligned frame.
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == list(df_no_counters.columns)
    assert len(result) == 0


def test_impute_counters_iteration_multiplex_unrecognized_policy(
    tmp_path: Path,
) -> None:
    """
    Test imputation when the policy is unrecognized.
    Any policy other than "kernel" falls through to the else branch
    (same as "kernel_launch_params"). The output must match exactly.
    """

    data_policy = {
        "Dispatch_ID": [1, 2, 3],
        "GPU_ID": [0, 0, 0],
        "Grid_Size": [1024, 1024, 1024],
        "Workgroup_Size": [64, 64, 64],
        "LDS_Per_Workgroup": [32, 32, 32],
        "Scratch_Per_Workitem": [0, 0, 0],
        "Arch_VGPR": [16, 16, 16],
        "Accum_VGPR": [0, 0, 0],
        "SGPR": [32, 32, 32],
        "Kernel_Name": ["kernel_a", "kernel_a", "kernel_a"],
        "Start_Timestamp": [1000, 1200, 1400],
        "End_Timestamp": [1500, 1700, 1900],
        "Kernel_ID": [1, 1, 1],
        "C1": [100, None, None],
        "C2": [None, 500, 300],
    }
    df_policy = pd.DataFrame(data_policy)
    result_invalid = utils.impute_counters_iteration_multiplex(
        df_policy, "invalid_policy", tmp_path
    )
    result_klp = utils.impute_counters_iteration_multiplex(
        df_policy, "kernel_launch_params", tmp_path
    )
    assert isinstance(result_invalid, pd.DataFrame)
    pd.testing.assert_frame_equal(
        result_invalid.sort_values(by="Dispatch_ID").reset_index(drop=True),
        result_klp.sort_values(by="Dispatch_ID").reset_index(drop=True),
    )


def test_incomplete_dispatches_nullify_counter_values(tmp_path: Path) -> None:
    """
    After imputation, any dispatch row that still has at least one NaN counter
    value should have ALL counter columns set to NaN (fully nullified).
    Non-counter columns (timestamps, kernel name, etc.) must be preserved so
    that Top Stats (Block 1) timing data remains accurate.

    Scenario:
      kernel_a: 2 dispatches, 3 counter buckets {C1}, {C2}, {C3}.
      Only 2 dispatches are available so the {C3} bucket is never reached:
        - Dispatch 1: C1=10, C2=NaN, C3=NaN
        - Dispatch 2: C1=NaN, C2=20, C3=NaN
      After bfill/ffill imputation:
        - C1 and C2 are filled for both dispatches (C1=10, C2=20)
        - C3 remains NaN for both dispatches (never collected)
      Post-imputation nullification:
        - Both dispatches have C3=NaN -> all counter columns set to NaN
        - Timestamp and Kernel_Name columns are preserved
    """
    data = {
        "Dispatch_ID": [1, 2],
        "GPU_ID": [0, 0],
        "Grid_Size": [1024, 1024],
        "Workgroup_Size": [64, 64],
        "LDS_Per_Workgroup": [32, 32],
        "Scratch_Per_Workitem": [0, 0],
        "Arch_VGPR": [16, 16],
        "Accum_VGPR": [0, 0],
        "SGPR": [32, 32],
        "Kernel_Name": ["kernel_a", "kernel_a"],
        "Start_Timestamp": [1000, 1200],
        "End_Timestamp": [1500, 1700],
        "Kernel_ID": [1, 1],
        "C1": [10, None],
        "C2": [None, 20],
        "C3": [None, None],
    }
    df = pd.DataFrame(data)
    # C1, C2, C3 form 3 round-robin buckets but the kernel only had 2 dispatches.
    num_counter_bucket = 3
    seed_perfmon_files(tmp_path, count=num_counter_bucket)
    result = utils.impute_counters_iteration_multiplex(df, "kernel", tmp_path)
    result = result.sort_values(by="Dispatch_ID").reset_index(drop=True)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2

    # Both dispatches: C3 was never collected so it remains NaN after imputation,
    # triggering nullification of all counter columns on both rows.
    assert pd.isna(result["C1"].iloc[0])
    assert pd.isna(result["C2"].iloc[0])
    assert pd.isna(result["C3"].iloc[0])
    assert pd.isna(result["C1"].iloc[1])
    assert pd.isna(result["C2"].iloc[1])
    assert pd.isna(result["C3"].iloc[1])

    # Non-counter columns must still be populated on both dispatches
    # (preserved for Top Stats / Block 1 timing display).
    assert result["Start_Timestamp"].iloc[0] == 1000
    assert result["End_Timestamp"].iloc[0] == 1500
    assert result["Kernel_Name"].iloc[0] == "kernel_a"
    assert result["Start_Timestamp"].iloc[1] == 1200
    assert result["End_Timestamp"].iloc[1] == 1700
    assert result["Kernel_Name"].iloc[1] == "kernel_a"


def test_undersampled_kernel_nullified_against_perfmon_file_count(
    tmp_path: Path,
) -> None:
    """
    A kernel whose dispatch count is below the number of configured perfmon
    files must be nullified even when its visible counter columns are fully
    imputed. This guards the degenerate case where some buckets never reached
    the joined dataframe at all.
    """
    # 5 perfmon buckets configured, kernel only has 2 dispatches; the visible
    # counters look fully populated but bucket coverage is incomplete.
    num_counter_bucket = 5
    seed_perfmon_files(tmp_path, count=num_counter_bucket)

    data = {
        "Dispatch_ID": [1, 2],
        "GPU_ID": [0, 0],
        "Grid_Size": [1024, 1024],
        "Workgroup_Size": [64, 64],
        "LDS_Per_Workgroup": [32, 32],
        "Scratch_Per_Workitem": [0, 0],
        "Arch_VGPR": [16, 16],
        "Accum_VGPR": [0, 0],
        "SGPR": [32, 32],
        "Kernel_Name": ["kernel_a", "kernel_a"],
        "Start_Timestamp": [1000, 1200],
        "End_Timestamp": [1500, 1700],
        "Kernel_ID": [1, 1],
        "C1": [10, 30],
        "C2": [20, 40],
    }
    df = pd.DataFrame(data)
    result = utils.impute_counters_iteration_multiplex(df, "kernel", tmp_path)
    result = result.sort_values(by="Dispatch_ID").reset_index(drop=True)

    assert pd.isna(result["C1"].iloc[0])
    assert pd.isna(result["C2"].iloc[0])
    assert pd.isna(result["C1"].iloc[1])
    assert pd.isna(result["C2"].iloc[1])

    # Timestamps and kernel name preserved for Top Stats.
    assert result["Start_Timestamp"].iloc[0] == 1000
    assert result["Kernel_Name"].iloc[0] == "kernel_a"
