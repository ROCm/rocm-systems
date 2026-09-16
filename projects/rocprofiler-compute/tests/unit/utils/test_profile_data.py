# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for the Phase D profiling data reader."""

import common
import pandas as pd

from utils.profile_data import export_pmc_data, get_profile_data_reader


def test_reader_combines_result_files_in_memory(tmp_path) -> None:
    """Multiple ROCPD result artifacts produce one pivoted PMC frame."""
    header = (
        "GPU_ID,Dispatch_ID,Grid_Size,Workgroup_Size,LDS_Per_Workgroup,"
        "Scratch_Per_Workitem,Arch_VGPR,Accum_VGPR,SGPR,Kernel_Name,"
        "Start_Timestamp,End_Timestamp,Kernel_ID,Counter_Name,Counter_Value\n"
    )
    row_prefix = "0,0,256,64,0,0,8,0,16,kernel_a,10,20,0,"
    common.write_gzip_csv(
        tmp_path / "results_pmc_perf_0.csv.gz", header + row_prefix + "SQ_WAVES,4\n"
    )
    common.write_gzip_csv(
        tmp_path / "results_pmc_perf_1.csv.gz",
        header + row_prefix + "SQ_BUSY_CYCLES,100\n",
    )

    pmc_df = get_profile_data_reader().read_pmc(tmp_path, verbose=0)

    assert len(pmc_df) == 1
    assert pmc_df["SQ_WAVES"].iloc[0] == 4
    assert pmc_df["SQ_BUSY_CYCLES"].iloc[0] == 100
    assert pmc_df["Dispatch_Unit"].iloc[0] == 1


def test_export_does_not_become_reader_input(tmp_path) -> None:
    """The debug export is one-way and is not treated as profiling data."""
    export_pmc_data(tmp_path, pd.DataFrame({"Kernel_Name": ["kernel_a"]}))

    assert get_profile_data_reader().read_pmc(tmp_path, verbose=0).empty
