# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for the Phase D profiling data reader."""

import common

from utils.profile_data import read_rocpd_pmc_csv


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

    pmc_df = read_rocpd_pmc_csv(tmp_path, verbose=0)

    assert len(pmc_df) == 1
    assert pmc_df["SQ_WAVES"].iloc[0] == 4
    assert pmc_df["SQ_BUSY_CYCLES"].iloc[0] == 100
    assert pmc_df["Dispatch_Unit"].iloc[0] == 1


def test_reader_rejects_header_only_result_file(tmp_path, monkeypatch) -> None:
    """A header-only result artifact must not be accepted as profiling data."""
    result_file = tmp_path / "results_pmc_perf_0.csv.gz"
    common.write_gzip_csv(result_file, "Counter_Name,Counter_Value\n")
    errors: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "utils.profile_data.console_error",
        lambda category, message: errors.append((category, message)),
    )

    assert read_rocpd_pmc_csv(tmp_path, verbose=0).empty
    assert errors == [
        (
            "profiling",
            f"No counter data in {result_file}. Profiling data could be corrupt.",
        )
    ]
