# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

from pathlib import Path

import pandas as pd

from interface.csv_data import CsvAnalysisData, CsvProfileDataWriter
from interface.factory import (
    create_profile_data_reader,
    create_profile_data_writer,
)
from interface.pmc_frame import to_canonical_pmc_frame
from interface.profile_data import ProfileDataReaderOptions
from interface.rocpd_data import RocpdAnalysisData, RocpdProfileDataWriter


def test_factory_selects_csv_and_rocpd_implementations():
    csv_reader = create_profile_data_reader(
        {"format_rocprof_output": "csv"},
        ProfileDataReaderOptions(),
    )
    rocpd_reader = create_profile_data_reader(
        {"format_rocprof_output": "rocpd"},
        ProfileDataReaderOptions(),
    )

    assert type(csv_reader).__name__ == "CsvProfileDataReader"
    assert type(rocpd_reader).__name__ == "RocpdProfileDataReader"
    assert isinstance(create_profile_data_writer("csv"), CsvProfileDataWriter)
    assert isinstance(
        create_profile_data_writer("rocpd"),
        RocpdProfileDataWriter,
    )


def test_csv_analysis_materializes_joined_pmc_perf(tmp_path):
    write_counter_result(
        tmp_path / "results_pmc_perf_SQ_WAVES.csv",
        "SQ_WAVES",
        "1",
    )
    write_counter_result(
        tmp_path / "results_pmc_perf_GRBM_COUNT.csv",
        "GRBM_COUNT",
        "2",
    )

    output_path = tmp_path / "pmc_perf.csv"
    CsvAnalysisData(join_type="grid").materialize_pmc_perf(tmp_path, output_path)

    frame = pd.read_csv(output_path)
    assert frame["SQ_WAVES"].tolist() == [1]
    assert frame["GRBM_COUNT"].tolist() == [2]
    assert "key" not in frame.columns


def test_rocpd_analysis_materializes_legacy_results(tmp_path):
    (tmp_path / "results_a.csv").write_text(
        "Dispatch_ID,Kernel_Name,Counter_Name,Counter_Value\n0,kernel,SQ_WAVES,1\n",
        encoding="utf-8",
    )
    (tmp_path / "results_b.csv").write_text(
        "Dispatch_ID,Kernel_Name,Counter_Name,Counter_Value\n1,kernel,GRBM_COUNT,2\n",
        encoding="utf-8",
    )

    output_path = tmp_path / "pmc_perf.csv"
    RocpdAnalysisData().materialize_pmc_perf(tmp_path, output_path)

    rows = output_path.read_text(encoding="utf-8").splitlines()
    assert rows == [
        "Dispatch_ID,Kernel_Name,Counter_Name,Counter_Value",
        "0,kernel,SQ_WAVES,1",
        "1,kernel,GRBM_COUNT,2",
    ]


def test_rocpd_long_counter_rows_convert_to_canonical_frame():
    frame = pd.DataFrame([
        {
            "Dispatch_ID": 7,
            "GPU_ID": 3,
            "Kernel_Name": "kernel",
            "Grid_Size": 64,
            "Workgroup_Size": 256,
            "LDS_Per_Workgroup": 0,
            "Counter_Name": "SQ_WAVES",
            "Counter_Value": 11,
        },
        {
            "Dispatch_ID": 7,
            "GPU_ID": 3,
            "Kernel_Name": "kernel",
            "Grid_Size": 64,
            "Workgroup_Size": 256,
            "LDS_Per_Workgroup": 0,
            "Counter_Name": "GRBM_COUNT",
            "Counter_Value": 22,
        },
    ])

    canonical_frame = to_canonical_pmc_frame(frame)

    assert canonical_frame["Dispatch_ID"].tolist() == [0]
    assert canonical_frame["GPU_ID"].tolist() == [0]
    assert canonical_frame["SQ_WAVES"].tolist() == [11]
    assert canonical_frame["GRBM_COUNT"].tolist() == [22]


def write_counter_result(csv_path: Path, counter_name: str, value: str) -> None:
    csv_path.write_text(
        "Kernel_Name,Grid_Size,Workgroup_Size,LDS_Per_Workgroup,"
        "Start_Timestamp,End_Timestamp,"
        f"{counter_name}\n"
        f"kernel,64,256,0,10,20,{value}\n",
        encoding="utf-8",
    )
