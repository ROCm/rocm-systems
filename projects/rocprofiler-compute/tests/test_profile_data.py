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
from interface.profile_data import ProfileDataReaderOptions, ProfilePassContext
from interface.rocpd_data import RocpdAnalysisData, RocpdProfileDataWriter
from orchestrator.rocprofiler_sdk import (
    RocprofilerSdkAnalysisOrchestrator,
    RocprofilerSdkProfileOrchestrator,
)
from orchestrator.rocprofv3 import (
    Rocprofv3AnalysisOrchestrator,
    Rocprofv3ProfileOrchestrator,
)


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


def test_rocprofv3_profile_orchestrator_runs_and_finalizes(monkeypatch, tmp_path):
    calls = []
    pmc_file = tmp_path / "pmc_perf_SQ.yaml"
    pmc_file.write_text("pmc: []\n", encoding="utf-8")

    class FakeWriter:
        def finalize_pass(self, context: ProfilePassContext) -> None:
            calls.append(context)

    monkeypatch.setattr(
        "orchestrator.rocprofv3.build_counter_collection_environment",
        lambda fnames: {},
    )
    monkeypatch.setattr(
        "orchestrator.rocprofv3.cleanup_counter_collection_environment",
        lambda new_env: None,
    )
    monkeypatch.setattr(
        "orchestrator.rocprofv3.capture_subprocess_output",
        lambda command, new_env, profileMode: (True, ""),
    )
    monkeypatch.setattr(
        "orchestrator.rocprofv3.create_profile_data_writer",
        lambda data_format: FakeWriter(),
    )

    Rocprofv3ProfileOrchestrator().run_pass(
        str(pmc_file),
        ["--arg"],
        str(tmp_path),
        "csv",
    )

    assert len(calls) == 1
    assert calls[0].profiler_command == "rocprofv3"
    assert calls[0].fbase == "pmc_perf_SQ"


def test_rocprofiler_sdk_profile_orchestrator_runs_and_finalizes(
    monkeypatch,
    tmp_path,
):
    calls = []
    pmc_file = tmp_path / "pmc_perf_SQ.yaml"
    pmc_file.write_text("pmc: SQ_WAVES\n", encoding="utf-8")

    class FakeWriter:
        def finalize_pass(self, context: ProfilePassContext) -> None:
            calls.append(context)

    monkeypatch.setattr(
        "orchestrator.rocprofiler_sdk.build_counter_collection_environment",
        lambda fnames: {},
    )
    monkeypatch.setattr(
        "orchestrator.rocprofiler_sdk.cleanup_counter_collection_environment",
        lambda new_env: None,
    )
    monkeypatch.setattr(
        "orchestrator.rocprofiler_sdk.capture_subprocess_output",
        lambda command, new_env, profileMode: (True, ""),
    )
    monkeypatch.setattr(
        "orchestrator.rocprofiler_sdk.create_profile_data_writer",
        lambda data_format: FakeWriter(),
    )

    RocprofilerSdkProfileOrchestrator().run_pass(
        str(pmc_file),
        {"APP_CMD": ["app"]},
        str(tmp_path),
        "rocpd",
    )

    assert len(calls) == 1
    assert calls[0].profiler_command == "rocprofiler-sdk"
    assert calls[0].fbase == "pmc_perf_SQ"


def test_analysis_orchestrators_delegate_to_reader(monkeypatch, tmp_path):
    calls = []

    class FakeReader:
        def has_profile_data(self, workload_dir: Path) -> bool:
            calls.append(("has", workload_dir))
            return True

        def materialize_pmc_perf(self, workload_dir: Path, output_path: Path) -> Path:
            calls.append(("materialize", workload_dir, output_path))
            return output_path

        def read_pmc_frame(self, workload_dir: Path) -> pd.DataFrame:
            calls.append(("read", workload_dir))
            return pd.DataFrame({"SQ_WAVES": [1]})

    monkeypatch.setattr(
        "orchestrator.common.create_profile_data_reader",
        lambda profiling_config, options: FakeReader(),
    )

    for orchestrator in [
        Rocprofv3AnalysisOrchestrator(),
        RocprofilerSdkAnalysisOrchestrator(),
    ]:
        profiling_config = {"format_rocprof_output": "csv"}
        options = ProfileDataReaderOptions()
        output_path = tmp_path / "pmc_perf.csv"

        assert orchestrator.has_profile_data(tmp_path, profiling_config, options)
        assert (
            orchestrator.materialize_pmc_perf(
                tmp_path,
                output_path,
                profiling_config,
                options,
            )
            == output_path
        )
        frame = orchestrator.read_pmc_frame(tmp_path, profiling_config, options)
        assert frame["SQ_WAVES"].tolist() == [1]

    assert [call[0] for call in calls] == [
        "has",
        "materialize",
        "read",
        "has",
        "materialize",
        "read",
    ]


def write_counter_result(csv_path: Path, counter_name: str, value: str) -> None:
    csv_path.write_text(
        "Kernel_Name,Grid_Size,Workgroup_Size,LDS_Per_Workgroup,"
        "Start_Timestamp,End_Timestamp,"
        f"{counter_name}\n"
        f"kernel,64,256,0,10,20,{value}\n",
        encoding="utf-8",
    )
