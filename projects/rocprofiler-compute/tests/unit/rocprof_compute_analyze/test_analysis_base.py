# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for src/rocprof_compute_analyze/analysis_base.py."""

import argparse
import gzip
from pathlib import Path

import common
import pandas as pd
import pytest

from rocprof_compute_analyze.analysis_base import OmniAnalyze_Base

MODULE = "rocprof_compute_analyze.analysis_base"


def test_join_workload_csvs_merges_out_artifacts(tmp_path, monkeypatch) -> None:
    common.patch_console(monkeypatch, MODULE, "debug", "warning", "log")

    pass_path = tmp_path / "out" / "pmc_perf_0"
    common.write_rocpd_pass_db(pass_path, "100")
    common.write_native_counter_csv(pass_path, "100")

    inst = OmniAnalyze_Base.__new__(OmniAnalyze_Base)
    inst.join_workload_csvs(tmp_path)

    merged = pd.read_csv(common.pmc_perf_path(tmp_path))
    assert len(merged) == 2
    assert set(merged["Counter_Name"]) == {"SQ_WAVES"}


def test_join_workload_csvs_reuses_existing_merge(tmp_path, monkeypatch) -> None:
    common.patch_console(monkeypatch, MODULE, "debug", "warning", "log")

    common.write_pmc_perf(
        tmp_path,
        "GPU_ID,Kernel_Name,Counter_Name,Counter_Value\n0,kernel_a,SQ_WAVES,10\n",
    )

    inst = OmniAnalyze_Base.__new__(OmniAnalyze_Base)
    inst.join_workload_csvs(tmp_path)

    assert pd.read_csv(common.pmc_perf_path(tmp_path))["Counter_Value"].tolist() == [10]


def test_join_workload_csvs_rebuilds_from_out_when_merge_exists(
    tmp_path, monkeypatch
) -> None:
    common.patch_console(monkeypatch, MODULE, "debug", "warning", "log")

    common.write_pmc_perf(
        tmp_path,
        "GPU_ID,Kernel_Name,Counter_Name,Counter_Value\n0,kernel_a,SQ_WAVES,99\n",
    )
    pass_path = tmp_path / "out" / "pmc_perf_0"
    common.write_rocpd_pass_db(pass_path, "100")
    common.write_native_counter_csv(pass_path, "100")

    inst = OmniAnalyze_Base.__new__(OmniAnalyze_Base)
    inst.join_workload_csvs(tmp_path)

    merged = pd.read_csv(common.pmc_perf_path(tmp_path))
    assert "SQ_WAVES" in set(merged["Counter_Name"])
    assert 99 not in merged["Counter_Value"].tolist()


def test_join_workload_csvs_rejects_legacy_results(tmp_path, monkeypatch) -> None:
    common.patch_console(monkeypatch, MODULE, "debug", "warning", "log")

    common.write_gzip_csv(
        tmp_path / "results_pmc_perf_0.csv.gz",
        "GPU_ID,Kernel_Name,Counter_Name,Counter_Value\n0,k,SQ_WAVES,1\n",
    )

    inst = OmniAnalyze_Base.__new__(OmniAnalyze_Base)
    with pytest.raises(SystemExit):
        inst.join_workload_csvs(tmp_path)

    assert not common.pmc_perf_path(tmp_path).exists()


def test_merge_profile_artifacts_errors_on_truncated_counter_csv(
    tmp_path, monkeypatch
) -> None:
    common.patch_console(monkeypatch, MODULE, "debug", "warning", "log")

    pass_path = tmp_path / "out" / "pmc_perf_0"
    (pass_path / "100").mkdir(parents=True)
    common.write_rocpd_pass_db(pass_path, "100")
    whole = gzip.compress(
        b"dispatch_id,gpu_id,kernel_id,lds_per_workgroup,"
        b"counter_id,counter_name,counter_value\n" + b"1,0,1,0,0,SQ_WAVES,5\n" * 2000
    )
    (pass_path / "100_native_counter_collection.csv.gz").write_bytes(
        whole[: len(whole) // 2]
    )

    inst = OmniAnalyze_Base.__new__(OmniAnalyze_Base)
    output = common.pmc_perf_path(tmp_path)
    with pytest.raises(SystemExit):
        inst.merge_profile_artifacts(tmp_path, output)

    assert not output.exists()


def test_sanitize_rejects_paths_sharing_a_workload_name(tmp_path, monkeypatch) -> None:
    mock_error = common.patch_console(monkeypatch, MODULE, "error")["error"]
    paths = [[str(tmp_path / parent / "vcopy" / "MI300")] for parent in ("a", "b")]
    for path in paths:
        Path(path[0]).mkdir(parents=True)

    with pytest.raises(SystemExit):
        OmniAnalyze_Base(argparse.Namespace(tui=False, path=paths), {}).sanitize()

    assert "last two components" in mock_error.call_args.args[1]
