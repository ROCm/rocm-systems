# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for src/rocprof_compute_analyze/analysis_base.py."""

import argparse
from pathlib import Path

import common
import pandas as pd
import pytest

from rocprof_compute_analyze.analysis_base import OmniAnalyze_Base
from tests.unit.utils.test_rocpd_data import create_rocpd_test_db
from utils import csv_compression, schema

MODULE = "rocprof_compute_analyze.analysis_base"


def _write_out_pass_db(
    tmp_path: Path, fbase: str = "pmc_perf_0", pid: str = "100"
) -> Path:
    pass_path = tmp_path / "out" / fbase / pid
    pass_path.mkdir(parents=True)
    db_path = create_rocpd_test_db(str(pass_path))
    Path(db_path).rename(pass_path / f"{pid}.db")
    return pass_path


def test_merge_profile_artifacts_writes_pmc_perf(tmp_path, monkeypatch) -> None:
    common.patch_console(monkeypatch, MODULE, "debug", "warning")

    _write_out_pass_db(tmp_path)

    inst = OmniAnalyze_Base.__new__(OmniAnalyze_Base)
    inst.merge_profile_artifacts(tmp_path, tmp_path / schema.PMC_PERF_CSV)
    merged = pd.read_csv(tmp_path / schema.PMC_PERF_CSV)

    assert "Counter_Name" in merged.columns
    assert len(merged) == 3
    assert set(merged["Counter_Name"]) == {"SQ_WAVES"}


def test_join_workload_csvs_merges_out_artifacts(tmp_path, monkeypatch) -> None:
    common.patch_console(monkeypatch, MODULE, "debug", "warning", "log")

    _write_out_pass_db(tmp_path)

    inst = OmniAnalyze_Base.__new__(OmniAnalyze_Base)
    inst.join_workload_csvs(tmp_path)

    merged = pd.read_csv(tmp_path / schema.PMC_PERF_CSV)
    assert len(merged) == 3


def test_join_workload_csvs_uses_existing_pmc_perf(tmp_path, monkeypatch) -> None:
    common.patch_console(monkeypatch, MODULE, "debug", "warning", "log")

    pmc_perf = tmp_path / schema.PMC_PERF_CSV
    with csv_compression.open_gzip_csv_write(pmc_perf) as handle:
        handle.write(
            "GPU_ID,Kernel_Name,Counter_Name,Counter_Value\n0,kernel_a,SQ_WAVES,99\n"
        )

    inst = OmniAnalyze_Base.__new__(OmniAnalyze_Base)
    inst.join_workload_csvs(tmp_path)

    assert pd.read_csv(pmc_perf)["Counter_Value"].tolist() == [99]


def test_join_workload_csvs_errors_without_artifacts(tmp_path, monkeypatch) -> None:
    common.patch_console(monkeypatch, MODULE, "debug", "warning", "log")

    inst = OmniAnalyze_Base.__new__(OmniAnalyze_Base)
    with pytest.raises(SystemExit):
        inst.join_workload_csvs(tmp_path)

    assert not (tmp_path / schema.PMC_PERF_CSV).exists()


def test_merge_profile_artifacts_errors_when_no_counter_rows(
    tmp_path, monkeypatch
) -> None:
    common.patch_console(monkeypatch, MODULE, "debug", "warning")

    pass_path = tmp_path / "out" / "pmc_perf_0" / "100"
    pass_path.mkdir(parents=True)
    (pass_path / "100.db").write_bytes(b"")

    inst = OmniAnalyze_Base.__new__(OmniAnalyze_Base)
    with pytest.raises(SystemExit):
        inst.merge_profile_artifacts(tmp_path, tmp_path / schema.PMC_PERF_CSV)

    assert not (tmp_path / schema.PMC_PERF_CSV).exists()


def test_join_workload_csvs_rejects_legacy_results(tmp_path, monkeypatch) -> None:
    common.patch_console(monkeypatch, MODULE, "debug", "warning", "log")
    import gzip

    legacy = tmp_path / "results_pmc_perf_0.csv.gz"
    with gzip.open(legacy, "wt", encoding="utf-8") as handle:
        handle.write("GPU_ID,Kernel_Name,Counter_Name,Counter_Value\n0,k,SQ_WAVES,1\n")

    inst = OmniAnalyze_Base.__new__(OmniAnalyze_Base)
    with pytest.raises(SystemExit):
        inst.join_workload_csvs(tmp_path)

    assert not (tmp_path / schema.PMC_PERF_CSV).exists()


def test_sanitize_rejects_paths_sharing_a_workload_name(tmp_path, monkeypatch) -> None:
    """Reject two paths whose last two components match."""
    mock_error = common.patch_console(monkeypatch, MODULE, "error")["error"]
    paths = [[str(tmp_path / parent / "vcopy" / "MI300")] for parent in ("a", "b")]
    for path in paths:
        Path(path[0]).mkdir(parents=True)

    # The mock records instead of exiting, so sanitize runs on to a later error.
    with pytest.raises(SystemExit):
        OmniAnalyze_Base(argparse.Namespace(tui=False, path=paths), {}).sanitize()

    assert "last two components" in mock_error.call_args.args[1]
