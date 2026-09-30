# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import gzip
import inspect
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from threading import Thread
from typing import Set
from unittest.mock import Mock

from utils import csv_compression

ROOT = os.path.dirname(os.path.dirname(__file__))
src_candidate = os.path.join(ROOT, "src")
SRC = src_candidate if os.path.isdir(src_candidate) else ROOT
if SRC not in sys.path:
    sys.path.insert(0, SRC)

SUPPORTED_ARCHS = {
    "gfx908": {"mi100": ["MI100"]},
    "gfx90a": {"mi200": ["MI210", "MI250", "MI250X"]},
    "gfx940": {"mi300": ["MI300A_A0"]},
    "gfx941": {"mi300": ["MI300X_A0"]},
    "gfx942": {"mi300": ["MI300A_A1", "MI300X_A1"]},
    "gfx950": {"mi350": ["MI350"]},
    "gfx1150": {"rdna35_point_1": ["RDNA35_POINT_1"]},
    "gfx1151": {"rdna35_halo": ["RDNA35_HALO"]},
    "gfx1152": {"rdna35_point_2": ["RDNA35_POINT_2"]},
    "gfx1153": {"rdna35_gorgon_point": ["RDNA35_GORGON_POINT"]},
    "gfx1250": {"gfx1250_series": ["gfx1250"]},
}


ANALYSIS_CSV_HEADERS = {
    "kernel": [
        "kernel_uuid",
        "workload_id",
        "workload_name",
        "kernel_name",
        "short_name",
        "dispatch_count",
        "duration_ns_sum",
        "duration_ns_min",
        "duration_ns_max",
        "duration_ns_median",
        "duration_ns_mean",
    ],
    "kernel_metric": [
        "workload_id",
        "workload_name",
        "kernel_uuid",
        "kernel_name",
        "metric_uuid",
        "metric_name",
        "metric_id",
        "description",
        "table_name",
        "sub_table_name",
        "unit",
        "value_uuid",
        "value_name",
        "value",
    ],
    "workload_metric": [
        "workload_id",
        "workload_name",
        "metric_uuid",
        "metric_name",
        "metric_id",
        "description",
        "table_name",
        "sub_table_name",
        "unit",
        "value_uuid",
        "value_name",
        "value",
    ],
    "pc_sampling_summary": [
        "workload_id",
        "pid",
        "code_object_id",
        "kernel_uuid",
        "kernel_name",
        "offset",
        "instruction",
        "instruction_type",
        "source",
        "count",
        "issue_count",
        "stall_count",
        "wave_occupancy_percent",
        "active_thread_percent",
        "stall_reason",
    ],
    "source_lines": [
        "workload_id",
        "file_path",
        "md5_checksum",
        "line_number",
        "content",
    ],
    "roofline_ceiling": [
        "workload_id",
        "workload_name",
        "workload_sub_name",
        "device_id",
        "ceiling_kind",
        "mem_level",
        "datatype",
        "pipe",
        "pipe_label",
        "benchmark_column",
        "value",
        "unit",
    ],
    "roofline_roof": [
        "workload_id",
        "workload_name",
        "workload_sub_name",
        "device_id",
        "datatype",
        "mem_level",
        "bandwidth",
        "valu_peak",
        "matrix_peak",
        "roof_peak",
        "knee_ai",
    ],
    "kernel_roofline": [
        "workload_id",
        "workload_name",
        "workload_sub_name",
        "kernel_uuid",
        "kernel_name",
        "short_name",
        "kernel_rank",
        "dispatch_count",
        "total_duration_ns",
        "percent_runtime",
        "envelope",
        "limiter",
        "compute_ceiling",
        "compute_ceiling_label",
        "mem_level",
        "arithmetic_intensity",
        "performance",
        "roof_performance",
        "percent_of_roof",
    ],
    "kernel_roofline_metric": [
        "workload_id",
        "workload_name",
        "workload_sub_name",
        "kernel_uuid",
        "kernel_name",
        "short_name",
        "table_id",
        "metric_id",
        "metric",
        "value",
        "unit",
        "peak",
        "percent_of_peak",
    ],
}


COUNTER_RESULT_COLUMNS = (
    "GPU_ID",
    "Dispatch_ID",
    "Kernel_ID",
    "Kernel_Name",
    "Grid_Size",
    "Workgroup_Size",
    "LDS_Per_Workgroup",
    "Scratch_Per_Workitem",
    "Arch_VGPR",
    "Accum_VGPR",
    "SGPR",
    "Start_Timestamp",
    "End_Timestamp",
    "Counter_Name",
    "Counter_Value",
)


def check_resource_allocation():
    """Check if CTEST resource allocation is enabled for parallel testing and set
    HIP_VISIBLE_DEVICES variable accordingly with assigned gpu index.
    """

    if "CTEST_RESOURCE_GROUP_COUNT" not in os.environ:
        return

    if "CTEST_RESOURCE_GROUP_0_GPUS" in os.environ:
        resource = os.environ["CTEST_RESOURCE_GROUP_0_GPUS"]
        # extract assigned gpu id from env var: example format -> 'id:0,slots:1'
        for item in resource.split(","):
            key, value = item.split(":")
            if key == "id":
                os.environ["HIP_VISIBLE_DEVICES"] = value
                return

    return


def check_file_pattern(pattern, file_path):
    """Check if the given pattern exists in the file.

    Callers pass compressed counter artifacts as well as plain files such as
    sysinfo.csv and profiling_config.yaml, so the reader follows the name.
    """
    if str(file_path).endswith(".gz"):
        opener = gzip.open(file_path, "rt", encoding="utf-8")
    else:
        opener = open(file_path, encoding="utf-8")
    with opener as f:
        content = f.read()
    return len(re.findall(pattern, content)) != 0


def write_gzip_csv(path, text):
    """Write text to a gzip CSV through the interface the source uses."""
    with csv_compression.open_gzip_csv_write(path) as f:
        f.write(text)
    return Path(path)


def write_result_csv(workload_dir, text, pass_index=0):
    """Write a compressed rocpd counter artifact into workload_dir."""
    return write_gzip_csv(workload_dir / f"results_pmc_perf_{pass_index}.csv.gz", text)


def get_output_dir(suffix="_output", clean_existing=True, param_id=None):
    """
    Provides a unique output directory based on the name of the calling test function
    with a suffix applied. For parametrized tests, pass param_id to ensure unique
    directory names and avoid NFS conflicts.

    Args:
        suffix (str, optional): suffix to append to output_dir.
            Defaults to "_output".
        clean_existing (bool, optional): Whether to remove existing directory if exists.
            Defaults to True.
        param_id (str, optional): Unique identifier for parametrized tests.
            When provided, appended to the directory name to ensure uniqueness.
            Defaults to None.
    """

    func_name = inspect.stack()[1].function

    param_suffix = ""
    if param_id:
        param_suffix = "_" + re.sub(r"[^\w\-]", "_", str(param_id))

    output_dir = func_name + param_suffix + suffix
    if clean_existing:
        if Path(output_dir).exists():
            shutil.rmtree(output_dir)
    return output_dir


def clean_output_dir(cleanup, output_dir):
    """Remove output directory generated from rocprofiler-compute execution

    Args:
        cleanup (boolean): flag to enable/disable directory cleanup
        output_dir (string): name of directory to remove
    """
    if cleanup:
        if Path(output_dir).exists():
            try:
                shutil.rmtree(output_dir)
            except OSError:
                print(
                    "WARNING: shutil.rmdir(output_dir): directory may not be empty..."
                )
    return


def read_binary_file_tree(root: Path) -> dict[Path, bytes]:
    """Return binary contents keyed by each file's path relative to root."""
    return {
        file_path.relative_to(root): file_path.read_bytes()
        for file_path in root.rglob("*")
        if file_path.is_file()
    }


def _tee(pipe, sink, out) -> None:
    """Echo each line from pipe to sink while accumulating it in out."""
    with pipe:
        for line in pipe:
            print(line, end="", file=sink, flush=True)
            out.append(line)


def run_subprocess(
    command, capture_output=False, stream=False
) -> subprocess.CompletedProcess:
    """Run command in text mode and return a CompletedProcess.

    capture_output: capture stdout and stderr onto the returned object.
    stream: echo output line by line as the child produces it (requires
        capture_output); otherwise captured output is printed once at the end.
    """
    if not capture_output:
        return subprocess.run(command, text=True)

    if not stream:
        # Capture everything, then echo it in one shot after the child exits.
        process = subprocess.run(command, text=True, capture_output=True)
        if process.stdout:
            print(process.stdout, end="")
        if process.stderr:
            print(process.stderr, end="", file=sys.stderr)
        return process

    # Read each pipe on its own thread; reading serially can deadlock if one
    # fills its buffer while we block on the other.
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    out_buf, err_buf = [], []
    # Tee to the real fds, not sys.stdout/stderr, which pytest's capsys swaps
    # for in-memory buffers that never reach the terminal.
    with os.fdopen(os.dup(1), "w", closefd=True) as real_out, os.fdopen(
        os.dup(2), "w", closefd=True
    ) as real_err:
        readers = [
            Thread(target=_tee, args=(proc.stdout, real_out, out_buf)),
            Thread(target=_tee, args=(proc.stderr, real_err, err_buf)),
        ]
        for r in readers:
            r.start()
        for r in readers:
            r.join()
        proc.wait()
    return subprocess.CompletedProcess(
        command, proc.returncode, "".join(out_buf), "".join(err_buf)
    )


def patch_console(monkeypatch, module, *names, **overrides):
    """Patch ``module.console_<name>`` with a Mock for each name; return {name: Mock}.

    Pass ``name=callable`` to substitute a specific mock (e.g. a record-and-raise
    stub for the console_error exit path).
    """
    mocks = {}
    for name in names:
        mock = overrides.get(name, Mock())
        monkeypatch.setattr(f"{module}.console_{name}", mock)
        mocks[name] = mock
    return mocks


_KERNEL_SUFFIX_RE = re.compile(r"(?:\s*(?:\[clone \.[^\]]+\]|\.kd))+\s*$")


def normalize_kernel_name(name: str) -> str:
    """Return ``name`` without trailing ``.kd`` or ``[clone ...]`` suffixes."""
    return _KERNEL_SUFFIX_RE.sub("", name).strip()


def normalize_kernel_names(names: Set[str]) -> Set[str]:
    """Return the normalized form of each name in ``names``."""
    return {normalize_kernel_name(name) for name in names}


def read_counter_results(workload_dir):
    """Read all long-form counter artifacts in deterministic filename order."""
    import pandas as pd

    result_files = sorted(Path(workload_dir).glob("results_*.csv.gz"))
    assert result_files, f"No counter result files in {workload_dir}"
    return pd.concat([pd.read_csv(path) for path in result_files], ignore_index=True)


def check_counter_results(df, *, required_counters=()):
    """Validate the identities, launch fields, values and timing of counters."""
    import numpy as np
    import pandas as pd

    assert not df.empty, "Counter results are empty"
    assert set(COUNTER_RESULT_COLUMNS) <= set(df.columns), "Missing counter columns"
    for column in ("Counter_Value", "Start_Timestamp", "End_Timestamp"):
        values = pd.to_numeric(df[column], errors="coerce")
        assert np.isfinite(values).all(), f"Nonfinite or nonnumeric {column}"
    assert (
        pd.to_numeric(df.Start_Timestamp) < pd.to_numeric(df.End_Timestamp)
    ).all(), "Unordered timestamps"
    assert df.Kernel_Name.notna().all(), "Null kernel name"
    assert df.Kernel_Name.astype(str).str.strip().ne("").all(), "Blank kernel name"
    assert df.Counter_Name.notna().all(), "Null counter name"
    assert df.Counter_Name.astype(str).str.strip().ne("").all(), "Blank counter name"
    assert set(required_counters) <= set(df.Counter_Name), "Missing required counters"
    return df


def check_sysinfo(path):
    """Read and validate the hardware metadata of every profiled device."""
    import pandas as pd

    return _check_sysinfo_frame(pd.read_csv(path))


def _check_sysinfo_frame(df):
    """Validate model, architecture and positive finite hardware counts."""
    import numpy as np
    import pandas as pd

    required = {"gpu_model", "gpu_arch", "cu_per_gpu", "se_per_gpu", "simd_per_cu"}
    assert not df.empty, "Sysinfo is empty"
    assert required <= set(df.columns), "Missing sysinfo columns"
    assert df.gpu_model.notna().all(), "Null GPU model"
    assert df.gpu_model.astype(str).str.strip().ne("").all(), "Blank GPU model"
    assert df.gpu_arch.isin(SUPPORTED_ARCHS).all(), "Unsupported GPU architecture"
    for column in ("cu_per_gpu", "se_per_gpu", "simd_per_cu"):
        values = pd.to_numeric(df[column], errors="coerce")
        assert (np.isfinite(values) & (values > 0)).all(), f"Invalid {column}"
    return df


def check_analysis_db(db_path, *, expected_workloads):
    """Validate a saved analysis database and summarize workload ownership."""
    import sqlite3
    from contextlib import closing

    assert Path(db_path).is_file(), f"Missing analysis database: {db_path}"
    with closing(sqlite3.connect(db_path)) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
        assert not connection.execute("PRAGMA foreign_key_check").fetchall()
        workloads = connection.execute(
            "SELECT workload_id, name, sub_name FROM compute_workload"
        ).fetchall()
        kernels = connection.execute(
            "SELECT kernel_uuid, workload_id, kernel_name FROM compute_kernel"
        ).fetchall()
        dispatches = connection.execute(
            "SELECT kernel_uuid, start_timestamp, end_timestamp FROM compute_dispatch"
        ).fetchall()
    return _check_analysis_rows(workloads, kernels, dispatches, expected_workloads)


def _check_analysis_rows(workloads, kernels, dispatches, expected_workloads):
    """Check owners and ordered timestamps without hiding orphan rows in joins."""
    assert len(workloads) == expected_workloads, "Incorrect workload count"
    assert workloads and kernels and dispatches, "Empty analysis tables"
    summary = {
        identifier: {
            "name": name,
            "sub_name": sub_name,
            "kernels": set(),
            "kernel_uuids": set(),
            "dispatch_count": 0,
        }
        for identifier, name, sub_name in workloads
    }
    owners = {}
    for identifier, workload_id, name in kernels:
        assert workload_id in summary, "Orphan kernel owner"
        owners[identifier] = workload_id
        summary[workload_id]["kernels"].add(name)
        summary[workload_id]["kernel_uuids"].add(identifier)
    for kernel_id, start, end in dispatches:
        assert kernel_id in owners, "Orphan dispatch owner"
        assert start is not None and end is not None, "Null timestamps"
        assert start <= end, "Reversed timestamps"
        summary[owners[kernel_id]]["dispatch_count"] += 1
    assert all(row["kernels"] and row["dispatch_count"] for row in summary.values()), (
        "Empty workload data"
    )
    return summary


def check_analysis_csv_dir(directory):
    """Read every analysis CSV and validate ordered headers and composite owners."""
    import pandas as pd

    frames = {}
    for name, columns in ANALYSIS_CSV_HEADERS.items():
        path = Path(directory) / f"{name}.csv"
        assert path.is_file(), f"Missing analysis CSV: {path}"
        frames[name] = pd.read_csv(path)
        assert list(frames[name].columns) == columns, f"Incorrect {name} header"
    _check_analysis_csv_owners(frames)
    return frames


def _check_analysis_csv_owners(frames):
    """Validate all workload references and workload/kernel pairs against kernel."""
    kernel = frames["kernel"]
    assert not kernel.empty, "Empty kernel CSV"
    assert kernel[["workload_id", "kernel_uuid"]].notna().all().all()
    owners = set(zip(kernel.workload_id, kernel.kernel_uuid))
    workloads = set(kernel.workload_id)
    for name, frame in frames.items():
        if "workload_id" in frame:
            assert frame.workload_id.notna().all(), f"Null workload in {name}"
            assert set(frame.workload_id) <= workloads, f"Unknown workload in {name}"
        if {"workload_id", "kernel_uuid"} <= set(frame.columns):
            assert frame.kernel_uuid.notna().all(), f"Null kernel in {name}"
            assert set(zip(frame.workload_id, frame.kernel_uuid)) <= owners, (
                f"Mismatched workload/kernel owner in {name}"
            )
