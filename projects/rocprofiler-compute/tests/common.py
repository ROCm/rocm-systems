# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import inspect
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
src_candidate = os.path.join(ROOT, "src")
SRC = src_candidate if os.path.isdir(src_candidate) else ROOT
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from utils import rocpd_data  # noqa: E402  (sys.path adjusted above)

SUPPORTED_ARCHS = {
    "gfx908": {"mi100": ["MI100"]},
    "gfx90a": {"mi200": ["MI210", "MI250", "MI250X"]},
    "gfx940": {"mi300": ["MI300A_A0"]},
    "gfx941": {"mi300": ["MI300X_A0"]},
    "gfx942": {"mi300": ["MI300A_A1", "MI300X_A1"]},
    "gfx950": {"mi350": ["MI350"]},
    "gfx1151": {"rdna35_halo": ["RDNA35_HALO"]},
}


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
    """Check if the given pattern exists in the file"""
    content = ""
    with open(file_path) as f:
        content = f.read()
    return len(re.findall(pattern, content)) != 0


def _check_in_workload(workload_dir, db_query, db_params, csv_pattern):
    """Return True when `csv_pattern` (or `db_query`) matches workload data.

    Looks at rocpd .db files first (default profile path) and falls back
    to results_*.csv / pmc_perf*.csv (legacy csv profile path).
    """
    import sqlite3
    from contextlib import closing

    workload_path = Path(workload_dir)
    db_paths = rocpd_data.find_workload_db_paths(workload_path)
    for db_path in db_paths:
        try:
            with closing(sqlite3.connect(str(db_path))) as conn:
                cursor = conn.execute(db_query, db_params)
                if cursor.fetchone():
                    return True
        except sqlite3.DatabaseError:
            continue

    for results_file in workload_path.glob("results_*.csv"):
        if check_file_pattern(csv_pattern, str(results_file)):
            return True
    for pmc_perf in workload_path.glob("pmc_perf*.csv"):
        if check_file_pattern(csv_pattern, str(pmc_perf)):
            return True
    return False


def check_counter_in_workload(counter_name, workload_dir):
    """Return True when `counter_name` was collected in this workload."""
    return _check_in_workload(
        workload_dir,
        "SELECT 1 FROM counters_collection WHERE counter_name = ? LIMIT 1",
        (counter_name,),
        counter_name,
    )


def check_kernel_in_workload(kernel_name_substr, workload_dir):
    """Return True when a kernel matching `kernel_name_substr` was profiled."""
    return _check_in_workload(
        workload_dir,
        "SELECT 1 FROM counters_collection WHERE kernel_name LIKE ? LIMIT 1",
        (f"%{kernel_name_substr}%",),
        kernel_name_substr,
    )


def load_workload_timestamps(workload_dir):
    """Return a DataFrame with Start_Timestamp/End_Timestamp for the
    workload, reading rocpd .db (default) or the legacy results_*.csv.
    """
    import sqlite3
    from contextlib import closing

    workload_path = Path(workload_dir)
    db_paths = rocpd_data.find_workload_db_paths(workload_path)
    if db_paths:
        frames = []
        for db_path in db_paths:
            with closing(sqlite3.connect(str(db_path))) as conn:
                frames.append(
                    pd.read_sql_query(
                        "SELECT start AS Start_Timestamp, "
                        "end AS End_Timestamp "
                        "FROM counters_collection",
                        conn,
                    )
                )
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    results_files = list(workload_path.glob("results_*.csv"))
    if not results_files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in results_files], ignore_index=True)


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


def setup_workload_dir(input_dir, suffix="_tmp", clean_existing=True, param_id=None):
    """Provides a unique input workload directory with contents of input_dir
    based on the name of the calling test function. For parametrized tests,
    pass param_id to ensure unique directory names and avoid NFS conflicts.

    Creates a copy to avoid modifying source workload data.

    Args:
        input_dir (str): Source directory to copy from.
        suffix (str, optional): suffix to append to output_dir.
            Defaults to "_tmp".
        clean_existing (bool, optional): Whether to remove existing directory if exists.
            Defaults to True.
        param_id (str, optional): Unique identifier for parametrized tests.
            When provided, appended to the directory name to ensure uniqueness.
            Defaults to None.
    """

    func_name = inspect.stack()[1].function

    # Include param_id in directory name if provided
    param_suffix = ""
    if param_id:
        # Sanitize param_id: replace special chars that may not be valid in paths
        param_suffix = "_" + re.sub(r"[^\w\-]", "_", str(param_id))

    output_dir = func_name + param_suffix + suffix
    if clean_existing:
        if Path(output_dir).exists():
            shutil.rmtree(output_dir)

    shutil.copytree(input_dir, output_dir)
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


def check_csv_files(output_dir, num_devices, num_kernels):
    """Check profiling output csv files for expected
    number of entries (based on kernel invocations)

    Args:
        output_dir (string): output directory containing csv files
        num_kernels (int): number of kernels expected to have been profiled

    Returns:
        dict: dictionary housing file contents as pandas dataframe
              (excludes PMC files - those are validated internally)
    """
    files_in_workload = os.listdir(output_dir)

    has_separate = any(
        f.startswith("pmc_perf_") and f.endswith(".csv") for f in files_in_workload
    )
    has_db = bool(rocpd_data.find_workload_db_paths(output_dir))

    assert not any(
        f.startswith("results_") and f.endswith(".csv") for f in files_in_workload
    ), "results_*.csv must not be written at the workload root"

    assert has_separate or has_db, (
        "Expected pmc_perf_*.csv (csv format) or "
        "<workload>/<fbase>.db (rocpd format) from profile mode"
    )

    assert not (Path(output_dir) / "out").exists(), (
        "out/ must be removed after profiling completes"
    )

    for file in files_in_workload:
        if file.startswith("pmc_perf_") and file.endswith(".csv"):
            df = pd.read_csv(output_dir + "/" + file)
            err_msg = (
                f"PMC file {file} has insufficient rows: "
                f"{len(df.index)} < {num_kernels}"
            )
            assert len(df.index) >= num_kernels, err_msg

    # Check and return non-PMC files
    return check_non_pmc_files(output_dir, num_devices, num_kernels)


def check_non_pmc_files(output_dir, num_devices, num_kernels):
    """
    Check profiling output non-PMC files and return them as a dictionary.

    Args:
        output_dir (string): output directory containing non-PMC files
        num_devices (int): number of devices expected to have been profiled
        num_kernels (int): number of kernels expected to have been profiled

    Returns:
        dict: dictionary housing file contents as pandas dataframe
    """
    file_dict = {}
    files_in_workload = os.listdir(output_dir)

    # Load non-PMC files into return dict
    for file in files_in_workload:
        if file.endswith(".csv"):
            # Skip PMC files (already validated above)
            if file.startswith("pmc_perf_"):
                continue

            # Load other CSV files
            file_dict[file] = pd.read_csv(output_dir + "/" + file)
            if "roofline" in file:
                assert len(file_dict[file].index) >= num_devices
            elif "sysinfo" not in file and "ps_file" not in file:
                assert len(file_dict[file].index) >= num_kernels
        elif file.endswith(".html"):
            file_dict[file] = "html"
        elif file.endswith(".json"):
            file_dict[file] = "json"

    return file_dict


def get_num_pmc_file(output_dir):
    """
    Returns:
        int: number of pmc perf yaml files in perfmon dir
    """

    perfmon_path = Path(output_dir) / "perfmon"
    return len([
        f
        for f in perfmon_path.iterdir()
        if f.is_file() and f.name.startswith("pmc_perf_") and f.suffix == ".yaml"
    ])


def strip_ansi(s: str) -> str:
    ansi_escape = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")
    return ansi_escape.sub("", s)


def gpu_soc():
    """Return (arch, model) from rocminfo, e.g. ('gfx942', 'MI300').

    Both are '' when no supported GPU is detected.
    """
    # decode with utf-8 to account for rocm-smi changes in latest rocm
    rocminfo = (
        subprocess
        .run(["rocminfo"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        .stdout.decode("utf-8")
        .split("\n")
    )
    soc_regex = re.compile(r"^\s*Name\s*:\s+ ([a-zA-Z0-9]+)\s*$", re.MULTILINE)
    devices = list(filter(soc_regex.match, rocminfo))
    if not devices:
        return "", ""
    arch = devices[0].split()[1]
    if arch not in SUPPORTED_ARCHS:
        return "", ""
    model = list(SUPPORTED_ARCHS[arch].keys())[0].upper()
    return arch, model
