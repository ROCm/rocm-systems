# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import inspect
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from threading import Thread
from unittest.mock import Mock

import pandas as pd
import pytest

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

    # Validate PMC data exists (profile creates pmc_perf_*.csv or results_*.csv)
    has_separate = any(
        f.startswith("pmc_perf_") and f.endswith(".csv") for f in files_in_workload
    )
    has_results = any(
        f.startswith("results_") and f.endswith(".csv") for f in files_in_workload
    )

    assert has_separate or has_results, (
        "Expected pmc_perf_*.csv or results_*.csv from profile mode"
    )

    # Validate row counts for PMC files (but don't add to return dict)
    for file in files_in_workload:
        is_pmc = file.startswith("pmc_perf_") or file.startswith("results_")
        if is_pmc and file.endswith(".csv"):
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
            if file.startswith("pmc_perf_") or file.startswith("results_"):
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


def skip_unsupported_pc_sampling_soc(is_stochastic=False):
    """Skip PC-sampling tests on SoCs that do not support the selected mode."""
    _, soc = gpu_soc()

    unsupported_socs = {"MI100", "RDNA35_POINT_1", "RDNA35_HALO", "RDNA35_POINT_2"}
    if is_stochastic:
        unsupported_socs.add("MI200")

    if soc in unsupported_socs:
        pytest.skip(f"PC sampling is not supported on {soc}")


def require_pc_sampling_gpu(is_stochastic=False):
    """Skip the test unless a GPU that supports the selected PC sampling mode is
    present."""
    _, soc = gpu_soc()
    if not soc:
        pytest.skip("GPU not supported")
    skip_unsupported_pc_sampling_soc(is_stochastic=is_stochastic)
