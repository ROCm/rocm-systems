# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Shared isolated import checks and ML API trace input factories."""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from utils import csv_compression

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TEST_GUID = "00000000-0000-0000-0000-000000000001"
TEST_GUID_OTHER = "00000000-0000-0000-0000-000000000002"
LINEAR_FORWARD_WIRE = (
    "nn.Module.Linear.forward:simple_torch_code.py:19"
    "|seqNr=n/a|tid=n/a|ftid=n/a|scope=n/a|args=()|torch"
)

STDLIB_IMPORT_SCRIPT = """
import importlib
import sys
from pathlib import Path

assert sys.dont_write_bytecode, "Isolated import checks must disable bytecode writes"
project_root = Path(sys.argv[1])
module_root = project_root / "src"
if not module_root.is_dir():
    module_root = project_root
sys.path[:0] = [str(project_root / "tests"), str(module_root)]

from conftest import ProfileModeImportGuard

with ProfileModeImportGuard():
    modules_before_package = set(sys.modules)
    importlib.import_module("ml_api_tracing")
    package_imports = set(sys.modules) - modules_before_package
    assert package_imports == {"ml_api_tracing"}, (
        f"Package initializer imported unexpected modules: {sorted(package_imports)}"
    )
    module_name = sys.argv[2]
    module = importlib.import_module(module_name)
    imported_roots = {name.partition(".")[0] for name in sys.modules}
    project_imports = imported_roots & ProfileModeImportGuard.ALLOWED_PROJECT_MODULES
    assert project_imports == {"ml_api_tracing"}, (
        f"{module_name} loaded unexpected project modules: {sorted(project_imports)}"
    )
    expected_path = module_root.joinpath(*module_name.split(".")).with_suffix(".py")
    actual_path = Path(module.__file__).resolve()
    assert actual_path == expected_path.resolve(), (
        f"{module_name} loaded from {actual_path}; expected {expected_path.resolve()}"
    )
"""


def run_stdlib_import(module_name):
    import_process = subprocess.run(
        [
            sys.executable,
            "-B",
            "-I",
            "-S",
            "-c",
            STDLIB_IMPORT_SCRIPT,
            str(PROJECT_ROOT),
            module_name,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )
    assert import_process.returncode == 0, import_process.stderr


@pytest.fixture
def assert_stdlib_import():
    return run_stdlib_import


def build_write_ml_api_pass(workload_dir, pass_id, marker_rows, counter_rows):
    """Write one gzip marker/counter pair for process_ml_api_trace_output."""
    workload_dir.mkdir(parents=True, exist_ok=True)
    stem = f"ml_api_trace_pmc_perf_{pass_id}"
    marker_path = csv_compression.compressed_name(
        workload_dir / f"{stem}_marker_api_trace.csv"
    )
    counter_path = csv_compression.compressed_name(
        workload_dir / f"{stem}_counter_collection.csv"
    )
    pd.DataFrame(marker_rows).to_csv(marker_path, index=False, compression="gzip")
    pd.DataFrame(counter_rows).to_csv(counter_path, index=False, compression="gzip")


@pytest.fixture
def write_ml_api_pass():
    return build_write_ml_api_pass


def build_linear_marker_row(
    correlation_id, start, end, thread_id=1, function=None, guid=TEST_GUID
):
    return {
        "Function": LINEAR_FORWARD_WIRE if function is None else function,
        "Thread_Id": thread_id,
        "Correlation_ID": correlation_id,
        "GUID": guid,
        "Start_Timestamp": start,
        "End_Timestamp": end,
    }


@pytest.fixture
def linear_marker_row():
    return build_linear_marker_row


def build_counter_row(
    correlation_id,
    kernel_name,
    start,
    end,
    dispatch_id=1,
    counter_name="SQ_WAVES",
    guid=TEST_GUID,
):
    return {
        "Correlation_ID": correlation_id,
        "GUID": guid,
        "Kernel_Name": kernel_name,
        "Dispatch_ID": dispatch_id,
        "Grid_Size": 256,
        "Workgroup_Size": 64,
        "LDS_Per_Workgroup": 0,
        "Start_Timestamp": start,
        "End_Timestamp": end,
        "Counter_Name": counter_name,
    }


@pytest.fixture
def counter_row():
    return build_counter_row


@pytest.fixture
def test_guid():
    return TEST_GUID


@pytest.fixture
def test_guid_other():
    return TEST_GUID_OTHER
