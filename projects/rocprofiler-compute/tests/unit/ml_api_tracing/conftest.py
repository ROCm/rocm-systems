# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Fixtures for the ML API tracing tests: trace inputs and the backend registry."""

import pandas as pd
import pytest

from ml_api_tracing.inject_roctx import backend_contract
from utils import csv_compression

TEST_GUID = "00000000-0000-0000-0000-000000000001"
TEST_GUID_OTHER = "00000000-0000-0000-0000-000000000002"
LINEAR_FORWARD_WIRE = (
    "nn.Module.Linear.forward:simple_torch_code.py:19"
    "|seqNr=n/a|tid=n/a|ftid=n/a|scope=n/a|args=()|torch"
)


@pytest.fixture
def write_ml_api_pass():
    return build_write_ml_api_pass


@pytest.fixture
def linear_marker_row():
    return build_linear_marker_row


@pytest.fixture
def counter_row():
    return build_counter_row


@pytest.fixture
def guid():
    return TEST_GUID


@pytest.fixture
def other_guid():
    return TEST_GUID_OTHER


@pytest.fixture
def fresh_registry(monkeypatch):
    """Isolate the backend registry shared by the contract and installer tests."""
    monkeypatch.setattr(backend_contract, "_REGISTRY", {})
    return backend_contract


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
