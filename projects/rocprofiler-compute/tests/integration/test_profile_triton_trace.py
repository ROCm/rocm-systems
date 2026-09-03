# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Integration tests for Triton operator tracing during profiling."""

import csv
from pathlib import Path

import common
import pandas as pd

from tests.integration.common import (
    config,
    require_triton,
)
from utils import csv_compression


def test_triton_trace_profile(
    binary_handler_profile_rocprof_compute,
    binary_handler_analyze_rocprof_compute,
    capsys,
):
    """
    Profile and analyze flow for the Triton backend.

    Profiles a Triton FFN workload with --triton-trace, verifies the marker and
    counter CSV outputs contain Triton markers, then runs analyze with
    --list-triton-operators and --triton-operator and checks the call-tree
    banner, the consolidated ml_api_trace CSV, and the matched and no-match output.
    Requires PyTorch, Triton, and a GPU.
    """
    require_triton(gpu=True)
    workload_dir = common.get_output_dir(param_id="triton_trace")

    options = [
        "--experimental",
        "--triton-trace",
        "--iteration-multiplexing",
    ]

    returncode = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=True,
        app_name="triton_test_app",
    )

    # ---- Profiling output ----

    assert returncode == 0, "Profiling the Triton application failed"

    marker_api_trace_files = list(Path(workload_dir).glob("**/*marker_api_trace.csv"))
    assert marker_api_trace_files, "No marker_api_trace.csv produced"
    assert all(
        csv_compression.resolve_csv(
            f.parent / f.name.replace("marker_api_trace", "counter_collection")
        ).is_file()
        for f in marker_api_trace_files
    ), "counter_collection CSV missing for a marker_api_trace.csv"

    found_triton_marker = False
    for marker_file in marker_api_trace_files:
        with open(marker_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames is not None, f"No columns in {marker_file}"
            assert "Function" in reader.fieldnames, (
                f"'Function' column missing in {marker_file}"
            )
            for row in reader:
                if "triton" in str(row["Function"]).lower():
                    found_triton_marker = True
                    break
        if found_triton_marker:
            break
    assert found_triton_marker, "No Triton markers in marker_api_trace output"

    # Flush profiling output so capsys captures only the analyze output.
    capsys.readouterr()

    # ---- analyze --list-triton-operators ----

    returncode_list = binary_handler_analyze_rocprof_compute([
        "--experimental",
        "analyze",
        "--path",
        workload_dir,
        "--list-triton-operators",
    ])
    assert returncode_list == 0, "Analyze with --list-triton-operators failed"

    list_output = capsys.readouterr().out
    assert "Triton Operator Call Tree:" in list_output, "Missing call-tree banner"
    # The workload launches a Triton matmul kernel.
    assert "matmul" in list_output, "matmul kernel missing from operator list"

    consolidated_csv = Path(workload_dir) / "ml_api_trace" / "consolidated.csv"
    assert consolidated_csv.exists(), "consolidated.csv not found in ml_api_trace"
    df = pd.read_csv(consolidated_csv)
    assert not df.empty, "consolidated.csv is empty"
    assert "Operator_Name" in df.columns, "Operator_Name column missing"
    assert df["Operator_Name"].astype(str).str.contains("triton").any(), (
        "No Triton operators in consolidated.csv"
    )

    # ---- analyze --triton-operator ----

    capsys.readouterr()
    returncode_match = binary_handler_analyze_rocprof_compute([
        "--experimental",
        "analyze",
        "--path",
        workload_dir,
        "--triton-operator",
        "*matmul*",
    ])
    assert returncode_match == 0, "Analyze with --triton-operator *matmul* failed"
    out_match = capsys.readouterr().out
    assert "Matched Triton Operators" in out_match, "Missing matched-operators header"
    assert "matmul" in out_match, "matmul kernel missing from matched output"

    capsys.readouterr()
    returncode_nomatch = binary_handler_analyze_rocprof_compute([
        "--experimental",
        "analyze",
        "--path",
        workload_dir,
        "--triton-operator",
        "nonexistent_kernel_xyz",
    ])
    assert returncode_nomatch == 0, (
        "Analyze with a non-matching --triton-operator failed"
    )
    out_nomatch = capsys.readouterr().out
    assert "No Triton operators matched" in out_nomatch, "Missing no-match warning"

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_ml_api_trace_torch_compile_triton(
    binary_handler_profile_rocprof_compute,
    binary_handler_analyze_rocprof_compute,
    capsys,
):
    """
    Validate the ML API trace flow for a torch.compile Triton workload.

    Profiles the workload with --ml-api-trace and runs analyze with
    --list-triton-operators, --triton-operator, and --torch-operator. Verifies
    that a Triton kernel is attributed to an operator in the consolidated
    trace. Requires PyTorch, Triton, and a GPU.
    """
    require_triton(gpu=True)
    workload_dir = common.get_output_dir(param_id="ml_api_trace")

    options = [
        "--experimental",
        "--ml-api-trace",
        "--iteration-multiplexing",
    ]

    returncode = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=True,
        app_name="torch_compile_test_app",
    )

    # ---- Profiling output ----

    assert returncode == 0, "Profiling the torch.compile/Triton workload failed"

    marker_api_trace_files = list(Path(workload_dir).glob("**/*marker_api_trace.csv"))
    assert marker_api_trace_files, "No marker_api_trace.csv produced"
    assert all(
        csv_compression.resolve_csv(
            f.parent / f.name.replace("marker_api_trace", "counter_collection")
        ).is_file()
        for f in marker_api_trace_files
    ), "counter_collection CSV missing for a marker_api_trace.csv"

    # Discard captured profiling output.
    capsys.readouterr()

    # ---- Consolidated ml_api_trace ----

    returncode_list = binary_handler_analyze_rocprof_compute([
        "--experimental",
        "analyze",
        "--path",
        workload_dir,
        "--list-triton-operators",
    ])
    assert returncode_list == 0, "Analyze with --list-triton-operators failed"
    capsys.readouterr()

    consolidated_csv = Path(workload_dir) / "ml_api_trace" / "consolidated.csv"
    assert consolidated_csv.exists(), "consolidated.csv not found in ml_api_trace"
    df = pd.read_csv(consolidated_csv)
    assert not df.empty, "consolidated.csv is empty"
    for column in ("Operator_Name", "Backend", "Kernel_Name"):
        assert column in df.columns, f"{column} column missing in consolidated.csv"

    # A Triton kernel is attributed to an operator marker from the torch or
    # triton backend.
    attributed_triton = df[
        df["Kernel_Name"].astype(str).str.contains("triton_", case=False, na=False)
        & df["Operator_Name"].notna()
        & df["Backend"].isin(["torch", "triton"])
    ]
    assert not attributed_triton.empty, (
        "No torch.compile Triton kernel (triton_*) was attributed to an "
        "operator marker in consolidated.csv"
    )

    # ---- analyze --triton-operator ----

    capsys.readouterr()
    returncode_triton = binary_handler_analyze_rocprof_compute([
        "--experimental",
        "analyze",
        "--path",
        workload_dir,
        "--triton-operator",
        "all",
    ])
    assert returncode_triton == 0, "Analyze with --triton-operator all failed"

    capsys.readouterr()
    returncode_torch = binary_handler_analyze_rocprof_compute([
        "--experimental",
        "analyze",
        "--path",
        workload_dir,
        "--torch-operator",
        "all",
    ])
    assert returncode_torch == 0, "Analyze with --torch-operator all failed"

    common.clean_output_dir(config["cleanup"], workload_dir)
