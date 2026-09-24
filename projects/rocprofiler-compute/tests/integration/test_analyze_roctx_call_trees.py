# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""CLI tests for ROCTX call-tree operator analyze."""

import shutil
from argparse import Namespace
from pathlib import Path

import pandas as pd

from rocprof_compute_analyze.analysis_cli import cli_analysis
from utils import csv_compression, parser, schema
from utils.utils_analysis import process_ml_api_trace_output

FIXTURE_ROOT = (
    Path(__file__).resolve().parents[1] / "workloads" / "simple_net_call_trees"
)


def copy_fixture(tmp_path, name):
    destination = tmp_path / name
    shutil.copytree(FIXTURE_ROOT / name, destination)
    return destination


def process_fixture(workload_dir):
    workload = schema.Workload()
    process_ml_api_trace_output(workload, str(workload_dir))
    return workload


def cli_for(workload_dir, workload, torch_operator=None):
    args = Namespace(torch_operator=torch_operator)
    cli = cli_analysis(args, {})
    cli._runs[str(workload_dir)] = workload
    return cli, args


def test_list_torch_operators_happy_path_collapses_passes(tmp_path, capsys):
    workload_dir = copy_fixture(tmp_path, "happy_path")
    workload = process_fixture(workload_dir)
    cli, _args = cli_for(workload_dir, workload)
    cli.list_operators(
        str(workload_dir), pd.DataFrame(columns=["Kernel_Name"]), ["torch"]
    )
    output = capsys.readouterr().out
    tree_text = output.split("Operator summary")[0]
    assert "PyTorch Operator Call Tree" in tree_text
    assert tree_text.count("nn.Module.SimpleNet.forward") == 1
    assert tree_text.count("nn.Module.Linear.forward") == 1
    assert "nn.Module.Linear.forward simple_net.py:36 (calls: 2)" in tree_text
    assert "training_loop" in tree_text


def test_torch_operator_addmm_prints_nested_path_once(tmp_path, capsys):
    workload_dir = copy_fixture(tmp_path, "happy_path")
    workload = process_fixture(workload_dir)
    workload.dfs[parser.PMC_KERNEL_TOP_TABLE_ID] = pd.DataFrame({
        "Kernel_Name": ["addmm_kernel"]
    })
    cli, args = cli_for(workload_dir, workload, torch_operator=["*addmm*"])
    cli.apply_operator_filter(args, workload, str(workload_dir), ["torch"])
    cli.handle_operator(args, workload, ["torch"])
    output = capsys.readouterr().out
    assert "aten::addmm" in output
    assert "nn.Module.Linear.forward" in output
    assert "nn.Module.SimpleNet.forward" in output
    assert workload.filter_kernel_ids == [0]


def test_backward_thread_missing_source_location(tmp_path):
    from utils.ml_api_trace_errors import MissingSourceLocationError

    workload_dir = copy_fixture(tmp_path, "backward_thread")
    workload = process_fixture(workload_dir)
    assert any(
        isinstance(err, MissingSourceLocationError)
        for err in workload.ml_api_trace_errors
    )
    assert any("AddmmBackward0" in str(err) for err in workload.ml_api_trace_errors)


def test_unaccounted_kernel_correlation_mismatch(tmp_path):
    from utils.ml_api_trace_errors import UnaccountedKernelError

    workload_dir = copy_fixture(tmp_path, "happy_path")
    counter_path = csv_compression.compressed_name(
        workload_dir / "ml_api_trace_pmc_perf_0_counter_collection.csv"
    )
    counter_df = pd.read_csv(counter_path)
    counter_df.loc[0, "Correlation_Id"] = 999999
    counter_df.to_csv(counter_path, index=False, compression="gzip")
    workload = process_fixture(workload_dir)
    assert any(
        isinstance(err, UnaccountedKernelError) for err in workload.ml_api_trace_errors
    )
    assert "unmatched dispatches" in str(workload.ml_api_trace_errors[0])


def test_pass_marker_mismatch_kernel_names(tmp_path):
    from utils.ml_api_trace_errors import PassMarkerMismatchError

    workload_dir = copy_fixture(tmp_path, "happy_path")
    counter_path = csv_compression.compressed_name(
        workload_dir / "ml_api_trace_pmc_perf_1_counter_collection.csv"
    )
    counter_df = pd.read_csv(counter_path)
    counter_df["Kernel_Name"] = "other_kernel"
    counter_df.to_csv(counter_path, index=False, compression="gzip")
    workload = process_fixture(workload_dir)
    assert any(
        isinstance(err, PassMarkerMismatchError) for err in workload.ml_api_trace_errors
    )


def test_analyze_leaves_no_consolidated_csv(tmp_path):
    workload_dir = copy_fixture(tmp_path, "happy_path")
    process_fixture(workload_dir)
    assert not (workload_dir / "ml_api_trace" / "consolidated.csv").exists()
    assert not (workload_dir / "ml_api_trace").exists()
