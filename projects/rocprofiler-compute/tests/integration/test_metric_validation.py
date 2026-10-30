# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

from pathlib import Path

import common
import pandas as pd

from tests.integration import common as integration_common


def read_workload_csv(analysis_workload_dir, csv_name):
    """Read one CSV of the workload folder, 1-<workload>, of an analyze run folder."""
    (workload_folder,) = Path(analysis_workload_dir).glob("1-*")
    return pd.read_csv(workload_folder / csv_name, dtype={"Metric_ID": str})


def get_metric_value(analysis_workload_dir, data):
    metrics_df = read_workload_csv(analysis_workload_dir, data["bw_csv"])
    metric_rows = metrics_df[metrics_df["Metric_ID"] == data["bw_metric_id"]]
    return metric_rows[data["bw_value_name"]].values[0]


def get_hbm_data_transfer(analysis_workload_dir, data):
    bw = get_metric_value(analysis_workload_dir, data)
    kernels_df = read_workload_csv(analysis_workload_dir, data["duration_csv"])
    duration_ns = kernels_df[data["duration_column"]].sum()
    # Bandwidth is stored in Bytes/s and the duration in ns, so bytes moved are
    # bw * duration_ns / 1e9 and GB moved are that divided by another 1e9.
    return bw * duration_ns / 1e18


def get_hbm_bandwidth_peak(analysis_workload_dir, data):
    # Stored in Bytes/s, while the expected values below are in GB/s.
    return get_metric_value(analysis_workload_dir, data) / 1e9


# workload -> gfx -> metric definition. CSV references point at the files of the
# workload folder that analyze writes: workload-level metric values live in
# metrics-summary.csv (one row per metric_id, one column per value_name), and
# kernel dispatch durations live in kernels.csv (Duration_ns per kernel).
VALIDATE_METRICS = {
    "memcopy": {
        "command": ["tests/memcopy"],
        "name": "HBM Data Transfer",
        "get_actual_func": get_hbm_data_transfer,
        "MI200": [
            {
                "profile_metric_id": ["4.1.8"],
                "expected_values": [4096.0],
                "tolerance": 0.10,
                "get_actual_data": {
                    "soc": "MI200",
                    "bw_csv": "metrics-summary.csv",
                    "bw_metric_id": "4.1.8",
                    "bw_value_name": "Value",
                    "duration_csv": "kernels.csv",
                    "duration_column": "Duration_ns",
                },
            },
        ],
        "MI300": [
            {
                "profile_metric_id": ["4.1.9"],
                "expected_values": [4096.0],
                "tolerance": 0.10,
                "get_actual_data": {
                    "soc": "MI300",
                    "bw_csv": "metrics-summary.csv",
                    "bw_metric_id": "4.1.9",
                    "bw_value_name": "Value",
                    "duration_csv": "kernels.csv",
                    "duration_column": "Duration_ns",
                },
            },
        ],
        "MI350": [
            {
                "profile_metric_id": ["4.1.11"],
                "expected_values": [4096.0],
                "tolerance": 0.10,
                "get_actual_data": {
                    "soc": "MI350",
                    "bw_csv": "metrics-summary.csv",
                    "bw_metric_id": "4.1.11",
                    "bw_value_name": "Value",
                    "duration_csv": "kernels.csv",
                    "duration_column": "Duration_ns",
                },
            },
        ],
        # Ignore warmup dispatch
        # Collect roofline block
        "profile_options": ["-d", "2:1001", "-b", "4"],
        "roof": True,
    },
    "hbm_bandwidth": {
        # Roofline needs an app to profile; memcopy is used only as the vehicle.
        # The validated value is the empirical HBM peak from the roofline
        # benchmark, not the memcopy kernel's achieved bandwidth.
        "command": ["tests/memcopy"],
        "name": "HBM Bandwidth (Empirical Peak)",
        "get_actual_func": get_hbm_bandwidth_peak,
        # Expected value measured on MI300X (gfx942). MI300 series may need
        # additional expected values for other SKUs (e.g. MI300A, MI308).
        # MI300X: 4916 GB/s
        # MI308X: 4225 GB/s
        "MI300": [
            {
                "profile_metric_id": ["4.1.9"],
                "expected_values": [4916.0, 4225.0],
                "tolerance": 0.10,
                "get_actual_data": {
                    "soc": "MI300",
                    "bw_csv": "metrics-summary.csv",
                    "bw_metric_id": "4.1.9",
                    "bw_value_name": "Peak (Empirical)",
                },
            },
        ],
        "profile_options": ["-d", "2:1001", "-b", "4"],
        "roof": True,
    },
}


def test_validate_metrics(
    binary_handler_profile_rocprof_compute, binary_handler_analyze_rocprof_compute, soc
):
    for workload in VALIDATE_METRICS:
        metrics = VALIDATE_METRICS[workload].get(soc, [])
        metric_ids = [mid for metric in metrics for mid in metric["profile_metric_id"]]
        if not metric_ids:
            print(
                f"Skipping metric validation for {workload} on {soc}. "
                "No metrics to validate."
            )
            continue

        profile_workload_dir = common.get_output_dir(param_id=f"{workload}_profile")
        analysis_workload_dir = common.get_output_dir(param_id=f"{workload}_analysis")
        try:
            # Copy to prevent upstream global mutations
            options = list(VALIDATE_METRICS[workload].get("profile_options", []))
            profile_config = {workload: VALIDATE_METRICS[workload]["command"]}
            _ = binary_handler_profile_rocprof_compute(
                profile_config,
                profile_workload_dir,
                options,
                check_success=True,
                roof=VALIDATE_METRICS[workload].get("roof", False),
                app_name=workload,
            )
            # Ensure non zero length of profile df
            _ = integration_common.check_csv_files(
                profile_workload_dir, num_devices=1, num_kernels=1
            )

            # Check whether metric values are correct
            code = binary_handler_analyze_rocprof_compute([
                "analyze",
                "--output-directory",
                ".",
                "--output-name",
                f"{analysis_workload_dir}",
                "-b",
                *metric_ids,
                "--path",
                profile_workload_dir,
            ])
            assert code == 0, (
                f"rocprof-compute analyze failed for {workload} on {soc} "
                f"with metric ids {metric_ids}: exit code {code}"
            )

            get_actual_func = VALIDATE_METRICS[workload]["get_actual_func"]
            for metric in metrics:
                actual = get_actual_func(
                    analysis_workload_dir, metric["get_actual_data"]
                )
                expected_values = metric["expected_values"]
                tolerance = metric["tolerance"]
                matches = [
                    abs(actual - expected) / expected <= tolerance
                    for expected in expected_values
                ]
                diffs = [(abs(actual - exp) / exp * 100) for exp in expected_values]
                assert any(matches), (
                    f"{VALIDATE_METRICS[workload]['name']}: "
                    f"actual={actual}, expected_values={expected_values}, "
                    f"diffs={diffs} (tolerance: {tolerance * 100}%)"
                )
        finally:
            common.clean_output_dir(True, analysis_workload_dir)
            common.clean_output_dir(True, profile_workload_dir)
