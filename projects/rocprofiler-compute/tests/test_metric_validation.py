# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import pandas as pd
import pytest
import test_utils

_, soc = test_utils.gpu_soc()


def get_hbm_data_transfer(analysis_workload_dir, data):
    bw = pd.read_csv(f"{analysis_workload_dir}/{data['bw_csv']}")[
        data["bw_column"]
    ].values[0]
    duration_ns = pd.read_csv(f"{analysis_workload_dir}/{data['duration_csv']}")[
        data["duration_column"]
    ].values[0]
    return bw * duration_ns / 1e9


# workload -> gfx -> metric definition
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
                    "bw_csv": "4.1_Roofline_Performance_Rates.csv",
                    "bw_column": "Value",
                    "duration_csv": "0.1_Top_Kernels.csv",
                    "duration_column": "Sum(ns)",
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
                    "bw_csv": "4.1_Roofline_Performance_Rates.csv",
                    "bw_column": "Value",
                    "duration_csv": "0.1_Top_Kernels.csv",
                    "duration_column": "Sum(ns)",
                },
            },
        ],
        "MI350": [
            {
                "profile_metric_id": ["4.1.10"],
                "expected_values": [4096.0],
                "tolerance": 0.10,
                "get_actual_data": {
                    "soc": "MI350",
                    "bw_csv": "4.1_Roofline_Performance_Rates.csv",
                    "bw_column": "Value",
                    "duration_csv": "0.1_Top_Kernels.csv",
                    "duration_column": "Sum(ns)",
                },
            },
        ],
        # Ignore warmup dispatch
        # Collect roofline block
        "profile_options": ["-d", "2-1001", "-b", "4"],
        "roof": True,
    }
}


@pytest.mark.path
def test_validate_metrics(
    binary_handler_profile_rocprof_compute, binary_handler_analyze_rocprof_compute
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

        profile_workload_dir = test_utils.get_output_dir(param_id=f"{workload}_profile")
        analysis_workload_dir = test_utils.get_output_dir(
            param_id=f"{workload}_analysis"
        )
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
            _ = test_utils.check_csv_files(
                profile_workload_dir, num_devices=1, num_kernels=1
            )

            # Check whether metric values are correct
            code = binary_handler_analyze_rocprof_compute([
                "analyze",
                "--output-name",
                f"{analysis_workload_dir}",
                "--output-format",
                "csv",
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
            test_utils.clean_output_dir(True, analysis_workload_dir)
            test_utils.clean_output_dir(True, profile_workload_dir)
