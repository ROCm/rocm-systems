##############################################################################
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

##############################################################################

import pandas as pd
import pytest
import test_utils

config = {}
config["memcopy"] = ["tests/memcopy"]
config["cleanup"] = True

soc = test_utils.gpu_soc()

# workload -> gfx -> metric definition
VALIDATE_METRICS = {
    "memcopy": {
        "MI200": [
            {
                "name": "HBM Bandwidth",
                "metric_id": "4.1.8",
                "csv_file": "4.1_Roofline_Performance_Rates.csv",
                "column": "Value",
                "expected_values": [1389.17],
            },
        ],
        "MI300": [
            {
                "name": "HBM Bandwidth",
                "metric_id": "4.1.9",
                "csv_file": "4.1_Roofline_Performance_Rates.csv",
                "column": "Value",
                # MI 300 series contains MI 325X GPU which
                # uses improved HBM3E instead of HBM3 used in
                # MI 300X GPU. Hence, multiple expected values
                # to cover both cases.
                # MI 308 has lower bandwidth.
                # MI 300X: 3910.62 GB/s
                # MI 325X: 4287.31 GB/s
                # MI 308: 2003.45 GB/s
                "expected_values": [2003.45, 3910.62, 4287.31],
            },
        ],
        "MI350": [
            {
                "name": "HBM Bandwidth",
                "metric_id": "4.1.10",
                "csv_file": "4.1_Roofline_Performance_Rates.csv",
                "column": "Value",
                "expected_values": [5690.42],
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
    for workload in VALIDATE_METRICS.keys():
        metrics = VALIDATE_METRICS[workload].get(soc, [])
        metric_ids = [metric["metric_id"] for metric in metrics]
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
            # Ensure non zero length of profile df
            options = VALIDATE_METRICS[workload].get("profile_options", [])
            _ = binary_handler_profile_rocprof_compute(
                config,
                profile_workload_dir,
                options,
                check_success=True,
                roof=VALIDATE_METRICS[workload].get("roof", False),
                app_name=workload,
            )
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
            assert code == 0

            for metric in metrics:
                actual = pd.read_csv(f"{analysis_workload_dir}/{metric['csv_file']}")[
                    metric["column"]
                ].values[0]
                expected_values = metric["expected_values"]
                # 5% tolerance in checking - assert if actual matches any expected value
                matches = [
                    abs(actual - expected) / expected <= 0.05
                    for expected in expected_values
                ]
                diffs = [(abs(actual - exp) / exp * 100) for exp in expected_values]
                assert any(matches), (
                    f"{metric['name']} ({metric['metric_id']}): "
                    f"actual={actual}, expected_values={expected_values}, "
                    f"diffs={diffs} (tolerance: 5%)"
                )
        finally:
            test_utils.clean_output_dir(config["cleanup"], analysis_workload_dir)
            test_utils.clean_output_dir(config["cleanup"], profile_workload_dir)
