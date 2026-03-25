##############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

import csv
import importlib.util
import inspect
import os
import re
import socket
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import test_utils
from scipy.stats import zscore

# Runtime config options
config = {}
config["kernel_name_1"] = "vecCopy"
config["app_1"] = ["./tests/vcopy", "-n", "1048576", "-b", "256", "-i", "3"]
config["app_occupancy"] = ["./tests/occupancy"]
config["app_mat_mul_max"] = ["./tests/mat_mul_max"]
config["app_hip_dynamic_shared"] = ["./tests/hip_dynamic_shared"]
config["app_laplace_eqn"] = ["./tests/laplace_eqn", "-i", "5000"]
config["app_laplace_eqn_iter"] = ["./tests/laplace_eqn", "-i", "15000"]
config["app_laplace_eqn_insufficient"] = ["./tests/laplace_eqn", "-i", "3"]
config["app_vcopy_multikernel_iter"] = [
    "./tests/vcopy",
    "-n",
    "1048576",
    "-b",
    "256",
    "-i",
    "500",
    "--multikernel",
]
config["app_mpi_aware_laplace_eqn"] = ["./tests/mpi_aware_laplace_eqn", "-i", "5"]
config["rocflop"] = ["./tests/rocflop", "--device", "0", "--fp16"]
config["torch_test_app"] = ["python3", "./tests/simple_net.py"]
config["cleanup"] = True
config["METRIC_COMPARE"] = False
config["METRIC_LOGGING"] = False

arch_config = {}

num_kernels = 3
num_devices = 1

attach_detach_interval_msec_no_delay = 1000
attach_detach_interval_msec_with_delay = 60000
DEFAULT_ABS_DIFF = 15
DEFAULT_REL_DIFF = 50
MAX_REOCCURING_COUNT = 28

CSVS = sorted([
    "sysinfo.csv",
])

ROOF_ONLY_FILES = sorted([
    "roofline.csv",
    "sysinfo.csv",
])

METRIC_THRESHOLDS = {
    "2.1.12": {"absolute": 0, "relative": 8},
    "3.1.1": {"absolute": 0, "relative": 10},
    "3.1.10": {"absolute": 0, "relative": 10},
    "3.1.11": {"absolute": 0, "relative": 1},
    "3.1.12": {"absolute": 0, "relative": 1},
    "3.1.13": {"absolute": 0, "relative": 1},
    "5.1.0": {"absolute": 0, "relative": 15},
    "5.2.0": {"absolute": 0, "relative": 15},
    "6.1.4": {"absolute": 4, "relative": 0},
    "6.1.5": {"absolute": 0, "relative": 1},
    "6.1.0": {"absolute": 0, "relative": 15},
    "6.1.3": {"absolute": 0, "relative": 11},
    "6.2.12": {"absolute": 0, "relative": 1},
    "6.2.13": {"absolute": 0, "relative": 1},
    "7.1.0": {"absolute": 0, "relative": 1},
    "7.1.1": {"absolute": 0, "relative": 1},
    "7.1.2": {"absolute": 0, "relative": 1},
    "7.1.5": {"absolute": 0, "relative": 1},
    "7.1.6": {"absolute": 0, "relative": 1},
    "7.1.7": {"absolute": 0, "relative": 1},
    "7.2.1": {"absolute": 0, "relative": 10},
    "7.2.3": {"absolute": 0, "relative": 12},
    "7.2.6": {"absolute": 0, "relative": 1},
    "10.1.4": {"absolute": 0, "relative": 1},
    "10.1.5": {"absolute": 0, "relative": 1},
    "10.1.6": {"absolute": 0, "relative": 1},
    "10.1.7": {"absolute": 0, "relative": 1},
    "10.3.4": {"absolute": 0, "relative": 1},
    "10.3.5": {"absolute": 0, "relative": 1},
    "10.3.6": {"absolute": 0, "relative": 1},
    "11.2.1": {"absolute": 0, "relative": 1},
    "11.2.4": {"absolute": 0, "relative": 5},
    "13.2.0": {"absolute": 0, "relative": 1},
    "13.2.2": {"absolute": 0, "relative": 1},
    "14.2.0": {"absolute": 0, "relative": 1},
    "14.2.5": {"absolute": 0, "relative": 1},
    "14.2.7": {"absolute": 0, "relative": 1},
    "14.2.8": {"absolute": 0, "relative": 1},
    "15.1.4": {"absolute": 0, "relative": 1},
    "15.1.5": {"absolute": 0, "relative": 1},
    "15.1.6": {"absolute": 0, "relative": 1},
    "15.1.7": {"absolute": 0, "relative": 1},
    "15.2.4": {"absolute": 0, "relative": 1},
    "15.2.5": {"absolute": 0, "relative": 1},
    "16.1.0": {"absolute": 0, "relative": 1},
    "16.1.3": {"absolute": 0, "relative": 1},
    "16.3.0": {"absolute": 0, "relative": 1},
    "16.3.1": {"absolute": 0, "relative": 1},
    "16.3.2": {"absolute": 0, "relative": 1},
    "16.3.5": {"absolute": 0, "relative": 1},
    "16.3.6": {"absolute": 0, "relative": 1},
    "16.3.7": {"absolute": 0, "relative": 1},
    "16.3.9": {"absolute": 0, "relative": 1},
    "16.3.10": {"absolute": 0, "relative": 1},
    "16.3.11": {"absolute": 0, "relative": 1},
    "16.4.3": {"absolute": 0, "relative": 1},
    "16.4.4": {"absolute": 0, "relative": 1},
    "16.5.0": {"absolute": 0, "relative": 1},
    "17.3.3": {"absolute": 0, "relative": 1},
    "17.3.6": {"absolute": 0, "relative": 1},
    "18.1.0": {"absolute": 0, "relative": 1},
    "18.1.1": {"absolute": 0, "relative": 1},
    "18.1.2": {"absolute": 0, "relative": 1},
    "18.1.3": {"absolute": 0, "relative": 1},
    "18.1.5": {"absolute": 0, "relative": 1},
    "18.1.6": {"absolute": 1, "relative": 0},
}

# Shared constants for output directory tests.
GPU_MODEL = "MIXXX"
GPU_ARCH = "gfx000"

RANK_ENV_VARS = [
    "SLURM_PROCID",
    "FLUX_TASK_RANK",
    "PMI_RANK",
    "PMIX_RANK",
    "MPI_RANK",
    "MPI_LOCALRANKID",
    "MPI_RANKID",
    "MV2_COMM_WORLD_RANK",
    "OMPI_COMM_WORLD_RANK",
    "PALS_RANKID",
]

# check for parallel resource allocation
test_utils.check_resource_allocation()


def counter_compare(test_name, errors_pd, baseline_df, run_df, threshold=5):
    # iterate data one row at a time
    for idx_1 in run_df.index:
        run_row = run_df.iloc[idx_1]
        baseline_row = baseline_df.iloc[idx_1]
        if not run_row["KernelName"] == baseline_row["KernelName"]:
            print("Kernel/dispatch mismatch")
            assert 0
        kernel_name = run_row["KernelName"]
        gpu_id = run_row["gpu-id"]
        differences = {}

        for pmc_counter in run_row.index:
            if "Ns" in pmc_counter or "id" in pmc_counter or "[" in pmc_counter:
                # print("skipping "+pmc_counter)
                continue
                # assert 0

            if not pmc_counter in list(baseline_df.columns):
                print("error: pmc mismatch! " + pmc_counter + " is not in baseline_df")
                continue

            run_data = run_row[pmc_counter]
            baseline_data = baseline_row[pmc_counter]
            if isinstance(run_data, str) and isinstance(baseline_data, str):
                if run_data not in baseline_data:
                    print(baseline_data)
            else:
                # relative difference
                if not run_data == 0:
                    diff = round(100 * abs(baseline_data - run_data) / run_data, 2)
                    if diff > threshold:
                        print("[" + pmc_counter + "] diff is :" + str(diff) + "%")
                        if pmc_counter not in differences.keys():
                            print(
                                "[" + pmc_counter + "] not found in ",
                                list(differences.keys()),
                            )
                            differences[pmc_counter] = [diff]
                        else:
                            # Why are we here?
                            print(
                                "Why did we get here?!?!? errors_pd[idx_1]:",
                                list(differences.keys()),
                            )
                            differences[pmc_counter].append(diff)
                else:
                    # if 0 show absolute difference
                    diff = round(baseline_data - run_data, 2)
                    if diff > threshold:
                        print(
                            str(idx_1) + "[" + pmc_counter + "] diff is :" + str(diff)
                        )
        differences["kernel_name"] = [kernel_name]
        differences["test_name"] = [test_name]
        differences["gpu-id"] = [gpu_id]
        errors_pd = pd.concat([errors_pd, pd.DataFrame.from_dict(differences)])
    return errors_pd


soc = test_utils.gpu_soc()

os.environ["ROCPROF"] = "rocprofiler-sdk"

Baseline_dir = str(Path("tests/workloads/vcopy/" + soc).resolve())


def baseline_compare_metric(test_name, workload_dir, args=[]):
    t = subprocess.Popen(
        [
            sys.executable,
            "src/rocprof_compute",
            "analyze",
            "--path",
            Baseline_dir,
        ]
        + args
        + ["--path", workload_dir, "--report-diff", "-1"],
        stdout=subprocess.PIPE,
    )
    captured_output = t.communicate(timeout=1300)[0].decode("utf-8")
    print(captured_output)
    assert t.returncode == 0

    if "DEBUG ERROR" in captured_output:
        error_df = pd.DataFrame()
        if Path(Baseline_dir + "/metric_error_log.csv").exists():
            error_df = pd.read_csv(
                Baseline_dir + "/metric_error_log.csv",
                index_col=0,
            )
        output_metric_errors = re.findall(r"(\')([0-9.]*)(\')", captured_output)
        high_diff_metrics = [x[1] for x in output_metric_errors]
        for metric in high_diff_metrics:
            metric_info = re.findall(
                r"(^"
                + metric
                + (
                    r")(?: *)([()0-9A-Za-z- ]+ )"
                    r"(?: *)([0-9.-]*)"
                    r"(?: *)([0-9.-]*)"
                    r"(?: *)\(([-0-9.]*)%\)"
                    r"(?: *)([-0-9.e]*)"
                ),
                captured_output,
                flags=re.MULTILINE,
            )
            if len(metric_info):
                metric_info = metric_info[0]
                metric_idx = metric_info[0]
                metric_name = metric_info[1].strip()
                baseline_val = metric_info[-3]
                current_val = metric_info[-4]
                relative_diff = float(metric_info[-2])
                absolute_diff = float(metric_info[-1])
                if relative_diff > -99:
                    if metric_idx in METRIC_THRESHOLDS.keys():
                        # print(metric_idx+" is in FIXED_METRICS")
                        threshold_type = (
                            "absolute"
                            if METRIC_THRESHOLDS[metric_idx]["absolute"]
                            > METRIC_THRESHOLDS[metric_idx]["relative"]
                            else "relative"
                        )

                        isValid = (
                            (
                                abs(absolute_diff)
                                <= METRIC_THRESHOLDS[metric_idx]["absolute"]
                            )
                            if (threshold_type == "absolute")
                            else (
                                abs(relative_diff)
                                <= METRIC_THRESHOLDS[metric_idx]["relative"]
                            )
                        )
                        if not isValid:
                            print(
                                "index "
                                + metric_idx
                                + " "
                                + threshold_type
                                + " difference is supposed to be "
                                + str(METRIC_THRESHOLDS[metric_idx][threshold_type])
                                + ", absolute diff:",
                                absolute_diff,
                                "relative diff: ",
                                relative_diff,
                            )
                            assert 0
                        continue

                    # Used for debugging metric lists
                    if config["METRIC_LOGGING"] and (
                        (
                            abs(relative_diff) <= abs(DEFAULT_REL_DIFF)
                            or (abs(absolute_diff) <= abs(DEFAULT_ABS_DIFF))
                        )
                        and (False if baseline_val == "" else float(baseline_val) > 0)
                    ):
                        # print("logging...")
                        # print(metric_info)

                        new_error = pd.DataFrame.from_dict({
                            "Index": [metric_idx],
                            "Metric": [metric_name],
                            "Percent Difference": [relative_diff],
                            "Absolute Difference": [absolute_diff],
                            "Baseline": [baseline_val],
                            "Current": [current_val],
                            "Test Name": [test_name],
                        })
                        error_df = pd.concat([error_df, new_error])
                        counts = error_df.groupby(["Index"]).cumcount()
                        reoccurring_metrics = error_df.loc[
                            counts > MAX_REOCCURING_COUNT
                        ]
                        reoccurring_metrics["counts"] = counts[
                            counts > MAX_REOCCURING_COUNT
                        ]
                        if reoccurring_metrics.any(axis=None):
                            with pd.option_context(
                                "display.max_rows",
                                None,
                                "display.max_columns",
                                None,
                                #    'display.precision', 3,
                            ):
                                print(
                                    "These metrics appear alot\n",
                                    reoccurring_metrics,
                                )
                                # print(list(reoccurring_metrics["Index"]))

                        # log into csv
                        if not error_df.empty:
                            error_df.to_csv(Baseline_dir + "/metric_error_log.csv")


def validate(test_name, workload_dir, file_dict, args=[]):
    if config["METRIC_COMPARE"]:
        baseline_compare_metric(test_name, workload_dir, args)


def are_stochastic_counters_similar(test_dfs, baseline_df):
    """
    Compares multiple test dataframes against a baseline dataframe to check
    if the stochastic counter values are similar. Returns True if all test dataframes
    have similar counter values to the baseline, otherwise returns False.
    """
    group_labels = [
        "Kernel_Name",
        "Grid_Size",
        "Workgroup_Size",
        "LDS_Per_Workgroup",
        "Counter_Name",
    ]

    baseline_grouped = baseline_df.groupby(group_labels)
    tests_grouped = [df.groupby(group_labels) for df in test_dfs]

    baseline_group_keys = set(baseline_grouped.groups.keys())
    tests_group_keys = [set(group.groups.keys()) for group in tests_grouped]

    # Check if all test dataframes have the same group keys as the baseline
    if not all(baseline_group_keys == keys for keys in tests_group_keys):
        return False

    stochastic_counter_patterns = list(
        map(
            re.compile,
            [
                ".*REQ_sum$",
                ".*REQ_.*_sum$",
                ".*READ_sum$",
                ".*WRITE_sum$",
            ],
        )
    )

    for group_key, baseline_group in baseline_grouped:
        test_groups = [
            test_grouped.get_group(group_key) for test_grouped in tests_grouped
        ]

        baseline_counters = baseline_group["Counter_Value"]
        test_counters_list = [test_group["Counter_Value"] for test_group in test_groups]

        counter_name = group_key[4]

        # Warmup values aren't ignored as they do not significantly impact
        # the analysis for stochastic counters and leaves too few data points
        # for baseline.
        if any(
            re.match(pattern, counter_name) for pattern in stochastic_counter_patterns
        ):
            # Remove outliers using Z-score method
            z_score_threshold = 2.0

            test_z_scores_list = [
                np.abs(zscore(test_counters)) for test_counters in test_counters_list
            ]
            test_counters_list_trimmed = [
                test_counters[test_z_scores < z_score_threshold]
                for test_counters, test_z_scores in zip(
                    test_counters_list, test_z_scores_list
                )
            ]

            baseline_mean = baseline_counters.mean()
            baseline_std = baseline_counters.std()
            upper_bound = baseline_mean + 3 * baseline_std
            lower_bound = baseline_mean - 3 * baseline_std

            for test_counters in test_counters_list_trimmed:
                if test_counters.between(lower_bound, upper_bound).all() is False:
                    return False

    return True


def are_deterministic_counters_equal(test_dfs, baseline_df):
    """
    Compares multiple test dataframes against a baseline dataframe to check
    if the deterministic counter values are equal. Returns True if all test dataframes
    have equal counter values to the baseline, otherwise returns False.
    """
    group_labels = [
        "Kernel_Name",
        "Grid_Size",
        "Workgroup_Size",
        "LDS_Per_Workgroup",
        "Counter_Name",
    ]

    baseline_grouped = baseline_df.groupby(group_labels)
    tests_grouped = [df.groupby(group_labels) for df in test_dfs]

    baseline_group_keys = set(baseline_grouped.groups.keys())
    tests_group_keys = [set(group.groups.keys()) for group in tests_grouped]

    # Check if all test dataframes have the same group keys as the baseline
    if not all(baseline_group_keys == keys for keys in tests_group_keys):
        return False

    # series prior to MI350 use CSN, MI350 uses CS{0,1,2,3}
    deterministic_counter_patterns = list(
        map(
            re.compile,
            [
                "SQ_INSTS_.*",
                "SPI_CS\\d_NUM_THREADGROUPS",
                "SPI_CSN_NUM_THREADGROUPS",
                "SPI_CS\\d_WAVE",
                "SPI_CSN_WAVE",
                "SQ_WAVES",
            ],
        )
    )

    for group_key, baseline_group in baseline_grouped:
        test_groups = [
            test_grouped.get_group(group_key) for test_grouped in tests_grouped
        ]

        baseline_counters = baseline_group["Counter_Value"]
        test_counters_list = [test_group["Counter_Value"] for test_group in test_groups]

        counter_name = group_key[4]
        if any(
            re.match(pattern, counter_name)
            for pattern in deterministic_counter_patterns
        ):
            if (
                all([
                    test_counters.unique().size == 1
                    for test_counters in test_counters_list
                ])
                and baseline_counters.unique().size == 1
                and all([
                    test_counters.values[0] == baseline_counters.values[0]
                    for test_counters in test_counters_list
                ])
            ):
                continue

            return False

    return True


# --
# Shared mocks and helpers for output directory tests
# --


class MockProfiler:
    """Mock profiler used by output directory tests."""

    def __init__(self, *args, **kwargs):
        pass

    def run_profiling(self, *args, **kwargs):
        pass

    def sanitize(self, *args, **kwargs):
        pass

    def pre_processing(self, *args, **kwargs):
        pass

    def post_processing(self, *args, **kwargs):
        pass


class MockMachineSpecs:
    def __init__(self, model, arch):
        self.gpu_model = model
        self.gpu_arch = arch


class MockSoc:
    def post_profiling(self, *args, **kwargs):
        pass


def mock_generate_machine_specs(self):
    """Set mock machine specs so %gpumodel% resolves before load_soc_specs runs."""
    self._RocProfCompute__mspec = MockMachineSpecs(GPU_MODEL, GPU_ARCH)


def mock_load_soc_specs(self, sysinfo=None):
    self._RocProfCompute__mspec = MockMachineSpecs(GPU_MODEL, GPU_ARCH)
    self._RocProfCompute__soc[GPU_ARCH] = MockSoc()


def clear_rank_env(monkeypatch):
    """Remove all known MPI rank environment variables."""
    for key in RANK_ENV_VARS:
        monkeypatch.delenv(key, raising=False)


# --
# Start of profiling tests
# --


@pytest.mark.path
def test_path(binary_handler_profile_rocprof_compute):
    workload_dir = test_utils.get_output_dir()
    binary_handler_profile_rocprof_compute(config, workload_dir)

    file_dict = test_utils.check_csv_files(workload_dir, num_devices, num_kernels)

    if soc == "MI100":
        assert sorted(list(file_dict.keys())) == CSVS
    elif soc == "MI200":
        assert sorted(list(file_dict.keys())) == CSVS
    elif "MI300" in soc:
        assert sorted(list(file_dict.keys())) == CSVS
    elif "MI350" in soc:
        assert sorted(list(file_dict.keys())) == CSVS
    else:
        print(f"This test is not supported for {soc}")
        assert 0

    validate(inspect.stack()[0][3], workload_dir, file_dict)

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.path
def test_path_rocflop(binary_handler_profile_rocprof_compute):
    # Test whether multiprocess workloads like rocflop are handled correctly
    workload_dir = test_utils.get_output_dir()
    options = ["--block", "2.1.1"]
    _ = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=True,
        roof=False,
        app_name="rocflop",
    )
    test_utils.check_csv_files(workload_dir, num_devices, num_kernels)
    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.path
def test_path_no_native(binary_handler_profile_rocprof_compute):
    workload_dir = test_utils.get_output_dir()
    options = ["--no-native-tool"]
    binary_handler_profile_rocprof_compute(config, workload_dir, options)

    file_dict = test_utils.check_csv_files(workload_dir, num_devices, num_kernels)

    if soc == "MI100":
        assert sorted(list(file_dict.keys())) == CSVS
    elif soc == "MI200":
        assert sorted(list(file_dict.keys())) == CSVS
    elif "MI300" in soc:
        assert sorted(list(file_dict.keys())) == CSVS
    elif "MI350" in soc:
        assert sorted(list(file_dict.keys())) == CSVS
    else:
        print(f"This test is not supported for {soc}")
        assert 0

    validate(inspect.stack()[0][3], workload_dir, file_dict)

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.path
def test_path_rocpd(
    binary_handler_profile_rocprof_compute, binary_handler_analyze_rocprof_compute
):
    workload_dir = test_utils.get_output_dir()
    options = ["--format-rocprof-output", "rocpd"]
    binary_handler_profile_rocprof_compute(config, workload_dir, options)

    # Validate profile outputs (results_*.csv for rocpd format)
    test_utils.check_csv_files(workload_dir, num_devices, num_kernels)
    assert test_utils.check_file_pattern(
        "format_rocprof_output: rocpd", f"{workload_dir}/profiling_config.yaml"
    )

    # Run analyze to create merged pmc_perf.csv
    code = binary_handler_analyze_rocprof_compute(["analyze", "--path", workload_dir])
    assert code == 0

    # Validate merged pmc_perf.csv content
    assert test_utils.check_file_pattern("Counter_Name", f"{workload_dir}/pmc_perf.csv")

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.path
def test_path_csv(
    binary_handler_profile_rocprof_compute, binary_handler_analyze_rocprof_compute
):
    workload_dir = test_utils.get_output_dir()
    options = ["--format-rocprof-output", "csv"]
    binary_handler_profile_rocprof_compute(config, workload_dir, options)

    file_dict = test_utils.check_csv_files(workload_dir, num_devices, num_kernels)
    all_csvs_mi100 = sorted([
        "SQC_DCACHE_INFLIGHT_LEVEL.csv",
        "SQC_ICACHE_INFLIGHT_LEVEL.csv",
        "SQ_IFETCH_LEVEL.csv",
        "SQ_INST_LEVEL_LDS.csv",
        "SQ_LEVEL_WAVES.csv",
        "sysinfo.csv",
    ])
    all_csvs_mi200 = sorted([
        "SQC_DCACHE_INFLIGHT_LEVEL.csv",
        "SQC_ICACHE_INFLIGHT_LEVEL.csv",
        "SQ_IFETCH_LEVEL.csv",
        "SQ_INST_LEVEL_LDS.csv",
        "SQ_INST_LEVEL_SMEM.csv",
        "SQ_INST_LEVEL_VMEM.csv",
        "SQ_LEVEL_WAVES.csv",
        "sysinfo.csv",
    ])
    all_csvs_mi300 = sorted([
        "SQC_DCACHE_INFLIGHT_LEVEL.csv",
        "SQC_ICACHE_INFLIGHT_LEVEL.csv",
        "SQ_IFETCH_LEVEL.csv",
        "SQ_INST_LEVEL_LDS.csv",
        "SQ_INST_LEVEL_SMEM.csv",
        "SQ_INST_LEVEL_VMEM.csv",
        "SQ_LEVEL_WAVES.csv",
        "sysinfo.csv",
    ])
    all_csvs_mi350 = sorted([
        "SQC_DCACHE_INFLIGHT_LEVEL.csv",
        "SQC_ICACHE_INFLIGHT_LEVEL.csv",
        "SQ_IFETCH_LEVEL.csv",
        "SQ_INST_LEVEL_LDS.csv",
        "SQ_INST_LEVEL_SMEM.csv",
        "SQ_INST_LEVEL_VMEM.csv",
        "SQ_LEVEL_WAVES.csv",
        "sysinfo.csv",
    ])

    if soc == "MI100":
        assert sorted(list(file_dict.keys())) == all_csvs_mi100
    elif soc == "MI200":
        assert sorted(list(file_dict.keys())) == all_csvs_mi200
    elif "MI300" in soc:
        assert sorted(list(file_dict.keys())) == all_csvs_mi300
    elif "MI350" in soc:
        assert sorted(list(file_dict.keys())) == all_csvs_mi350
    else:
        print(f"This test is not supported for {soc}")
        assert 0

    validate(inspect.stack()[0][3], workload_dir, file_dict)

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.path
def test_output_directory_hostname(binary_handler_profile_rocprof_compute, monkeypatch):
    """Test that %hostname% placeholder is replaced with the actual hostname."""
    from rocprof_compute_base import RocProfCompute

    hostname = "test_node"

    monkeypatch.setattr(RocProfCompute, "create_profiler", lambda self: MockProfiler())
    monkeypatch.setattr(socket, "gethostname", lambda: hostname)

    workload_base_dir = test_utils.get_output_dir(param_id="hostname")
    workload_dir = os.path.join(workload_base_dir, "%hostname%")

    binary_handler_profile_rocprof_compute(config, workload_dir)

    workload_dir = workload_dir.replace("%hostname%", hostname)
    assert os.path.exists(workload_dir)

    test_utils.clean_output_dir(config["cleanup"], workload_base_dir)


@pytest.mark.path
def test_output_directory_gpumodel(binary_handler_profile_rocprof_compute, monkeypatch):
    """Test that %gpumodel% placeholder is replaced with the GPU model name."""
    from rocprof_compute_base import RocProfCompute

    monkeypatch.setattr(RocProfCompute, "create_profiler", lambda self: MockProfiler())
    monkeypatch.setattr(
        RocProfCompute, "generate_machine_specs", mock_generate_machine_specs
    )
    monkeypatch.setattr(RocProfCompute, "load_soc_specs", mock_load_soc_specs)

    workload_base_dir = test_utils.get_output_dir(param_id="gpumodel")
    workload_dir = os.path.join(workload_base_dir, "%gpumodel%_output")

    binary_handler_profile_rocprof_compute(config, workload_dir)

    workload_dir = workload_dir.replace("%gpumodel%", GPU_MODEL)
    assert os.path.exists(workload_dir)

    test_utils.clean_output_dir(config["cleanup"], workload_base_dir)


@pytest.mark.path
def test_output_directory_rank_ignored_without_mpi(
    binary_handler_profile_rocprof_compute, monkeypatch
):
    """Test that %rank% is ignored when no MPI rank env var is set."""
    from rocprof_compute_base import RocProfCompute

    clear_rank_env(monkeypatch)
    monkeypatch.setattr(RocProfCompute, "create_profiler", lambda self: MockProfiler())

    workload_base_dir = test_utils.get_output_dir(param_id="no_rank")
    workload_dir = os.path.join(workload_base_dir, "%rank%_output")

    binary_handler_profile_rocprof_compute(config, workload_dir)

    workload_dir = workload_dir.replace("%rank%", "")
    assert os.path.exists(workload_dir)

    test_utils.clean_output_dir(config["cleanup"], workload_base_dir)


@pytest.mark.path
def test_output_directory_rank_replaced_with_mpi(
    binary_handler_profile_rocprof_compute, monkeypatch
):
    """Test that %rank% is replaced with the rank value for each MPI env var."""
    from rocprof_compute_base import RocProfCompute

    clear_rank_env(monkeypatch)
    rank = "3"

    monkeypatch.setattr(RocProfCompute, "create_profiler", lambda self: MockProfiler())

    for key in RANK_ENV_VARS:
        monkeypatch.setenv(key, rank)

        workload_base_dir = test_utils.get_output_dir(param_id=f"rank_env_{key}")
        workload_dir = os.path.join(workload_base_dir, "%rank%_output")

        binary_handler_profile_rocprof_compute(config, workload_dir)

        workload_dir = workload_dir.replace("%rank%", rank)
        assert os.path.exists(workload_dir)

        test_utils.clean_output_dir(config["cleanup"], workload_base_dir)
        monkeypatch.delenv(key, raising=False)


@pytest.mark.path
def test_output_directory_env_variable(
    binary_handler_profile_rocprof_compute, monkeypatch
):
    """Test that %env{VAR}% is replaced with the environment variable value."""
    from rocprof_compute_base import RocProfCompute

    monkeypatch.setenv("ENV_1", "custom_env")
    monkeypatch.setattr(RocProfCompute, "create_profiler", lambda self: MockProfiler())

    workload_base_dir = test_utils.get_output_dir(param_id="env")
    workload_dir = os.path.join(workload_base_dir, "%env{ENV_1}%")

    binary_handler_profile_rocprof_compute(config, workload_dir)

    workload_dir = workload_dir.replace("%env{ENV_1}%", "custom_env")
    assert os.path.exists(workload_dir)

    test_utils.clean_output_dir(config["cleanup"], workload_base_dir)
    monkeypatch.delenv("ENV_1", raising=False)


@pytest.mark.path
def test_output_directory_env_variable_unset(
    binary_handler_profile_rocprof_compute, monkeypatch
):
    """Test that %env{VAR}% resolves to empty string when the var is unset."""
    from rocprof_compute_base import RocProfCompute

    monkeypatch.delenv("ENV_2", raising=False)
    monkeypatch.setattr(RocProfCompute, "create_profiler", lambda self: MockProfiler())

    workload_base_dir = test_utils.get_output_dir(param_id="no_env")
    workload_dir = os.path.join(workload_base_dir, "%env{ENV_2}%")

    binary_handler_profile_rocprof_compute(config, workload_dir)
    workload_dir = workload_dir.replace("%env{ENV_2}%", "")

    assert os.path.exists(workload_dir)
    test_utils.clean_output_dir(config["cleanup"], workload_base_dir)


@pytest.mark.path
def test_output_directory_all_placeholders_combined(
    binary_handler_profile_rocprof_compute, monkeypatch
):
    """Test that all placeholders work together in a single path."""
    from rocprof_compute_base import RocProfCompute

    hostname = "test_node"
    rank = "3"

    monkeypatch.setattr(RocProfCompute, "create_profiler", lambda self: MockProfiler())
    monkeypatch.setattr(socket, "gethostname", lambda: hostname)
    monkeypatch.setattr(
        RocProfCompute, "generate_machine_specs", mock_generate_machine_specs
    )
    monkeypatch.setattr(RocProfCompute, "load_soc_specs", mock_load_soc_specs)
    monkeypatch.setenv("ENV_1", "custom_env")
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", rank)

    workload_base_dir = test_utils.get_output_dir(param_id="host_gpu_env_rank")
    workload_dir = os.path.join(
        workload_base_dir,
        "%hostname%_%gpumodel%_%env{ENV_1}%_%rank%_output",
    )

    binary_handler_profile_rocprof_compute(config, workload_dir)

    workload_dir = (
        workload_dir
        .replace("%hostname%", hostname)
        .replace("%gpumodel%", GPU_MODEL)
        .replace("%env{ENV_1}%", "custom_env")
        .replace("%rank%", rank)
    )
    assert os.path.exists(workload_dir)

    test_utils.clean_output_dir(config["cleanup"], workload_base_dir)
    monkeypatch.delenv("OMPI_COMM_WORLD_RANK", raising=False)
    monkeypatch.delenv("ENV_1", raising=False)


@pytest.mark.path
def test_output_directory_default_with_rank(
    binary_handler_profile_rocprof_compute, monkeypatch
):
    """
    Test that rank is appended to the default output
    directory when MPI rank is set.
    """
    from rocprof_compute_base import RocProfCompute

    rank = "3"
    original_cwd = os.getcwd()

    monkeypatch.setattr(RocProfCompute, "create_profiler", lambda self: MockProfiler())
    monkeypatch.setattr(
        RocProfCompute, "generate_machine_specs", mock_generate_machine_specs
    )
    monkeypatch.setattr(RocProfCompute, "load_soc_specs", mock_load_soc_specs)
    monkeypatch.setenv("PMI_RANK", rank)

    workload_base_dir = test_utils.get_output_dir(param_id="rank_def_dir")
    p = Path(workload_base_dir)
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
    os.chdir(workload_base_dir)

    binary_handler_profile_rocprof_compute(
        config, workload_dir=workload_base_dir, workload_dir_type="default"
    )

    workload_dir = os.path.join(
        workload_base_dir,
        "workloads",
        "app_1",
        rank,
    )

    os.chdir(original_cwd)

    assert os.path.exists(workload_dir)

    test_utils.clean_output_dir(config["cleanup"], workload_base_dir)
    monkeypatch.delenv("PMI_RANK", raising=False)


@pytest.mark.path
def test_output_directory_default_without_rank(
    binary_handler_profile_rocprof_compute, monkeypatch
):
    """Test default output directory layout when no MPI rank is set."""
    from rocprof_compute_base import RocProfCompute

    clear_rank_env(monkeypatch)
    original_cwd = os.getcwd()

    monkeypatch.setattr(RocProfCompute, "create_profiler", lambda self: MockProfiler())
    monkeypatch.setattr(
        RocProfCompute, "generate_machine_specs", mock_generate_machine_specs
    )
    monkeypatch.setattr(RocProfCompute, "load_soc_specs", mock_load_soc_specs)

    workload_base_dir = test_utils.get_output_dir(param_id="no_rank_def_dir")
    p = Path(workload_base_dir)
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
    os.chdir(workload_base_dir)

    binary_handler_profile_rocprof_compute(
        config, workload_dir=workload_base_dir, workload_dir_type="default"
    )

    os.chdir(original_cwd)

    workload_dir = os.path.join(
        workload_base_dir,
        "workloads",
        "app_1",
        GPU_MODEL,
    )
    assert os.path.exists(workload_dir)

    test_utils.clean_output_dir(config["cleanup"], workload_base_dir)


@pytest.mark.path
def test_output_directory_no_name_with_output_dir(
    binary_handler_profile_rocprof_compute, monkeypatch
):
    """Test that --output-directory works without --name."""
    from rocprof_compute_base import RocProfCompute

    monkeypatch.setattr(RocProfCompute, "create_profiler", lambda self: MockProfiler())
    monkeypatch.setattr(
        RocProfCompute, "generate_machine_specs", mock_generate_machine_specs
    )
    monkeypatch.setattr(RocProfCompute, "load_soc_specs", mock_load_soc_specs)

    workload_dir = test_utils.get_output_dir(param_id="dir_no_name")

    binary_handler_profile_rocprof_compute(
        config, workload_dir=workload_dir, skip_app_name=True
    )

    assert os.path.exists(workload_dir)

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.path
def test_output_directory_no_name_no_output_dir(
    binary_handler_profile_rocprof_compute, monkeypatch
):
    """Test that profiling fails when neither --name nor --output-directory is given."""
    from rocprof_compute_base import RocProfCompute

    monkeypatch.setattr(RocProfCompute, "create_profiler", lambda self: MockProfiler())
    monkeypatch.setattr(
        RocProfCompute, "generate_machine_specs", mock_generate_machine_specs
    )
    monkeypatch.setattr(RocProfCompute, "load_soc_specs", mock_load_soc_specs)

    workload_dir = test_utils.get_output_dir(param_id="no_name_no_dir")

    error_code = binary_handler_profile_rocprof_compute(
        config,
        skip_app_name=True,
        workload_dir=workload_dir,
        check_success=False,
        workload_dir_type="default",
    )

    assert error_code == 1

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.roofline_1
def test_roof_basic_validation(binary_handler_profile_rocprof_compute):
    """
    Test basic roofline CSV generation in profile mode.
    Validates that roofline.csv is generated via microbenchmarks.
    """
    if soc in ("MI100"):
        # roofline is not supported on MI100
        assert True
        # Do not continue testing
        return

    options = ["--device", "0", "--roof-only"]
    workload_dir = test_utils.get_output_dir()
    returncode = binary_handler_profile_rocprof_compute(
        config, workload_dir, options, check_success=False, roof=True
    )

    assert returncode == 0
    file_dict = test_utils.check_csv_files(workload_dir, 1, num_kernels)

    assert sorted(list(file_dict.keys())) == ROOF_ONLY_FILES

    validate(
        inspect.stack()[0][3],
        workload_dir,
        file_dict,
    )

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.roofline_1
def test_roof_file_validation(binary_handler_profile_rocprof_compute):
    """Test file validation paths in roofline"""
    if soc in ("MI100"):
        pytest.skip("Roofline not supported on MI100")
        return

    options = ["--device", "0", "--roof-only"]
    workload_dir = test_utils.get_output_dir()

    try:
        returncode = binary_handler_profile_rocprof_compute(
            config, workload_dir, options, check_success=False, roof=True
        )

        if returncode == 0:
            roofline_csv = f"{workload_dir}/roofline.csv"
            if os.path.exists(roofline_csv):
                import pandas as pd

                df = pd.read_csv(roofline_csv)
                assert len(df) >= 0

    finally:
        test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.roofline_1
def test_roof_rocpd(
    binary_handler_profile_rocprof_compute,
    binary_handler_analyze_rocprof_compute,
):
    if soc == "MI100":
        pytest.skip("Roofline not supported on MI100")
        return

    workload_dir = test_utils.get_output_dir()
    options = ["--device", "0", "--roof-only", "--format-rocprof-output", "rocpd"]
    binary_handler_profile_rocprof_compute(config, workload_dir, options, roof=True)

    # Validate profile outputs
    test_utils.check_csv_files(workload_dir, num_devices, num_kernels)
    assert (Path(workload_dir) / "roofline.csv").exists()
    assert test_utils.check_file_pattern(
        "format_rocprof_output: rocpd", f"{workload_dir}/profiling_config.yaml"
    )

    # Run analyze to create merged pmc_perf.csv
    code = binary_handler_analyze_rocprof_compute(["analyze", "--path", workload_dir])
    assert code == 0

    # Validate merged pmc_perf.csv content
    assert test_utils.check_file_pattern("Counter_Name", f"{workload_dir}/pmc_perf.csv")

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.misc
def test_analyze_rocpd(
    binary_handler_profile_rocprof_compute, binary_handler_analyze_rocprof_compute
):
    workload_dir = test_utils.get_output_dir()
    options = ["--device", "0", "--format-rocprof-output", "rocpd"]
    binary_handler_profile_rocprof_compute(config, workload_dir, options, roof=True)

    db_name = "test"
    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--output-format",
        "db",
        "--output-name",
        f"{db_name}",
        "--path",
        workload_dir,
    ])
    assert code == 0
    assert os.path.isfile(f"{db_name}.db")

    # Open the sqlite database and assert the schema
    # Import Kernel from analysis_orm.py
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from utils.analysis_orm import (
        Dispatch,
        Kernel,
        KernelMetricValue,
        KernelRooflineData,
        Metadata,
        MetricDefinition,
        Workload,
        WorkloadMetricValue,
        WorkloadRooflineData,
    )

    table_name_map = {
        "compute_workload": Workload,
        "compute_metric_definition": MetricDefinition,
        "compute_kernel_roofline_data": KernelRooflineData,
        "compute_workload_roofline_data": WorkloadRooflineData,
        "compute_dispatch": Dispatch,
        "compute_kernel": Kernel,
        "compute_kernel_metric_value": KernelMetricValue,
        "compute_workload_metric_value": WorkloadMetricValue,
        "compute_metadata": Metadata,
    }

    def check_cols(table_name, orm_obj):
        conn = sqlite3.connect(f"{db_name}.db")
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info('{table_name}');")
        columns = cursor.fetchall()
        column_names = [column[1] for column in columns]
        expected_columns = [col.name for col in orm_obj.__table__.columns]
        assert column_names == expected_columns
        conn.close()

    for table_name, orm_obj in table_name_map.items():
        check_cols(table_name, orm_obj)

    os.remove(f"{db_name}.db")
    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.roofline_1
def test_roofline_workload_dir_not_set_error():
    """
    Test roof_setup() error: "Workload directory is not set. Cannot perform setup."
    This covers lines 113-117
    """
    if soc in ("MI100"):
        pytest.skip("Skipping roofline test for MI100")
        return

    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    try:
        from roofline import Roofline
        from utils.specs import generate_machine_specs

        class MockArgs:
            def __init__(self):
                self.roof_only = True
                self.mem_level = "ALL"
                self.sort = "ALL"
                self.roofline_data_type = ["FP32"]

        args = MockArgs()
        mspec = generate_machine_specs(None, None)

        run_parameters = {
            "workload_dir": None,
            "device_id": 0,
            "sort_type": "kernels",
            "mem_level": "ALL",
            "is_standalone": True,
            "roofline_data_type": ["FP32"],
        }

        roofline_instance = Roofline(args, mspec, run_parameters)

        import contextlib
        from io import StringIO

        captured_output = StringIO()

        with contextlib.redirect_stderr(captured_output):
            try:
                roofline_instance.roof_setup()
            except SystemExit:
                pass

        assert True

    except ImportError:
        pytest.skip("Could not import roofline module for direct testing")


@pytest.mark.roofline_1
def test_roof_workload_dir_validation(binary_handler_profile_rocprof_compute):
    if soc in ("MI100"):
        assert True
        return

    options = ["--device", "0", "--roof-only"]

    workload_dir = test_utils.get_output_dir()
    returncode = binary_handler_profile_rocprof_compute(
        config, workload_dir, options, check_success=False, roof=True
    )
    assert returncode == 0

    nested_dir = os.path.join(workload_dir, "nested", "structure")
    os.makedirs(nested_dir, exist_ok=True)
    returncode = binary_handler_profile_rocprof_compute(
        config, nested_dir, options, check_success=False, roof=True
    )
    assert returncode == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.roofline_1
def test_roofline_kernel_filter(binary_handler_profile_rocprof_compute):
    """
    Test roofline multi-attempt profiling with `--kernel`
    Expect to be able to re-profile from same workload if kernels are valid.

    Roofline now takes in a dataframe that should already have filtering applied.
    Any invald kernels should be handled prior to roof activity.
    Check the following cases:
    - no valid kernels
    - one valid kernel
    - 2 kernels, one valid and one invalid
    """
    if soc in ("MI100"):
        pytest.skip("Skipping roofline test for MI100")
        return

    options = [
        "--device",
        "0",
        "--roof-only",
    ]
    workload_dir = test_utils.get_output_dir()

    returncode = binary_handler_profile_rocprof_compute(  # noqa: F841
        config, workload_dir, options, check_success=True, roof=True
    )
    # Don't clean output dir, use same workload
    # Test only non-existent kernel: result should be passing
    # Dataframe given to roofline should just be all available kernels with no filtering
    options_bad = options.copy()
    options_bad.extend([
        "--kernel",
        "nonexistent_kernel_name_that_should_not_match_anything",
    ])
    returncode = binary_handler_profile_rocprof_compute(  # noqa: F841
        config,
        workload_dir,
        options_bad,
        check_success=True,
        roof=True,
    )
    assert returncode == 0

    # Test one good kernel using existing profiling data
    # Result should be passing as usual
    options_good = options.copy()
    options_good.extend(["--kernel", config["kernel_name_1"]])
    returncode = binary_handler_profile_rocprof_compute(  # noqa: F841
        config, workload_dir, options_good, check_success=True, roof=True
    )
    assert returncode == 0

    # Test one good and one nonexistent kernel using existing profiling data
    # Result should be passing as usual
    options_both = options.copy()
    options_both.extend([
        "--kernel",
        config["kernel_name_1"],
        "nonexistent_kernel_name_that_should_not_match_anything",
    ])
    returncode = binary_handler_profile_rocprof_compute(  # noqa: F841
        config, workload_dir, options_both, check_success=False, roof=True
    )
    assert returncode == 0

    # Verify CSV
    assert (Path(workload_dir) / "roofline.csv").exists()

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.roofline_2
def test_roof_cli_plot_generation(binary_handler_profile_rocprof_compute):
    if soc in ("MI100"):
        assert True
        return

    try:
        import plotext as plt  # noqa: F401

        cli_available = True
    except ImportError:
        cli_available = False

    if cli_available:
        options = ["--device", "0", "--roof-only"]
        workload_dir = test_utils.get_output_dir()

        returncode = binary_handler_profile_rocprof_compute(  # noqa: F841
            config, workload_dir, options, check_success=False, roof=True
        )

        test_utils.clean_output_dir(config["cleanup"], workload_dir)
    else:
        pytest.skip("plotext not available for CLI testing")


@pytest.mark.roofline_2
def test_roof_error_handling(binary_handler_profile_rocprof_compute):
    if soc in ("MI100"):
        assert True
        return

    options = ["--device", "0", "--roof-only"]
    workload_dir = test_utils.get_output_dir()

    returncode = binary_handler_profile_rocprof_compute(  # noqa: F841
        config, workload_dir, options, check_success=False, roof=True
    )

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.roofline_2
def test_roofline_plot_points_data_generation():
    """
    Test that plot points data structure is correctly generated with:
    - Symbol assignments
    - AI values (FLOPs/Byte)
    - Performance values (GFLOPs/s)
    - Memory/Compute bound status
    - Cache level information
    """
    if soc in ("MI100"):
        pytest.skip("Skipping roofline test for MI100")
        return

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    try:
        from roofline import Roofline
        from utils.specs import generate_machine_specs

        class MockArgs:
            def __init__(self):
                self.roof_only = True
                self.mem_level = "ALL"
                self.sort = "ALL"
                self.roofline_data_type = ["FP32"]

        args = MockArgs()
        mspec = generate_machine_specs(None, None)

        mock_ai_data = {
            "ai_l1": [[0.5, 1.2], [100.0, 150.0]],
            "ai_l2": [[0.3, 0.8], [80.0, 120.0]],
            "ai_hbm": [[0.1, 0.4], [50.0, 90.0]],
            "kernelNames": ["kernel_A", "kernel_B"],
        }

        mock_ceiling_data = {
            "l1": [[0.01, 10], [10, 1000], 100],
            "l2": [[0.01, 10], [10, 800], 80],
            "hbm": [[0.01, 10], [10, 500], 50],
            "valu": [[1, 100], [200, 200], 200],
            "mfma": [[1, 100], [500, 500], 500],
        }

        plot_points_data = []
        cache_colors = {
            "ai_l1": "blue",
            "ai_l2": "green",
            "ai_hbm": "red",
        }

        run_parameters = {
            "workload_dir": None,
            "device_id": 0,
            "sort_type": "kernels",
            "mem_level": "ALL",
            "is_standalone": False,
            "roofline_data_type": ["FP32"],
        }
        roofline_instance = Roofline(args, mspec, run_parameters)

        for cache_level in ["ai_l1", "ai_l2", "ai_hbm"]:
            if cache_level in mock_ai_data:
                x_vals = mock_ai_data[cache_level][0]
                y_vals = mock_ai_data[cache_level][1]
                num_kernels = len(mock_ai_data["kernelNames"])

                for i in range(min(len(x_vals), num_kernels)):
                    if x_vals[i] > 0 and y_vals[i] > 0:
                        status = roofline_instance._determine_kernel_bound_status(
                            ai_value=x_vals[i],
                            performance=y_vals[i],
                            cache_level=cache_level,
                            ceiling_data=mock_ceiling_data,
                        )

                        plot_points_data.append({
                            "symbol": None,
                            "color": cache_colors.get(cache_level, "gray"),
                            "cache_level": cache_level.replace("ai_", "", 1).upper(),
                            "ai": f"{x_vals[i]:.2f}",
                            "performance": f"{y_vals[i]:.2f}",
                            "status": status,
                            "kernel_idx": i,
                        })

        assert len(plot_points_data) > 0, "Plot points data should not be empty"

        for point in plot_points_data:
            assert "cache_level" in point
            assert "ai" in point
            assert "performance" in point
            assert "status" in point
            assert "kernel_idx" in point
            assert "color" in point

            assert point["cache_level"] in ["L1", "L2", "HBM"]

            assert point["status"] in ["Memory Bound", "Compute Bound", "Unknown"]

            assert isinstance(point["ai"], str)
            assert isinstance(point["performance"], str)

    except ImportError:
        pytest.skip("Could not import roofline module for direct testing")


@pytest.mark.roofline_2
def test_roofline_bound_status_calculation():
    """
    Test _determine_kernel_bound_status() correctly classifies kernels as
    Memory Bound or Compute Bound based on their AI and performance vs ceilings.
    """
    if soc in ("MI100"):
        pytest.skip("Skipping roofline test for MI100")
        return

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    try:
        from roofline import Roofline
        from utils.specs import generate_machine_specs

        class MockArgs:
            def __init__(self):
                self.roof_only = True
                self.mem_level = "ALL"
                self.sort = "ALL"
                self.roofline_data_type = ["FP32"]

        args = MockArgs()
        mspec = generate_machine_specs(None, None)
        run_parameters = {
            "workload_dir": None,
            "device_id": 0,
            "sort_type": "kernels",
            "mem_level": "ALL",
            "is_standalone": False,
            "roofline_data_type": ["FP32"],
        }
        roofline_instance = Roofline(args, mspec, run_parameters)

        ceiling_data = {
            "hbm": [[0.01, 10], [10, 1000], 100],
            "valu": [[1, 100], [200, 200], 200],
            "mfma": [[1, 100], [500, 500], 500],
        }

        status1 = roofline_instance._determine_kernel_bound_status(
            ai_value=1.0,
            performance=100.0,
            cache_level="ai_hbm",
            ceiling_data=ceiling_data,
        )
        assert status1 == "Memory Bound", f"Expected Memory Bound, got {status1}"

        status2 = roofline_instance._determine_kernel_bound_status(
            ai_value=5.0,
            performance=150.0,
            cache_level="ai_hbm",
            ceiling_data=ceiling_data,
        )
        assert status2 == "Compute Bound", f"Expected Compute Bound, got {status2}"

        status3 = roofline_instance._determine_kernel_bound_status(
            ai_value=1.0,
            performance=100.0,
            cache_level="ai_l1",
            ceiling_data=ceiling_data,
        )
        assert status3 == "Unknown", f"Expected Unknown, got {status3}"

        bad_ceiling_data = {
            "hbm": [100],
        }
        status4 = roofline_instance._determine_kernel_bound_status(
            ai_value=1.0,
            performance=100.0,
            cache_level="ai_hbm",
            ceiling_data=bad_ceiling_data,
        )
        assert status4 == "Unknown", f"Expected Unknown for bad data, got {status4}"

    except ImportError:
        pytest.skip("Could not import roofline module for direct testing")


@pytest.mark.roofline_2
def test_roofline_many_kernels_dynamic_height(binary_handler_profile_rocprof_compute):
    """
    Test roofline CSV generation with many kernels.
    """
    if soc in ("MI100"):
        pytest.skip("Skipping roofline test for MI100")
        return

    options = ["--device", "0", "--roof-only"]
    workload_dir = test_utils.get_output_dir()

    returncode = binary_handler_profile_rocprof_compute(
        config, workload_dir, options, check_success=False, roof=True
    )

    assert returncode == 0, "Roofline profiling should succeed"

    assert (Path(workload_dir) / "roofline.csv").exists()

    file_dict = test_utils.check_csv_files(workload_dir, 1, num_kernels)
    assert sorted(list(file_dict.keys())) == ROOF_ONLY_FILES

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.misc
def test_device_filter(binary_handler_profile_rocprof_compute):
    options = ["--device", "0"]
    workload_dir = test_utils.get_output_dir()
    binary_handler_profile_rocprof_compute(config, workload_dir, options)

    file_dict = test_utils.check_csv_files(workload_dir, 1, num_kernels)
    if soc == "MI100":
        assert sorted(list(file_dict.keys())) == CSVS
    elif soc == "MI200":
        assert sorted(list(file_dict.keys())) == CSVS
    elif "MI300" in soc:
        assert sorted(list(file_dict.keys())) == CSVS
    elif "MI350" in soc:
        assert sorted(list(file_dict.keys())) == CSVS
    else:
        print(f"Testing isn't supported yet for {soc}")
        assert 0

    # TODO - verify expected device id in results

    validate(
        inspect.stack()[0][3],
        workload_dir,
        file_dict,
    )

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.kernel_execution
def test_kernel(binary_handler_profile_rocprof_compute):
    options = ["--kernel", config["kernel_name_1"]]
    workload_dir = test_utils.get_output_dir()
    binary_handler_profile_rocprof_compute(config, workload_dir, options)

    file_dict = test_utils.check_csv_files(workload_dir, num_devices, num_kernels)
    if soc == "MI100":
        assert sorted(list(file_dict.keys())) == CSVS
    elif soc == "MI200":
        assert sorted(list(file_dict.keys())) == CSVS
    elif "MI300" in soc:
        assert sorted(list(file_dict.keys())) == CSVS
    elif "MI350" in soc:
        assert sorted(list(file_dict.keys())) == CSVS
    else:
        print(f"Testing isn't supported yet for {soc}")
        assert 0

    validate(
        inspect.stack()[0][3],
        workload_dir,
        file_dict,
    )

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.dispatch
def test_dispatch_0(binary_handler_profile_rocprof_compute):
    options = ["--dispatch", "1"]
    workload_dir = test_utils.get_output_dir()
    binary_handler_profile_rocprof_compute(config, workload_dir, options)

    file_dict = test_utils.check_csv_files(workload_dir, num_devices, 1)
    if soc == "MI100":
        assert sorted(list(file_dict.keys())) == CSVS
    elif soc == "MI200":
        assert sorted(list(file_dict.keys())) == CSVS
    elif "MI300" in soc:
        assert sorted(list(file_dict.keys())) == CSVS
    elif "MI350" in soc:
        assert sorted(list(file_dict.keys())) == CSVS
    else:
        print(f"Testing isn't supported yet for {soc}")
        assert 0

    validate(
        inspect.stack()[0][3],
        workload_dir,
        file_dict,
        [
            "--dispatch",
            "1",
        ],
    )

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.dispatch
def test_dispatch_0_1(binary_handler_profile_rocprof_compute):
    options = ["--dispatch", "1:2"]
    workload_dir = test_utils.get_output_dir()
    binary_handler_profile_rocprof_compute(config, workload_dir, options)

    file_dict = test_utils.check_csv_files(workload_dir, num_devices, 2)
    if soc == "MI100":
        assert sorted(list(file_dict.keys())) == CSVS
    elif soc == "MI200":
        assert sorted(list(file_dict.keys())) == CSVS
    elif "MI300" in soc:
        assert sorted(list(file_dict.keys())) == CSVS
    elif "MI350" in soc:
        assert sorted(list(file_dict.keys())) == CSVS
    else:
        print(f"Testing isn't supported yet for {soc}")
        assert 0

    validate(
        inspect.stack()[0][3],
        workload_dir,
        file_dict,
        ["--dispatch", "1", "2"],
    )

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.dispatch
def test_dispatch_2(binary_handler_profile_rocprof_compute):
    options = ["--dispatch", "1"]
    workload_dir = test_utils.get_output_dir()
    binary_handler_profile_rocprof_compute(config, workload_dir, options)

    file_dict = test_utils.check_csv_files(workload_dir, num_devices, 1)
    if soc == "MI100":
        assert sorted(list(file_dict.keys())) == CSVS
    elif soc == "MI200":
        assert sorted(list(file_dict.keys())) == CSVS
    elif "MI300" in soc:
        assert sorted(list(file_dict.keys())) == CSVS
    elif "MI350" in soc:
        assert sorted(list(file_dict.keys())) == CSVS
    else:
        print(f"Testing isn't supported yet for {soc}")
        assert 0

    validate(
        inspect.stack()[0][3],
        workload_dir,
        file_dict,
        [
            "--dispatch",
            "1",
        ],
    )

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.join
def test_join_type_grid(binary_handler_profile_rocprof_compute):
    options = ["--join-type", "grid"]
    workload_dir = test_utils.get_output_dir()
    binary_handler_profile_rocprof_compute(config, workload_dir, options)

    file_dict = test_utils.check_csv_files(workload_dir, num_devices, num_kernels)
    if soc == "MI100":
        assert sorted(list(file_dict.keys())) == CSVS
    elif soc == "MI200":
        assert sorted(list(file_dict.keys())) == CSVS
    elif "MI300" in soc:
        assert sorted(list(file_dict.keys())) == CSVS
    elif "MI350" in soc:
        assert sorted(list(file_dict.keys())) == CSVS
    else:
        print(f"Testing isn't supported yet for {soc}")
        assert 0

    validate(
        inspect.stack()[0][3],
        workload_dir,
        file_dict,
    )

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.join
def test_join_type_kernel(binary_handler_profile_rocprof_compute):
    options = ["--join-type", "kernel"]
    workload_dir = test_utils.get_output_dir()
    binary_handler_profile_rocprof_compute(config, workload_dir, options)

    file_dict = test_utils.check_csv_files(workload_dir, num_devices, num_kernels)

    if soc == "MI100":
        assert sorted(list(file_dict.keys())) == CSVS
    elif soc == "MI200":
        assert sorted(list(file_dict.keys())) == CSVS
    elif "MI300" in soc:
        assert sorted(list(file_dict.keys())) == CSVS
    elif "MI350" in soc:
        assert sorted(list(file_dict.keys())) == CSVS
    else:
        print(f"Testing isn't supported yet for {soc}")
        assert 0

    validate(
        inspect.stack()[0][3],
        workload_dir,
        file_dict,
    )

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.sort
def test_roof_sort_dispatches(
    binary_handler_profile_rocprof_compute,
    binary_handler_analyze_rocprof_compute,
):
    """Profile creates CSV; analyze with --sort dispatches generates output."""
    if soc in ("MI100"):
        pytest.skip("Roofline not supported on MI100")

    profile_options = ["--device", "0", "--roof-only"]
    workload_dir = test_utils.get_output_dir()
    returncode = binary_handler_profile_rocprof_compute(
        config, workload_dir, profile_options, check_success=False, roof=True
    )
    assert returncode == 0

    file_dict = test_utils.check_csv_files(workload_dir, 1, num_kernels)
    assert sorted(list(file_dict.keys())) == ROOF_ONLY_FILES

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
        "--sort",
        "dispatches",
    ])
    assert code == 0

    html_files = list(Path(workload_dir).glob("empirRoof_*.html"))
    assert len(html_files) > 0, "Analyze should generate roofline HTML files"

    validate(inspect.stack()[0][3], workload_dir, file_dict)
    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.sort
def test_roof_sort_kernels(
    binary_handler_profile_rocprof_compute,
    binary_handler_analyze_rocprof_compute,
):
    """Profile creates CSV; analyze with --sort kernels generates output."""
    if soc in ("MI100"):
        pytest.skip("Roofline not supported on MI100")

    profile_options = ["--device", "0", "--roof-only"]
    workload_dir = test_utils.get_output_dir()
    returncode = binary_handler_profile_rocprof_compute(
        config, workload_dir, profile_options, check_success=False, roof=True
    )
    assert returncode == 0

    file_dict = test_utils.check_csv_files(workload_dir, 1, num_kernels)
    assert sorted(list(file_dict.keys())) == ROOF_ONLY_FILES

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
        "--sort",
        "kernels",
    ])
    assert code == 0

    html_files = list(Path(workload_dir).glob("empirRoof_*.html"))
    assert len(html_files) > 0, "Analyze should generate roofline HTML files"

    validate(inspect.stack()[0][3], workload_dir, file_dict)
    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.section
def test_lds_section(binary_handler_profile_rocprof_compute):
    options = ["--block", "12"]
    workload_dir = test_utils.get_output_dir()
    _ = binary_handler_profile_rocprof_compute(
        config, workload_dir, options, check_success=True, roof=False
    )

    file_dict = test_utils.check_csv_files(workload_dir, 1, num_kernels)
    validate(
        inspect.stack()[0][3],
        workload_dir,
        file_dict,
    )

    assert test_utils.check_file_pattern(
        "- '12'", f"{workload_dir}/profiling_config.yaml"
    )
    results_files = Path(workload_dir).glob("results_*.csv")
    assert any(
        test_utils.check_file_pattern("SQ_INSTS_LDS", str(f)) for f in results_files
    )
    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.section
def test_instmix_memchart_section(binary_handler_profile_rocprof_compute):
    options = ["--block", "10", "3"]
    workload_dir = test_utils.get_output_dir()
    _ = binary_handler_profile_rocprof_compute(
        config, workload_dir, options, check_success=True, roof=False
    )

    file_dict = test_utils.check_csv_files(workload_dir, 1, num_kernels)
    validate(
        inspect.stack()[0][3],
        workload_dir,
        file_dict,
    )

    assert test_utils.check_file_pattern(
        "- '10'", f"{workload_dir}/profiling_config.yaml"
    )
    assert test_utils.check_file_pattern(
        "- '3'", f"{workload_dir}/profiling_config.yaml"
    )
    results_files = Path(workload_dir).glob("results_*.csv")
    assert any(
        test_utils.check_file_pattern("TA_FLAT_WAVEFRONTS", str(f))
        for f in results_files
    )
    results_files = Path(workload_dir).glob("results_*.csv")
    assert any(
        test_utils.check_file_pattern("SQC_TC_DATA_READ_REQ", str(f))
        for f in results_files
    )
    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.section
def test_lds_sol_section(binary_handler_profile_rocprof_compute):
    options = ["--block", "12.1"]
    workload_dir = test_utils.get_output_dir()
    _ = binary_handler_profile_rocprof_compute(
        config, workload_dir, options, check_success=True, roof=False
    )

    file_dict = test_utils.check_csv_files(workload_dir, 1, num_kernels)
    validate(
        inspect.stack()[0][3],
        workload_dir,
        file_dict,
    )

    assert test_utils.check_file_pattern(
        "- '12.1'", f"{workload_dir}/profiling_config.yaml"
    )
    results_files = Path(workload_dir).glob("results_*.csv")
    assert any(
        test_utils.check_file_pattern("SQ_ACTIVE_INST_LDS", str(f))
        for f in results_files
    )
    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.section
def test_instmix_section_global_write_kernel(binary_handler_profile_rocprof_compute):
    options = ["-k", "global_write", "--block", "10"]
    custom_config = dict(config)
    custom_config["kernel_name_1"] = "global_write"
    custom_config["app_1"] = ["./tests/vmem"]
    num_kernels = 1

    workload_dir = test_utils.get_output_dir()
    _ = binary_handler_profile_rocprof_compute(
        custom_config, workload_dir, options, check_success=True, roof=False
    )

    file_dict = test_utils.check_csv_files(workload_dir, 1, num_kernels)
    validate(
        inspect.stack()[0][3],
        workload_dir,
        file_dict,
    )

    assert test_utils.check_file_pattern(
        "- '10'", f"{workload_dir}/profiling_config.yaml"
    )
    assert test_utils.check_file_pattern(
        "- global_write", f"{workload_dir}/profiling_config.yaml"
    )
    results_files = Path(workload_dir).glob("results_*.csv")
    assert any(
        test_utils.check_file_pattern("TA_FLAT_WAVEFRONTS", str(f))
        for f in results_files
    )
    results_files = Path(workload_dir).glob("results_*.csv")
    assert any(
        test_utils.check_file_pattern("global_write", str(f)) for f in results_files
    )
    results_files = Path(workload_dir).glob("results_*.csv")
    assert not any(
        test_utils.check_file_pattern("global_read", str(f)) for f in results_files
    )
    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.section
def test_list_metrics(binary_handler_profile_rocprof_compute):
    options = ["--list-metrics", "gfx90a"]
    workload_dir = test_utils.get_output_dir()
    _ = binary_handler_profile_rocprof_compute(
        config, workload_dir, options, check_success=True, roof=False
    )
    # workload dir should not exist
    assert not Path(workload_dir).exists()
    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.section
def test_list_metrics_with_block(binary_handler_profile_rocprof_compute):
    options = ["--list-metrics", "gfx90a", "--block", "10"]
    workload_dir = test_utils.get_output_dir()
    code = binary_handler_profile_rocprof_compute(
        config, workload_dir, options, check_success=False, roof=False
    )
    # Should return code 1 since --block cannot be used with --list-metrics
    assert code == 1
    # workload dir should not exist
    assert not Path(workload_dir).exists()
    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.section
def test_list_available_metrics(binary_handler_profile_rocprof_compute, capsys):
    options = ["--list-available-metrics"]
    workload_dir = test_utils.get_output_dir()
    _ = binary_handler_profile_rocprof_compute(
        config, workload_dir, options, check_success=True, roof=False
    )
    # workload dir should not exist
    assert not Path(workload_dir).exists()
    test_utils.clean_output_dir(config["cleanup"], workload_dir)

    # Test output
    output = capsys.readouterr().out
    assert "0 -> Top Stats" in output
    assert "1 -> System Info" in output


@pytest.mark.section
def test_list_available_metrics_with_block(
    binary_handler_profile_rocprof_compute, capsys
):
    options = ["--list-available-metrics", "--block", "10"]
    workload_dir = test_utils.get_output_dir()
    code = binary_handler_profile_rocprof_compute(
        config, workload_dir, options, check_success=False, roof=False
    )
    # Should return code 1 since --block cannot be used with --list-available-metrics
    assert code == 1
    # workload dir should not exist
    assert not Path(workload_dir).exists()
    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.path
def test_comprehensive_error_paths():
    """Simplified test for error path coverage"""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    from utils.parser import (
        build_comparable_columns,
        build_eval_string,
        calc_builtin_var,
    )

    columns = build_comparable_columns("ms")
    expected = [
        "Count(ms)",
        "Sum(ms)",
        "Mean(ms)",
        "Median(ms)",
        "Standard Deviation(ms)",
    ]
    for expected_col in expected:
        assert expected_col in columns

    class MockSysInfo:
        total_l2_chan = 16

    sys_info = MockSysInfo()
    result = calc_builtin_var(42, sys_info)
    assert result == 42

    result = calc_builtin_var("$total_l2_chan", sys_info)
    assert result == 16

    try:
        build_eval_string("test", None, config={})
        assert False, "Should raise exception for None coll_level"
    except Exception as e:
        assert "coll_level can not be None" in str(e)


@pytest.mark.live_attach_detach
def test_live_attach_detach_block(binary_handler_profile_rocprof_compute):
    options = ["--block", "3.1.1", "4.1.1", "5.1.1"]
    workload_dir = test_utils.get_output_dir()

    # TODO: temp fix for sdk defautly disable attach/detach,
    # remove after it sets default to enable
    env = os.environ.copy()
    env["ROCP_TOOL_ATTACH"] = "1"

    process_workload = None

    try:
        # Start workload
        process_workload = subprocess.Popen(config["app_hip_dynamic_shared"], env=env)
        time.sleep(5)  # Give workload time to start

        attach_detach = {
            "attach_pid": process_workload.pid,
            "attach-duration-msec": attach_detach_interval_msec_no_delay,
        }

        # Run profiler (might fail / timeout / throw)
        binary_handler_profile_rocprof_compute(
            config,
            workload_dir,
            options,
            check_success=True,
            roof=False,
            app_name="app_hip_dynamic_shared",
            attach_detach_para=attach_detach,
        )

    finally:
        if process_workload and process_workload.poll() is None:
            print(f"[finally] killing workload pid={process_workload.pid}")
            process_workload.kill()
            process_workload.wait()

    # Validate results
    file_dict = test_utils.check_csv_files(workload_dir, 1, num_kernels)
    validate(inspect.stack()[0][3], workload_dir, file_dict)
    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.skip(
    reason="Temporarily disabled: \
                  waiting for SDK fix for no outputfile with thread sleeping"
)
@pytest.mark.live_attach_detach
def test_live_attach_detach_block_thread_sleep(binary_handler_profile_rocprof_compute):
    options = ["--block", "3.1.1", "4.1.1", "5.1.1"]
    workload_dir = test_utils.get_output_dir()

    # TODO: temp fix for sdk defautly disable attach/detach,
    # remove after it sets default to enable
    env = os.environ.copy()
    env["ROCP_TOOL_ATTACH"] = "1"

    process_workload = None

    try:
        # Start workload with sleep mode enabled
        process_workload = subprocess.Popen(
            [*config["app_hip_dynamic_shared"], "--enable-sleep"], env=env
        )
        time.sleep(5)  # Give workload time to start

        attach_detach = {
            "attach_pid": process_workload.pid,
            "attach-duration-msec": attach_detach_interval_msec_with_delay,
        }

        # Main profiling call (can fail or hang)
        binary_handler_profile_rocprof_compute(
            config,
            workload_dir,
            options,
            check_success=True,
            roof=False,
            app_name="app_hip_dynamic_shared",
            attach_detach_para=attach_detach,
        )

    finally:
        if process_workload and process_workload.poll() is None:
            print(f"[finally] killing workload pid={process_workload.pid}")
            process_workload.kill()
            process_workload.wait()

    # Validate output
    file_dict = test_utils.check_csv_files(workload_dir, 1, num_kernels)
    validate(
        inspect.stack()[0][3],
        workload_dir,
        file_dict,
    )

    # Check profiling_config.yaml block entries
    config_file = f"{workload_dir}/profiling_config.yaml"
    assert test_utils.check_file_pattern("- 3.1.1", config_file)
    assert test_utils.check_file_pattern("- 4.1.1", config_file)
    assert test_utils.check_file_pattern("- 5.1.1", config_file)
    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.live_attach_detach
def test_live_attach_detach_singlepass_launch_stats(
    binary_handler_profile_rocprof_compute,
):
    options = ["--set", "launch_stats"]
    workload_dir = test_utils.get_output_dir()

    # TODO: temp fix for sdk defautly disable attach/detach,
    # remove after it sets default to enable
    env = os.environ.copy()
    env["ROCP_TOOL_ATTACH"] = "1"

    process_workload = None

    try:
        # Start workload
        process_workload = subprocess.Popen(config["app_hip_dynamic_shared"], env=env)
        time.sleep(5)  # Give workload time to start

        attach_detach = {
            "attach_pid": process_workload.pid,
            "attach-duration-msec": attach_detach_interval_msec_no_delay,
        }

        # Profiling step (may fail)
        binary_handler_profile_rocprof_compute(
            config,
            workload_dir,
            options,
            check_success=True,
            roof=False,
            app_name="app_hip_dynamic_shared",
            attach_detach_para=attach_detach,
        )

    finally:
        if process_workload and process_workload.poll() is None:
            print(f"[finally] killing workload pid={process_workload.pid}")
            process_workload.kill()
            process_workload.wait()

    # Validate CSVs & output correctness
    file_dict = test_utils.check_csv_files(workload_dir, 1, num_kernels)
    validate(
        inspect.stack()[0][3],
        workload_dir,
        file_dict,
    )

    # Check that launch-stat sets were applied
    config_file = f"{workload_dir}/profiling_config.yaml"
    for tag in [
        "7.1.0",
        "7.1.1",
        "7.1.2",
        "7.1.5",
        "7.1.6",
        "7.1.7",
        "7.1.8",
        "7.1.9",
    ]:
        assert test_utils.check_file_pattern(f"- {tag}", config_file)

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.sets_func
class TestSetsIntegration:
    def test_memory_throughput_set(self, binary_handler_profile_rocprof_compute):
        options = ["--set", "mem_thruput"]
        workload_dir = test_utils.get_output_dir()

        binary_handler_profile_rocprof_compute(
            config,
            workload_dir,
            options,
            check_success=True,
            roof=False,
        )

        assert test_utils.get_num_pmc_file(workload_dir) == 1

        memory_metrics = ["16.1.2", "17.1.0"]
        for metric_id in memory_metrics:
            assert metric_id in open(Path(workload_dir) / "log.txt").read(), (
                f"Expected memory metric {metric_id} not found"
            )

        test_utils.clean_output_dir(config["cleanup"], workload_dir)

    def test_launch_stats_set(self, binary_handler_profile_rocprof_compute):
        options = ["--set", "launch_stats"]
        workload_dir = test_utils.get_output_dir()

        binary_handler_profile_rocprof_compute(
            config,
            workload_dir,
            options,
            check_success=True,
            roof=False,
        )

        assert test_utils.get_num_pmc_file(workload_dir) == 1

        test_utils.clean_output_dir(config["cleanup"], workload_dir)

    def test_compute_thruput_util_set(self, binary_handler_profile_rocprof_compute):
        options = ["--set", "compute_thruput_util"]
        workload_dir = test_utils.get_output_dir()

        binary_handler_profile_rocprof_compute(
            config,
            workload_dir,
            options,
            check_success=True,
            roof=False,
        )

        assert test_utils.get_num_pmc_file(workload_dir) == 1

        assert test_utils.check_file_pattern(
            "- 11.2.3", f"{workload_dir}/profiling_config.yaml"
        )

        test_utils.clean_output_dir(config["cleanup"], workload_dir)

    def test_compute_thruput_flops_set(self, binary_handler_profile_rocprof_compute):
        options = ["--set", "compute_thruput_flops"]
        workload_dir = test_utils.get_output_dir()

        binary_handler_profile_rocprof_compute(
            config,
            workload_dir,
            options,
            check_success=True,
            roof=False,
        )

        assert test_utils.get_num_pmc_file(workload_dir) == 1

        test_utils.clean_output_dir(config["cleanup"], workload_dir)

    def test_invalid_set_error_handling(self, binary_handler_profile_rocprof_compute):
        options = ["--set", "nonexistent_set"]
        workload_dir = test_utils.get_output_dir()

        returncode = binary_handler_profile_rocprof_compute(
            config,
            workload_dir,
            options,
            check_success=False,
            roof=False,
        )

        assert returncode == 1
        test_utils.clean_output_dir(config["cleanup"], workload_dir)

    def test_set_and_block_mutual_exclusion(
        self, binary_handler_profile_rocprof_compute
    ):
        options = ["--set", "compute_thruput_util", "--block", "12"]
        workload_dir = test_utils.get_output_dir()

        returncode = binary_handler_profile_rocprof_compute(
            config, workload_dir, options, check_success=False, roof=False
        )

        assert returncode == 1
        test_utils.clean_output_dir(config["cleanup"], workload_dir)

    def test_list_sets_functionality(self, binary_handler_profile_rocprof_compute):
        options = ["--list-sets"]
        workload_dir = test_utils.get_output_dir()

        binary_handler_profile_rocprof_compute(
            config,
            workload_dir,
            options,
            check_success=False,
            roof=False,
        )
        # workload dir should not exist
        assert not Path(workload_dir).exists()
        test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.iteration_multiplexing_1
def test_profiler_options(binary_handler_profile_rocprof_compute):
    options = ["--no-native-tool", "--iteration-multiplexing"]
    workload_dir = test_utils.get_output_dir()
    code = binary_handler_profile_rocprof_compute(
        config, workload_dir, options, check_success=False, roof=False
    )
    assert code == 1


@pytest.mark.iteration_multiplexing_1
def test_iteration_multiplexing(binary_handler_profile_rocprof_compute):
    options = ["--iteration-multiplexing"]
    workload_dir = test_utils.get_output_dir()
    _ = binary_handler_profile_rocprof_compute(
        config, workload_dir, options, check_success=True, roof=False
    )

    file_dict = test_utils.check_csv_files(workload_dir, num_devices, num_kernels)
    if soc == "MI100":
        assert sorted(list(file_dict.keys())) == CSVS
    elif soc == "MI200":
        assert sorted(list(file_dict.keys())) == CSVS
    elif "MI300" in soc:
        assert sorted(list(file_dict.keys())) == CSVS
    elif "MI350" in soc:
        assert sorted(list(file_dict.keys())) == CSVS
    else:
        print(f"Testing isn't supported yet for {soc}")
        assert 0

    validate(
        inspect.stack()[0][3],
        workload_dir,
        file_dict,
    )

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.iteration_multiplexing_1
def test_iteration_multiplexing_kernel(binary_handler_profile_rocprof_compute):
    options = ["--iteration-multiplexing", "kernel"]
    workload_dir = test_utils.get_output_dir()
    _ = binary_handler_profile_rocprof_compute(
        config, workload_dir, options, check_success=True, roof=False
    )

    file_dict = test_utils.check_csv_files(workload_dir, num_devices, num_kernels)
    if soc == "MI100":
        assert sorted(list(file_dict.keys())) == CSVS
    elif soc == "MI200":
        assert sorted(list(file_dict.keys())) == CSVS
    elif "MI300" in soc:
        assert sorted(list(file_dict.keys())) == CSVS
    elif "MI350" in soc:
        assert sorted(list(file_dict.keys())) == CSVS
    else:
        print(f"Testing isn't supported yet for {soc}")
        assert 0

    validate(
        inspect.stack()[0][3],
        workload_dir,
        file_dict,
    )

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.iteration_multiplexing_1
def test_iteration_multiplexing_kernel_launch_params(
    binary_handler_profile_rocprof_compute,
):
    options = ["--iteration-multiplexing", "kernel_launch_params"]
    workload_dir = test_utils.get_output_dir()
    _ = binary_handler_profile_rocprof_compute(
        config, workload_dir, options, check_success=True, roof=False
    )

    file_dict = test_utils.check_csv_files(workload_dir, num_devices, num_kernels)
    if soc == "MI100":
        assert sorted(list(file_dict.keys())) == CSVS
    elif soc == "MI200":
        assert sorted(list(file_dict.keys())) == CSVS
    elif "MI300" in soc:
        assert sorted(list(file_dict.keys())) == CSVS
    elif "MI350" in soc:
        assert sorted(list(file_dict.keys())) == CSVS
    else:
        print(f"Testing isn't supported yet for {soc}")
        assert 0

    validate(
        inspect.stack()[0][3],
        workload_dir,
        file_dict,
    )

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.iteration_multiplexing_2
def test_iteration_multiplexing_deterministic_counter_accuracy(
    binary_handler_profile_rocprof_compute,
    binary_handler_analyze_rocprof_compute,
):
    # These metrics should cover the deterministic counters being checked
    # Block 4 (roofline) included to verify roofline counters under multiplexing
    options = ["--block", "4", "6.1.5", "6.1.6", "7.2.2", "10.1"]
    workload_dir = test_utils.get_output_dir(param_id="no_iter_mplx")
    _ = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=True,
        roof=False,
        app_name="app_laplace_eqn",
    )
    test_utils.check_csv_files(workload_dir, num_devices, num_kernels)
    binary_handler_analyze_rocprof_compute(["analyze", "--path", workload_dir])
    counters_no_multiplexing = pd.read_csv(Path(workload_dir) / "pmc_perf.csv")
    test_utils.clean_output_dir(config["cleanup"], workload_dir)

    options = [
        "--block",
        "4",
        "6.1.5",
        "6.1.6",
        "7.2.2",
        "10.1",
        "--iteration-multiplexing",
        "kernel",
    ]
    workload_dir = test_utils.get_output_dir(param_id="iter_mplx_kernel")
    _ = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=True,
        roof=False,
        app_name="app_laplace_eqn_iter",
    )
    test_utils.check_csv_files(workload_dir, num_devices, num_kernels)
    binary_handler_analyze_rocprof_compute(["analyze", "--path", workload_dir])
    counters_kernel = pd.read_csv(Path(workload_dir) / "pmc_perf.csv")
    test_utils.clean_output_dir(config["cleanup"], workload_dir)

    options = [
        "--block",
        "4",
        "6.1.5",
        "6.1.6",
        "7.2.2",
        "10.1",
        "--iteration-multiplexing",
        "kernel_launch_params",
    ]
    workload_dir_klp = test_utils.get_output_dir(param_id="iter_mplx_params")
    _ = binary_handler_profile_rocprof_compute(
        config,
        workload_dir_klp,
        options,
        check_success=True,
        roof=True,
        app_name="app_laplace_eqn_iter",
    )
    test_utils.check_csv_files(workload_dir_klp, num_devices, num_kernels)
    binary_handler_analyze_rocprof_compute(["analyze", "--path", workload_dir_klp])
    counters_kernel_launch_params = pd.read_csv(Path(workload_dir_klp) / "pmc_perf.csv")

    assert are_deterministic_counters_equal(
        [counters_kernel, counters_kernel_launch_params], counters_no_multiplexing
    )

    # Roofline assertions (MI100 doesn't support roofline)
    if soc not in ("MI100"):
        assert os.path.exists(f"{workload_dir_klp}/roofline.csv")
        roofline_df = pd.read_csv(f"{workload_dir_klp}/roofline.csv")
        assert len(roofline_df) >= num_devices

    test_utils.clean_output_dir(config["cleanup"], workload_dir_klp)


@pytest.mark.iteration_multiplexing_stochastic
def test_iteration_multiplexing_stochastic_counter_accuracy(
    binary_handler_profile_rocprof_compute,
    binary_handler_analyze_rocprof_compute,
):
    workload_dir = test_utils.get_output_dir(param_id="no_iter_mplx")
    # These metrics should cover the L1 cache stochastic counters
    # Block 4 (roofline) included to verify roofline counters under multiplexing
    options = ["--block", "4", "16.1", "16.3"]
    _ = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=True,
        roof=False,
        app_name="app_laplace_eqn",
    )
    test_utils.check_csv_files(workload_dir, num_devices, num_kernels)
    binary_handler_analyze_rocprof_compute(["analyze", "--path", workload_dir])
    counters_no_multiplexing = pd.read_csv(Path(workload_dir) / "pmc_perf.csv")
    test_utils.clean_output_dir(config["cleanup"], workload_dir)

    options = [
        "--block",
        "4",
        "16.1",
        "16.3",
        "--iteration-multiplexing",
        "kernel",
    ]
    workload_dir = test_utils.get_output_dir(param_id="iter_mplx_kernel")
    _ = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=True,
        roof=False,
        app_name="app_laplace_eqn_iter",
    )
    test_utils.check_csv_files(workload_dir, num_devices, num_kernels)
    binary_handler_analyze_rocprof_compute(["analyze", "--path", workload_dir])
    counters_kernel = pd.read_csv(Path(workload_dir) / "pmc_perf.csv")
    test_utils.clean_output_dir(config["cleanup"], workload_dir)

    options = [
        "--block",
        "4",
        "16.1",
        "16.3",
        "--iteration-multiplexing",
        "kernel_launch_params",
    ]
    workload_dir_klp = test_utils.get_output_dir(param_id="iter_mplx_params")
    _ = binary_handler_profile_rocprof_compute(
        config,
        workload_dir_klp,
        options,
        check_success=True,
        roof=True,
        app_name="app_laplace_eqn_iter",
    )
    test_utils.check_csv_files(workload_dir_klp, num_devices, num_kernels)
    binary_handler_analyze_rocprof_compute(["analyze", "--path", workload_dir_klp])
    counters_kernel_launch_params = pd.read_csv(Path(workload_dir_klp) / "pmc_perf.csv")

    assert are_stochastic_counters_similar(
        [counters_kernel, counters_kernel_launch_params], counters_no_multiplexing
    )

    # Roofline assertions (MI100 doesn't support roofline)
    if soc not in ("MI100"):
        assert os.path.exists(f"{workload_dir_klp}/roofline.csv")
        roofline_df = pd.read_csv(f"{workload_dir_klp}/roofline.csv")
        assert len(roofline_df) >= num_devices

    test_utils.clean_output_dir(config["cleanup"], workload_dir_klp)


# Not part of automated test runs since testing all counters is expensive
def test_iteration_multiplexing_all_counter_accuracy(
    binary_handler_profile_rocprof_compute,
    binary_handler_analyze_rocprof_compute,
):
    workload_dir = test_utils.get_output_dir(param_id="no_iter_mplx")
    _ = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        check_success=True,
        roof=False,
        app_name="app_laplace_eqn",
    )
    test_utils.check_csv_files(workload_dir, num_devices, num_kernels)
    binary_handler_analyze_rocprof_compute(["analyze", "--path", workload_dir])
    counters_no_multiplexing = pd.read_csv(Path(workload_dir) / "pmc_perf.csv")
    test_utils.clean_output_dir(config["cleanup"], workload_dir)

    options = ["--iteration-multiplexing", "kernel"]
    workload_dir = test_utils.get_output_dir(param_id="iter_mplx_kernel")
    _ = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=True,
        roof=False,
        app_name="app_laplace_eqn_iter",
    )
    test_utils.check_csv_files(workload_dir, num_devices, num_kernels)
    binary_handler_analyze_rocprof_compute(["analyze", "--path", workload_dir])
    counters_kernel = pd.read_csv(Path(workload_dir) / "pmc_perf.csv")
    test_utils.clean_output_dir(config["cleanup"], workload_dir)

    options = ["--iteration-multiplexing", "kernel_launch_params"]
    workload_dir = test_utils.get_output_dir(param_id="iter_mplx_params")
    _ = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=True,
        roof=False,
        app_name="app_laplace_eqn_iter",
    )
    test_utils.check_csv_files(workload_dir, num_devices, num_kernels)
    binary_handler_analyze_rocprof_compute(["analyze", "--path", workload_dir])
    counters_kernel_launch_params = pd.read_csv(Path(workload_dir) / "pmc_perf.csv")
    test_utils.clean_output_dir(config["cleanup"], workload_dir)

    assert are_deterministic_counters_equal(
        [counters_kernel, counters_kernel_launch_params], counters_no_multiplexing
    )
    assert are_stochastic_counters_similar(
        [counters_kernel, counters_kernel_launch_params], counters_no_multiplexing
    )


@pytest.mark.iteration_multiplexing_2
def test_iteration_multiplexing_insufficient_dispatches(
    binary_handler_profile_rocprof_compute,
    binary_handler_analyze_rocprof_compute,
    capsys,
):
    """Verify graceful degradation when dispatches are too few for full
    counter coverage under iteration multiplexing.
    """
    options = [
        "--iteration-multiplexing",
        "kernel_launch_params",
    ]
    workload_dir = test_utils.get_output_dir(param_id="iter_mplx_insufficient")
    binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=True,
        roof=False,
        app_name="app_laplace_eqn_insufficient",
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    captured = capsys.readouterr()
    assert "missing counter data" in captured.out

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.iteration_multiplexing_2
def test_iteration_multiplexing_data_types(
    binary_handler_profile_rocprof_compute,
    binary_handler_analyze_rocprof_compute,
):
    """Verify roofline analysis with different data types (FP32 and FP16)
    on iteration-multiplexed profiling data.
    """
    if soc in ("MI100"):
        pytest.skip("Roofline not supported on MI100")
        return

    options = [
        "--block",
        "4",
        "--iteration-multiplexing",
        "kernel_launch_params",
    ]
    workload_dir = test_utils.get_output_dir(param_id="iter_mplx_dtypes")
    binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=True,
        roof=True,
        app_name="app_laplace_eqn",
    )

    assert os.path.exists(f"{workload_dir}/roofline.csv")
    roofline_df = pd.read_csv(f"{workload_dir}/roofline.csv")
    assert len(roofline_df) >= num_devices

    code_fp32 = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
        "--roofline-data-type",
        "FP32",
    ])
    assert code_fp32 == 0

    code_fp16 = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
        "--roofline-data-type",
        "FP16",
    ])
    assert code_fp16 == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


skip_if_no_torch_gpu = pytest.mark.skipif(
    (
        importlib.util.find_spec("torch") is None
        or not __import__("torch").cuda.is_available()
    ),
    reason=("PyTorch and GPU access are required for this test"),
)


@skip_if_no_torch_gpu
@pytest.mark.torch_trace
def test_torch_trace_profile(
    binary_handler_profile_rocprof_compute,
    binary_handler_analyze_rocprof_compute,
    capsys,
):
    """
    Test profile and analyze flow for PyTorch torch-trace.

    Runs profiling with --torch-trace, verifies profile outputs (pmc_perf, marker
    and counter CSVs), then runs analyze with --list-torch-operators and
    --torch-operator (PurePosixPath glob patterns like *relu, all), and verifies
    torch_trace directory, consolidated CSV contents (hierarchy, kernel, counters),
    and CLI output format (call tree grouped by source location, aggregated stats,
    kernel IDs, sort order).
    Requires PyTorch and GPU; not included in default suite.
    """
    workload_dir = test_utils.get_output_dir(param_id="torch_trace")

    # --torch-trace needs --experimental for profiling
    options = [
        "--experimental",
        "--torch-trace",
        "--iteration-multiplexing",
    ]

    returncode = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=True,
        app_name="torch_test_app",
    )

    # ---- Verify profiling output (checks 1–5) ----

    # 1. Profiling completed successfully
    assert returncode == 0, "Profiling the torch application failed"

    # 2. Validate profile outputs (PMC data validated by check_csv_files)
    num_devices = config.get("num_devices", 1)
    test_utils.check_csv_files(workload_dir, num_devices, 1)

    # 3. Marker/counter CSV pairs exist and counts match
    marker_api_trace_files = list(Path(workload_dir).glob("**/*marker_api_trace.csv"))
    counter_collection_files = list(
        Path(workload_dir).glob("**/*counter_collection.csv")
    )
    assert len(marker_api_trace_files) == len(counter_collection_files), (
        "Mismatch in number of marker_api_trace.csv and counter_collection.csv files"
    )
    for marker_file in marker_api_trace_files:
        corresponding_counter_file = marker_file.parent / marker_file.name.replace(
            "marker_api_trace", "counter_collection"
        )
        assert corresponding_counter_file.exists(), (
            f"counter_collection.csv not found for {marker_file}"
        )
        # 4. marker_api_trace CSVs: required columns and non-empty rows
        expected_marker_columns = {
            "Domain",
            "Function",
            "Process_Id",
            "Thread_Id",
            "Correlation_Id",
            "Start_Timestamp",
            "End_Timestamp",
        }
        with open(marker_file, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            assert fieldnames is not None, f"No columns in {marker_file}"
            for column in expected_marker_columns:
                assert column in fieldnames, (
                    f"Column '{column}' missing in {marker_file}"
                )
            found_row = False
            for row in reader:
                found_row = True
                assert row["Function"], f"Empty Function in {marker_file}"
                assert row["Correlation_Id"], f"Empty Correlation ID in {marker_file}"
                assert row["Start_Timestamp"], f"Empty Start_Timestamp in {marker_file}"
                assert row["End_Timestamp"], f"Empty End_Timestamp in {marker_file}"
            assert found_row, f"{marker_file} is empty"
        # 5. counter_collection CSVs: required columns and non-empty rows
        expected_counter_columns = {
            "Correlation_Id",
            "Kernel_Name",
            "Counter_Name",
            "Counter_Value",
            "Start_Timestamp",
            "End_Timestamp",
        }
        with open(corresponding_counter_file, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            assert fieldnames is not None, f"No columns in {corresponding_counter_file}"
            for column in expected_counter_columns:
                assert column in fieldnames, (
                    f"Column '{column}' missing in {corresponding_counter_file}"
                )
            found_row = False
            for row in reader:
                found_row = True

                assert row["Correlation_Id"], (
                    f"Empty Correlation_Id in {corresponding_counter_file}"
                )

                assert row["Kernel_Name"], (
                    f"Empty Kernel_Name in {corresponding_counter_file}"
                )

                assert row["Counter_Name"], (
                    f"Empty Counter_Name in {corresponding_counter_file}"
                )

                assert row["Start_Timestamp"], (
                    f"Empty Start_Timestamp in {corresponding_counter_file}"
                )

                assert row["End_Timestamp"], (
                    f"Empty End_Timestamp in {corresponding_counter_file}"
                )

            assert found_row, f"{corresponding_counter_file} is empty"

    # Flush any profiling output so capsys captures only the analyze output
    capsys.readouterr()

    # ---- Verify analysis output from --list-torch-operators (checks 6–8) ----

    # 6. Analyze with --list-torch-operators succeeds
    returncode_analyze = binary_handler_analyze_rocprof_compute([
        "--experimental",
        "analyze",
        "--path",
        workload_dir,
        "--list-torch-operators",
    ])
    assert returncode_analyze == 0, "Analyze with --list-torch-operators failed"

    list_output = capsys.readouterr().out

    # 7. torch_trace directory created with consolidated.csv
    torch_trace_dir = Path(workload_dir) / "torch_trace"
    assert torch_trace_dir.exists(), "torch_trace directory not created"

    consolidated_csv = torch_trace_dir / "consolidated.csv"
    assert consolidated_csv.exists(), "consolidated.csv not found in torch_trace"

    # 8. Consolidated CSV contains hierarchy, kernel names, and counter values
    df = pd.read_csv(consolidated_csv)
    assert not df.empty, "consolidated.csv is empty"
    assert "Operator_Name" in df.columns, "Operator_Name column missing"
    hierarchy_present = (
        df["Operator_Name"].apply(lambda x: "/" in str(x) or "::" in str(x)).any()
    )
    assert hierarchy_present, "No hierarchy information in consolidated.csv"
    assert "Kernel_Name" in df.columns, "Kernel_Name missing"
    assert df["Kernel_Name"].notnull().all() and (df["Kernel_Name"] != "").all(), (
        "Empty Kernel_Name in consolidated.csv"
    )
    assert "Counter_Value" in df.columns, "Counter_Value column missing"
    assert df["Counter_Value"].notnull().all()
    assert (df["Counter_Value"] != "").all(), "Empty Counter_Value in consolidated.csv"

    # ---- Verify --list-torch-operators CLI output format (checks 9–14) ----

    # 9. Banner
    assert "PyTorch Operator Call Tree:" in list_output, "Missing banner line"

    # 10. Source-location grouping (file:line headers)
    location_headers = re.findall(
        r"^(\S+:\d+)\s+\(kernel_launches:", list_output, re.MULTILINE
    )
    assert location_headers, "No source-location headers found in output"

    # 11. Aggregated stats on tree nodes
    assert re.search(r"\(kernel_launches:\s+\d+,\s+total_duration:", list_output), (
        "No aggregated stats found in output"
    )

    # 12. Kernel IDs
    kernel_ids = re.findall(r"\(id (\d+)\)", list_output)
    assert kernel_ids, "No kernel IDs found in output"

    # 13. Kernel launch durations
    assert re.search(r"kernel_launches:\s+\d+,\s+total_duration:", list_output), (
        "No kernel duration info in output"
    )

    # 14. Source locations sorted by descending total duration
    location_durations = re.findall(
        r"^(\S+:\d+)\s+\(kernel_launches:\s+\d+,\s+total_duration:\s+([\d.]+)\s+(ms|us)\)",
        list_output,
        re.MULTILINE,
    )
    assert location_durations, "No location durations found for sort-order check"
    durations_ms = [
        float(val) if unit == "ms" else float(val) / 1000.0
        for _, val, unit in location_durations
    ]
    assert durations_ms == sorted(durations_ms, reverse=True), (
        f"Source locations not sorted by descending duration: {location_durations}"
    )

    # 15. --list-torch-operators succeeds at every --kernel-verbose level 0-4
    #     (level 5 is the baseline run above)
    for verbose_level in range(5):
        capsys.readouterr()
        rc = binary_handler_analyze_rocprof_compute([
            "--experimental",
            "analyze",
            "--path",
            workload_dir,
            "--list-torch-operators",
            "--kernel-verbose",
            str(verbose_level),
        ])
        assert rc == 0, (
            f"--list-torch-operators failed with --kernel-verbose {verbose_level}"
        )

    # ---- Verify analysis output from --torch-operator (check 16) ----

    # Analyze with --torch-operator needs --experimental flag
    returncode_analyze_relu = binary_handler_analyze_rocprof_compute([
        "--experimental",
        "analyze",
        "--path",
        workload_dir,
        "--torch-operator",
        "*relu",
    ])
    # 16. Analyze with --torch-operator *relu succeeds
    assert returncode_analyze_relu == 0, "Analyze with --torch-operator *relu failed"

    # --- Verify torch-operator cli output ---

    # 17. Multi-component pattern matches operator in the middle of hierarchy.
    #     torch.nn.functional.relu is a wrapper that delegates to torch.relu;
    #     only the leaf operator appears in consolidated trace, so we use
    #     wildcards to match through the hierarchy.
    capsys.readouterr()
    rc_exact = binary_handler_analyze_rocprof_compute([
        "--experimental",
        "analyze",
        "--path",
        workload_dir,
        "--torch-operator",
        "*/torch.nn.functional.relu/*",
    ])
    assert rc_exact == 0, (
        "Analyze with --torch-operator */torch.nn.functional.relu/* failed"
    )
    out_exact = capsys.readouterr().out
    assert "Matched PyTorch Operators" in out_exact, (
        "Expected 'Matched PyTorch Operators' header in --torch-operator output"
    )
    assert "kernel_launches" in out_exact, (
        "Expected call tree with kernel_launches stats in --torch-operator output"
    )

    # 18. Glob wildcard pattern (*relu) matches the relu operator
    capsys.readouterr()
    rc_glob = binary_handler_analyze_rocprof_compute([
        "--experimental",
        "analyze",
        "--path",
        workload_dir,
        "--torch-operator",
        "*relu",
    ])
    assert rc_glob == 0, "Analyze with --torch-operator *relu failed"
    out_glob = capsys.readouterr().out
    assert "kernel_launches" in out_glob, (
        "Glob pattern *relu should match relu operator and render call tree"
    )

    # 19. 'all' keyword matches every operator
    capsys.readouterr()
    rc_all = binary_handler_analyze_rocprof_compute([
        "--experimental",
        "analyze",
        "--path",
        workload_dir,
        "--torch-operator",
        "all",
    ])
    assert rc_all == 0, "Analyze with --torch-operator all failed"
    out_all = capsys.readouterr().out
    assert "kernel_launches" in out_all, "'all' keyword should match operators"

    # 20. --torch-operator + -k intersection succeeds and renders call tree
    capsys.readouterr()
    rc_intersect = binary_handler_analyze_rocprof_compute([
        "--experimental",
        "analyze",
        "--path",
        workload_dir,
        "--torch-operator",
        "all",
        "-k",
        "0",
    ])
    assert rc_intersect == 0, "Analyze with --torch-operator all -k 0 failed"
    out_intersect = capsys.readouterr().out
    assert "Matched PyTorch Operators" in out_intersect, (
        "Expected call tree output with --torch-operator all -k 0"
    )
    assert "Torch operator filter selected" in out_intersect, (
        "Expected filter-selection log confirming -k intersection"
    )

    # 21. Non-matching pattern degrades gracefully with a warning
    capsys.readouterr()
    rc_nomatch = binary_handler_analyze_rocprof_compute([
        "--experimental",
        "analyze",
        "--path",
        workload_dir,
        "--torch-operator",
        "nonexistent_operator_xyz",
    ])
    assert rc_nomatch == 0, (
        "Analyze with non-matching --torch-operator should not crash"
    )
    out_nomatch = capsys.readouterr().out
    assert "No operators matched" in out_nomatch, (
        "Expected warning about no operators matched"
    )

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@skip_if_no_torch_gpu
@pytest.mark.torch_trace
def test_torch_trace_overhead(binary_handler_profile_rocprof_compute):
    """
    Measure overhead introduced by --torch-trace flag.
    Compares execution time with and without the flag to ensure overhead is acceptable.
    NOTE: Not included in the test suite since this requires PyTorch and GPU.
    """
    # Run WITHOUT --torch-trace (baseline)
    workload_dir_baseline = test_utils.get_output_dir(param_id="torch_trace_baseline")
    start_baseline = time.time()
    returncode_baseline = binary_handler_profile_rocprof_compute(
        config,
        workload_dir_baseline,
        [],  # No torch-trace flag
        check_success=True,
        roof=False,
        app_name="torch_test_app",
    )
    baseline_time = time.time() - start_baseline
    assert returncode_baseline == 0, "Baseline profiling failed"

    # Read baseline timestamps
    baseline_results_files = list(Path(workload_dir_baseline).glob("results_*.csv"))
    baseline_df = pd.concat(
        [pd.read_csv(f) for f in baseline_results_files], ignore_index=True
    )
    baseline_kernel_duration_total = (
        baseline_df["End_Timestamp"].max() - baseline_df["Start_Timestamp"].min()
    )
    test_utils.clean_output_dir(config["cleanup"], workload_dir_baseline)
    # Run WITH --torch-trace (requires --experimental)
    workload_dir_with_flag = test_utils.get_output_dir(param_id="torch_trace_with_flag")
    start_with_flag = time.time()
    returncode_with_flag = binary_handler_profile_rocprof_compute(
        config,
        workload_dir_with_flag,
        ["--experimental", "--torch-trace"],
        check_success=True,
        roof=False,
        app_name="torch_test_app",
    )
    with_flag_time = time.time() - start_with_flag
    assert returncode_with_flag == 0, "Profiling with torch-trace failed"
    # Read with-flag timestamps
    with_flag_results_files = list(Path(workload_dir_with_flag).glob("results_*.csv"))
    with_flag_df = pd.concat(
        [pd.read_csv(f) for f in with_flag_results_files], ignore_index=True
    )
    with_flag_kernel_duration_total = (
        with_flag_df["End_Timestamp"].max() - with_flag_df["Start_Timestamp"].min()
    )
    longest_running_kernel_baseline = (
        baseline_df["End_Timestamp"] - baseline_df["Start_Timestamp"]
    ).max()
    longest_running_kernel_with_flag = (
        with_flag_df["End_Timestamp"] - with_flag_df["Start_Timestamp"]
    ).max()
    # Calculate overheads
    longest_running_kernel_overhead = (
        (longest_running_kernel_with_flag - longest_running_kernel_baseline)
        / longest_running_kernel_baseline
    ) * 100
    wall_clock_overhead = ((with_flag_time - baseline_time) / baseline_time) * 100
    kernel_overhead = (
        (with_flag_kernel_duration_total - baseline_kernel_duration_total)
        / baseline_kernel_duration_total
    ) * 100
    print(f"\n{'=' * 70}")
    print("Performance Overhead Analysis:")
    print(f"  Longest running kernel overhead: {longest_running_kernel_overhead:.1f}%")
    print(f"  Baseline wall-clock time:     {baseline_time:.2f}s")
    print(f"  With --torch-trace time:  {with_flag_time:.2f}s")
    print(f"  Wall-clock overhead:          {wall_clock_overhead:.1f}%")
    print(f"  Baseline kernel duration:     {baseline_kernel_duration_total:.0f} ns")
    print(f"  With flag kernel duration:    {with_flag_kernel_duration_total:.0f} ns")
    print(f"  Kernel execution overhead:    {kernel_overhead:.1f}%")
    print(f"{'=' * 70}\n")

    test_utils.clean_output_dir(config["cleanup"], workload_dir_with_flag)
    # Assert overhead is reasonable (< 100% wall-clock, < 50% kernel)
    assert wall_clock_overhead < 100, (
        f"Wall-clock overhead too high: {wall_clock_overhead:.1f}%"
    )
    assert kernel_overhead < 50, (
        f"Kernel execution overhead too high: {kernel_overhead:.1f}%"
    )
    assert longest_running_kernel_overhead < 50, (
        f"longest running kernel increase too high: "
        f"{longest_running_kernel_overhead:.1f}%"
    )


@pytest.mark.multi_rank
def test_multi_rank_profiling_no_mpi_comm(binary_handler_profile_rocprof_compute):
    """
    Test multi-rank profiling of a non-MPI application.

    The fixture launches the profiling command with mpirun.
    """
    num_ranks = 2

    workload_dir = test_utils.get_output_dir()

    binary_handler_profile_rocprof_compute(config, workload_dir, num_ranks=num_ranks)

    # Check output for each rank
    for rank in range(num_ranks):
        rank_dir = Path(workload_dir) / str(rank)
        assert rank_dir.exists(), f"Rank directory {rank_dir} does not exist"

        file_dict = test_utils.check_csv_files(str(rank_dir), num_devices, num_kernels)
        if soc == "MI100":
            assert sorted(list(file_dict.keys())) == CSVS
        elif soc == "MI200":
            assert sorted(list(file_dict.keys())) == CSVS
        elif "MI300" in soc:
            assert sorted(list(file_dict.keys())) == CSVS
        elif "MI350" in soc:
            assert sorted(list(file_dict.keys())) == CSVS
        else:
            print(f"Testing isn't supported yet for {soc}")
            assert 0

        validate(
            inspect.stack()[0][3],
            str(rank_dir),
            file_dict,
        )

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.multi_rank
def test_multi_rank_profiling_mpi_comm(
    binary_handler_profile_rocprof_compute,
):
    """
    Test multi-rank profiling of an MPI application.

    The fixture launches the profiling command with mpirun.
    """
    # Skip test if mpi_aware_laplace_eqn is not available
    app_path = config.get("app_mpi_aware_laplace_eqn", [None])[0]
    if not (app_path and Path(app_path).exists()):
        pytest.skip(
            f"mpi_aware_laplace_eqn not found, skipping {inspect.stack()[0][3]}"
        )

    num_ranks = 2

    workload_dir = test_utils.get_output_dir()

    options = ["--iteration-multiplexing"]

    binary_handler_profile_rocprof_compute(
        config, workload_dir, options, app_name="app_mpi_aware_laplace_eqn", num_ranks=2
    )

    # Check output for each rank
    for rank in range(num_ranks):
        rank_dir = Path(workload_dir) / str(rank)
        assert rank_dir.exists(), f"Rank directory {rank_dir} does not exist"

        file_dict = test_utils.check_csv_files(str(rank_dir), num_devices, num_kernels)

        if soc == "MI100":
            assert sorted(list(file_dict.keys())) == CSVS
        elif soc == "MI200":
            assert sorted(list(file_dict.keys())) == CSVS
        elif "MI300" in soc:
            assert sorted(list(file_dict.keys())) == CSVS
        elif "MI350" in soc:
            assert sorted(list(file_dict.keys())) == CSVS
        else:
            print(f"Testing isn't supported yet for {soc}")
            assert 0

        validate(
            inspect.stack()[0][3],
            str(rank_dir),
            file_dict,
        )

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.multi_rank
def test_wrapped_mpi(binary_handler_profile_rocprof_compute):
    """
    Test that using MPI launchers (mpirun, mpiexec, srun, orterun) after '--'
    raises an error.
    """
    config["wrapped_mpi"] = ["mpirun", "-n", "2", "./tests/occupancy"]

    workload_dir = test_utils.get_output_dir()

    returncode = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options=[],
        check_success=False,
        app_name="wrapped_mpi",
    )

    # Should fail with exit code 1
    assert returncode == 1

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.multi_rank
def test_multi_rank_warning_application_replay(
    binary_handler_profile_rocprof_compute, monkeypatch
):
    """
    Test that a warning is printed when running a multi-rank application
    in application replay mode.
    """
    # Set MPI environment variable to simulate multi-rank
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")

    workload_dir = test_utils.get_output_dir()

    _, stdout, stderr = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        app_name="app_1",
        capture_output=True,
        check_success=False,
    )

    # Check that warning message is in output
    output = stdout + stderr
    assert "Multi-rank application detected" in output
    assert "Application replay mode" in output
    assert "--iteration-multiplexing" in output
    assert "--block" not in output
    assert "--set" in output

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.multi_rank
def test_multi_rank_no_warning_with_iteration_multiplexing(
    binary_handler_profile_rocprof_compute, monkeypatch
):
    """
    Test that no application replay warning is printed when running a
    multi-rank application with iteration multiplexing enabled.
    """
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")

    workload_dir = test_utils.get_output_dir()

    options = ["--iteration-multiplexing"]

    _, stdout, stderr = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        app_name="app_1",
        capture_output=True,
        check_success=False,
    )

    output = stdout + stderr
    assert "Multi-rank application detected" not in output
    assert "Application replay mode" not in output

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.multi_rank
def test_multi_rank_warning_pc_sampling(
    binary_handler_profile_rocprof_compute, monkeypatch
):
    """
    Test that a warning is printed when running a multi-rank application
    with PC sampling enabled.
    """
    # Set MPI environment variable to simulate multi-rank
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")

    workload_dir = test_utils.get_output_dir()

    # Enable PC sampling
    options = ["--block", "21"]

    _, stdout, stderr = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        app_name="app_1",
        capture_output=True,
        check_success=False,
    )

    # Check that PC sampling warning is in output
    output = stdout + stderr
    assert "Multi-rank application detected with PC sampling enabled" in output
    assert "--iteration-multiplexing" in output
    assert "--block" not in output
    assert "--set" in output

    test_utils.clean_output_dir(config["cleanup"], workload_dir)
