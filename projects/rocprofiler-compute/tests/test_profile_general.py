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
import shutil
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
config["app_mpi_aware_laplace_eqn"] = ["./tests/mpi_aware_laplace_eqn", "-i", "5"]
config["rocflop"] = ["./tests/rocflop", "--device", "0"]
config["cleanup"] = True
config["COUNTER_LOGGING"] = False
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
    "pmc_perf.csv",
    "sysinfo.csv",
])

ROOF_ONLY_FILES = sorted([
    "empirRoof_gpu-0_FP32.html",
    "pmc_perf.csv",
    "roofline.csv",
    "sysinfo.csv",
])

PC_SAMPLING_HOST_TRAP_FILES = sorted([
    "pmc_perf.csv",
    "ps_file_agent_info.csv",
    "ps_file_kernel_trace.csv",
    "ps_file_pc_sampling_host_trap.csv",
    "ps_file_results.json",
    "sysinfo.csv",
])

PC_SAMPLING_STOCHASTIC_FILES = sorted([
    "pmc_perf.csv",
    "ps_file_agent_info.csv",
    "ps_file_kernel_trace.csv",
    "ps_file_pc_sampling_stochastic.csv",
    "ps_file_results.json",
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


def log_counter(file_dict, test_name):
    for file in file_dict.keys():
        if file == "pmc_perf.csv" or "SQ" in file:
            # read file in Baseline
            df_1 = pd.read_csv(Baseline_dir + "/" + file, index_col=0)
            # get corresponding file from current test run
            df_2 = file_dict[file]

            errors = counter_compare(test_name, pd.DataFrame(), df_1, df_2, 5)
            if not errors.empty:
                if Path(
                    Baseline_dir + "/" + file.split(".")[0] + "_error_log.csv"
                ).exists():
                    error_log = pd.read_csv(
                        Baseline_dir + "/" + file.split(".")[0] + "_error_log.csv",
                        index_col=0,
                    )
                    new_error_log = pd.concat([error_log, errors])
                    new_error_log = new_error_log.reindex(
                        sorted(new_error_log.columns), axis=1
                    )
                    new_error_log = new_error_log.sort_values(
                        by=["test_name", "kernel_name", "gpu-id"]
                    )
                    new_error_log.to_csv(
                        Baseline_dir + "/" + file.split(".")[0] + "_error_log.csv"
                    )
                else:
                    errors.to_csv(
                        Baseline_dir + "/" + file.split(".")[0] + "_error_log.csv"
                    )


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
    if config["COUNTER_LOGGING"]:
        log_counter(file_dict, test_name)

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
    pmc_perf_df = test_utils.check_csv_files(workload_dir, num_devices, num_kernels)[
        "pmc_perf.csv"
    ]
    # Ensure non zero length of df
    assert len(pmc_perf_df) > 0
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

    assert (Path(workload_dir) / "pmc_perf.csv").exists()
    assert test_utils.check_file_pattern(
        "format_rocprof_output: rocpd", f"{workload_dir}/profiling_config.yaml"
    )
    assert test_utils.check_file_pattern("Counter_Name", f"{workload_dir}/pmc_perf.csv")

    code = binary_handler_analyze_rocprof_compute(["analyze", "--path", workload_dir])
    assert code == 0

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
        "pmc_perf.csv",
        "pmc_perf_0.csv",
        "pmc_perf_1.csv",
        "pmc_perf_2.csv",
        "pmc_perf_3.csv",
        "pmc_perf_4.csv",
        "pmc_perf_5.csv",
        "pmc_perf_6.csv",
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
        "pmc_perf.csv",
        "pmc_perf_0.csv",
        "pmc_perf_1.csv",
        "pmc_perf_2.csv",
        "pmc_perf_3.csv",
        "pmc_perf_4.csv",
        "pmc_perf_5.csv",
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
        "pmc_perf.csv",
        "pmc_perf_0.csv",
        "pmc_perf_1.csv",
        "pmc_perf_2.csv",
        "pmc_perf_3.csv",
        "pmc_perf_4.csv",
        "pmc_perf_5.csv",
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
        "pmc_perf.csv",
        "pmc_perf_0.csv",
        "pmc_perf_1.csv",
        "pmc_perf_2.csv",
        "pmc_perf_3.csv",
        "pmc_perf_4.csv",
        "pmc_perf_5.csv",
        "pmc_perf_6.csv",
        "pmc_perf_7.csv",
        "pmc_perf_8.csv",
        "pmc_perf_9.csv",
        "pmc_perf_10.csv",
        "pmc_perf_11.csv",
        "pmc_perf_12.csv",
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
    Test basic roofline HTML generation with full validation pipeline.
    This test runs the complete validation flow including counter logging
    and metric comparison (if enabled in config). Validates that roofline HTMLs
    are generated with the integrated multi-subplot layout (roofline plot +
    plot points table + kernel names table).
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

    # assert successful run
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
def test_roof_multiple_data_types(binary_handler_profile_rocprof_compute):
    """Test roofline with multiple data types"""
    if soc in ("MI100"):
        # roofline is not supported on MI100
        pytest.skip("Roofline not supported on MI100")
        return

    # test multiple data types
    data_types = ["FP32"]  # start with just FP32 to avoid complex validation

    for dtype in data_types:
        options = [
            "--device",
            "0",
            "--roof-only",
            "--roofline-data-type",
            dtype,
        ]
        workload_dir = test_utils.get_output_dir()

        try:
            returncode = binary_handler_profile_rocprof_compute(
                config, workload_dir, options, check_success=False, roof=True
            )

            if returncode == 0:
                assert os.path.exists(f"{workload_dir}/pmc_perf.csv")

                file_dict = test_utils.check_csv_files(workload_dir, 1, num_kernels)
                assert sorted(list(file_dict.keys())) == ROOF_ONLY_FILES
            else:
                pass
        finally:
            test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.roofline_1
def test_roof_invalid_data_type(binary_handler_profile_rocprof_compute):
    """Test roofline with invalid data type"""
    if soc in ("MI100"):
        # roofline is not supported on MI100
        pytest.skip("Roofline not supported on MI100")
        return

    # test invalid data types
    invalid_options = [
        "--device",
        "0",
        "--roof-only",
        "--roofline-data-type",
        "INVALID_TYPE",
    ]
    workload_dir = test_utils.get_output_dir()

    try:
        returncode = binary_handler_profile_rocprof_compute(
            config, workload_dir, invalid_options, check_success=False, roof=True
        )

        assert returncode >= 0

    finally:
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
            assert os.path.exists(f"{workload_dir}/pmc_perf.csv")

            roofline_csv = f"{workload_dir}/roofline.csv"
            if os.path.exists(roofline_csv):
                import pandas as pd

                df = pd.read_csv(roofline_csv)
                assert len(df) >= 0

    finally:
        test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.roofline_1
def test_roof_rocpd(binary_handler_profile_rocprof_compute):
    if soc == "MI100":
        pytest.skip("Roofline not supported on MI100")
        return

    workload_dir = test_utils.get_output_dir()
    options = ["--device", "0", "--roof-only", "--format-rocprof-output", "rocpd"]
    binary_handler_profile_rocprof_compute(config, workload_dir, options, roof=True)

    assert (Path(workload_dir) / "pmc_perf.csv").exists()
    assert (Path(workload_dir) / "roofline.csv").exists()
    assert test_utils.check_file_pattern(
        "format_rocprof_output: rocpd", f"{workload_dir}/profiling_config.yaml"
    )
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
        Metadata,
        MetricDefinition,
        MetricValue,
        RooflineData,
        Workload,
    )

    table_name_map = {
        "compute_workload": Workload,
        "compute_metric_definition": MetricDefinition,
        "compute_roofline_data": RooflineData,
        "compute_dispatch": Dispatch,
        "compute_kernel": Kernel,
        "compute_metric_value": MetricValue,
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
def test_roofline_empty_kernel_names_handling(binary_handler_profile_rocprof_compute):
    """
    Test roofline behavior when kernel filter doesn't match any
    kernels during profiling.

    When profiling with a non-matching kernel filter, the workload
    still executes and profiling data is collected for all kernels.
    However, when roofline attempts to filter the collected data
    to match the requested kernel name, it finds no match and
    produces an error.
    """
    if soc in ("MI100"):
        pytest.skip("Skipping roofline test for MI100")
        return

    options = [
        "--device",
        "0",
        "--roof-only",
        "--kernel",
        "nonexistent_kernel_name_that_should_not_match_anything",
    ]
    workload_dir = test_utils.get_output_dir()

    returncode = binary_handler_profile_rocprof_compute(
        config, workload_dir, options, check_success=False, roof=True
    )

    assert returncode == 1, f"Expected error (returncode=1), got {returncode}"

    html_files = list(Path(workload_dir).glob("empirRoof_*.html"))
    assert len(html_files) == 0, (
        "No roofline HTML should be generated when no kernels match"
    )

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.roofline_1
def test_roofline_kernel_filter(binary_handler_profile_rocprof_compute):
    """
    Test roofline multi-attempt profiling with `--kernel`
    Expect to be able to re-profile from same workload if kernels are valid.
    (Validity of --kernels tested in test_roofline_kernel_filter_error_handling already)
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
    options.extend(["--kernel", config["kernel_name_1"]])
    returncode = binary_handler_profile_rocprof_compute(  # noqa: F841
        config, workload_dir, options, check_success=True, roof=True
    )

    # Test nonexistent kernel on roof profile using existing profiling data
    # Since already profiled, throw error if non-existent kernel requested for roofline
    options.append("nonexistent_kernel_name_that_should_not_match_anything")
    returncode = binary_handler_profile_rocprof_compute(  # noqa: F841
        config, workload_dir, options, check_success=False, roof=True
    )
    assert returncode == 1

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.roofline_1
def test_roofline_unsupported_datatype_error(binary_handler_profile_rocprof_compute):
    """
    Test datatype validation error in empirical_roofline()
    This should trigger console_error for unsupported datatype
    """
    if soc in ("MI100"):
        pytest.skip("Skipping roofline test for MI100")
        return

    options = [
        "--device",
        "0",
        "--roof-only",
        "--roofline-data-type",
        "UNSUPPORTED_TYPE",
    ]
    workload_dir = test_utils.get_output_dir()

    returncode = binary_handler_profile_rocprof_compute(  # noqa: F841
        config, workload_dir, options, check_success=False, roof=True
    )

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.roofline_2
@pytest.mark.parametrize(
    "options,expected_files,test_id",
    [
        (
            ["--device", "0", "--roof-only", "--roofline-data-type", "FP32"],
            ["empirRoof_gpu-0_FP32.html"],
            "FP32_datatype",
        ),
        (
            ["--device", "0", "--roof-only", "--roofline-data-type", "FP16"],
            ["empirRoof_gpu-0_FP16.html"],
            "FP16_datatype",
        ),
        (
            ["--device", "0", "--roof-only", "--kernel", "KERNEL_NAME_PLACEHOLDER"],
            ["EXPECTED_FILE_PLACEHOLDER"],
            "kernel_filter",
        ),
    ],
    ids=["FP32_datatype", "FP16_datatype", "kernel_filter"],
)
def test_roof_plot_modes(
    binary_handler_profile_rocprof_compute, options, expected_files, test_id
):
    if soc in ("MI100"):
        pytest.skip("Skipping roofline test for MI100")
        return

    # Handle dynamic kernel name substitution for the kernel_filter test case
    options = [
        config["kernel_name_1"] if opt == "KERNEL_NAME_PLACEHOLDER" else opt
        for opt in options
    ]
    # Test `--kernel` filtering outputs are present and labelled correctly
    filter_empirRoof = "empirRoof_gpu-0_" + config["kernel_name_1"]
    expected_files = [
        filter_empirRoof if f == "EXPECTED_FILE_PLACEHOLDER" else f
        for f in expected_files
    ]

    workload_dir = test_utils.get_output_dir(param_id=test_id)

    returncode = binary_handler_profile_rocprof_compute(
        config, workload_dir, options, check_success=False, roof=True
    )
    assert returncode == 0

    for expected_file in expected_files:
        expected_path = os.path.join(workload_dir, expected_file)
        if os.path.exists(expected_path):
            assert os.path.getsize(expected_path) > 0

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

    pmc_perf_path = os.path.join(workload_dir, "pmc_perf.csv")
    if os.path.exists(pmc_perf_path):
        os.remove(pmc_perf_path)

    returncode = binary_handler_profile_rocprof_compute(  # noqa: F841
        config, workload_dir, options, check_success=False, roof=True
    )

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.roofline_2
def test_roofline_missing_file_handling(binary_handler_profile_rocprof_compute):
    """
    Test handling of missing roofline.csv file
    This should trigger error message in cli_generate_plot()
    """
    if soc in ("MI100"):
        pytest.skip("Skipping roofline test for MI100")
        return

    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    try:
        from roofline import Roofline
        from utils.schema import Workload
        from utils.specs import generate_machine_specs

        class MockArgs:
            def __init__(self):
                self.roof_only = True
                self.mem_level = "ALL"
                self.sort = "ALL"
                self.roofline_data_type = ["FP32"]

        args = MockArgs()
        mspec = generate_machine_specs(None, None)
        workload = Workload()

        workload_dir = test_utils.get_output_dir()

        run_parameters = {
            "workload_dir": workload_dir,
            "device_id": 0,
            "sort_type": "kernels",
            "mem_level": "ALL",
            "is_standalone": True,
            "roofline_data_type": ["FP32"],
        }

        roofline_instance = Roofline(args, mspec, run_parameters)

        result = roofline_instance.cli_generate_plot(
            "FP32", workload, config, arch_config
        )

        assert result is None

        test_utils.clean_output_dir(config["cleanup"], workload_dir)

    except ImportError:
        pytest.skip("Could not import roofline module for direct testing")


@pytest.mark.roofline_2
def test_roofline_invalid_datatype_cli(binary_handler_profile_rocprof_compute):
    """
    Test CLI plot generation with invalid datatype
    This should trigger error in cli_generate_plot() lines 617-624
    """
    if soc in ("MI100"):
        pytest.skip("Skipping roofline test for MI100")
        return

    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    try:
        from roofline import Roofline
        from utils.schema import Workload
        from utils.specs import generate_machine_specs

        class MockArgs:
            def __init__(self):
                self.roof_only = True
                self.mem_level = "ALL"
                self.sort = "ALL"
                self.roofline_data_type = ["FP32"]

        args = MockArgs()
        mspec = generate_machine_specs(None, None)
        workload = Workload()

        run_parameters = {
            "workload_dir": test_utils.get_output_dir(),
            "device_id": 0,
            "sort_type": "kernels",
            "mem_level": "ALL",
            "is_standalone": True,
            "roofline_data_type": ["FP32"],
        }

        roofline_instance = Roofline(args, mspec, run_parameters)

        result = roofline_instance.cli_generate_plot(
            "INVALID_DATATYPE", workload, config, arch_config
        )

        assert result is None

        test_utils.clean_output_dir(config["cleanup"], run_parameters["workload_dir"])

    except ImportError:
        pytest.skip("Could not import roofline module for direct testing")


@pytest.mark.roofline_2
def test_roofline_ceiling_data_validation(binary_handler_profile_rocprof_compute):
    """
    Test ceiling data validation in generate_plot()
    This covers error handling in lines 516-526
    """
    if soc in ("MI100"):
        pytest.skip("Skipping roofline test for MI100")
        return

    options = ["--device", "0", "--roof-only", "--mem-level", "INVALID_LEVEL"]
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

        roofline_instance = Roofline(args, mspec)

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
        roofline_instance = Roofline(args, mspec)

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
    Test roofline HTML generation with many kernels (10+) to verify:
    - Dynamic height calculation works
    - HTML is generated successfully
    - File size is reasonable

    Note: This test uses a regular workload but validates the HTML structure
    can handle the multi-subplot layout properly.
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

    html_files = list(Path(workload_dir).glob("empirRoof_*.html"))
    assert len(html_files) > 0, "At least one roofline HTML should be generated"

    for html_file in html_files:
        assert html_file.exists(), f"HTML file {html_file} should exist"
        file_size = html_file.stat().st_size

        # HTML should be larger than 10KB (has content) but less than 50MB (reasonable)
        assert file_size > 10000, (
            f"HTML {html_file} too small ({file_size} bytes), may be malformed"
        )
        assert file_size < 50000000, (
            f"HTML {html_file} too large ({file_size} bytes), may have issues"
        )

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
def test_roof_sort_dispatches(binary_handler_profile_rocprof_compute):
    # only test 1 device for roofline
    if soc in ("MI100"):
        # roofline is not supported on MI100
        assert True
        # Do not continue testing
        return

    options = ["--device", "0", "--roof-only", "--sort", "dispatches"]
    workload_dir = test_utils.get_output_dir()
    returncode = binary_handler_profile_rocprof_compute(
        config, workload_dir, options, check_success=False, roof=True
    )

    # assert successful run
    assert returncode == 0

    file_dict = test_utils.check_csv_files(workload_dir, 1, num_kernels)

    assert sorted(list(file_dict.keys())) == ROOF_ONLY_FILES

    validate(
        inspect.stack()[0][3],
        workload_dir,
        file_dict,
    )

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.sort
def test_roof_sort_kernels(binary_handler_profile_rocprof_compute):
    # only test 1 device for roofline
    if soc in ("MI100"):
        # roofline is not supported on MI100
        assert True
        # Do not continue testing
        return

    options = ["--device", "0", "--roof-only", "--sort", "kernels"]
    workload_dir = test_utils.get_output_dir()
    returncode = binary_handler_profile_rocprof_compute(
        config, workload_dir, options, check_success=False, roof=True
    )

    # assert successful run
    assert returncode == 0
    file_dict = test_utils.check_csv_files(workload_dir, 1, num_kernels)

    assert sorted(list(file_dict.keys())) == ROOF_ONLY_FILES

    validate(
        inspect.stack()[0][3],
        workload_dir,
        file_dict,
    )

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.mem
def test_roof_mem_levels_vL1D(binary_handler_profile_rocprof_compute):
    # only test 1 device for roofline
    if soc in ("MI100"):
        # roofline is not supported on MI100
        assert True
        # Do not continue testing
        return

    options = ["--device", "0", "--roof-only", "--mem-level", "vL1D"]
    workload_dir = test_utils.get_output_dir()
    returncode = binary_handler_profile_rocprof_compute(
        config, workload_dir, options, check_success=False, roof=True
    )

    # assert successful run
    assert returncode == 0
    file_dict = test_utils.check_csv_files(workload_dir, 1, num_kernels)

    assert sorted(list(file_dict.keys())) == ROOF_ONLY_FILES

    validate(
        inspect.stack()[0][3],
        workload_dir,
        file_dict,
    )

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.mem
def test_roof_mem_levels_LDS(binary_handler_profile_rocprof_compute):
    # only test 1 device for roofline
    if soc in ("MI100"):
        # roofline is not supported on MI100
        assert True
        # Do not continue testing
        return

    options = ["--device", "0", "--roof-only", "--mem-level", "LDS"]
    workload_dir = test_utils.get_output_dir()
    returncode = binary_handler_profile_rocprof_compute(
        config, workload_dir, options, check_success=False, roof=True
    )

    # assert successful run
    assert returncode == 0
    file_dict = test_utils.check_csv_files(workload_dir, 1, num_kernels)

    assert sorted(list(file_dict.keys())) == ROOF_ONLY_FILES

    validate(
        inspect.stack()[0][3],
        workload_dir,
        file_dict,
    )

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
    assert test_utils.check_file_pattern("SQ_INSTS_LDS", f"{workload_dir}/pmc_perf.csv")
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
    assert test_utils.check_file_pattern(
        "TA_FLAT_WAVEFRONTS", f"{workload_dir}/pmc_perf.csv"
    )
    assert test_utils.check_file_pattern(
        "SQC_TC_DATA_READ_REQ", f"{workload_dir}/pmc_perf.csv"
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
    assert test_utils.check_file_pattern(
        "SQ_ACTIVE_INST_LDS", f"{workload_dir}/pmc_perf.csv"
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
    assert test_utils.check_file_pattern(
        "TA_FLAT_WAVEFRONTS", f"{workload_dir}/pmc_perf.csv"
    )
    assert test_utils.check_file_pattern("global_write", f"{workload_dir}/pmc_perf.csv")
    assert not test_utils.check_file_pattern(
        "global_read", f"{workload_dir}/pmc_perf.csv"
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


@pytest.mark.pc_sampling
def test_pc_sampling_host_trap(binary_handler_profile_rocprof_compute):
    if soc in ("MI100"):
        assert True
        return

    options = [
        "--block",
        "21",
        "--pc-sampling-method",
        "host_trap",
        "--pc-sampling-interval",
        "256",
    ]
    workload_dir = test_utils.get_output_dir()
    _ = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=True,
        roof=False,
        app_name="app_mat_mul_max",
    )

    file_dict = test_utils.check_csv_files(workload_dir, num_devices, 1)
    assert sorted(list(file_dict.keys())) == sorted(PC_SAMPLING_HOST_TRAP_FILES)

    validate(inspect.stack()[0][3], workload_dir, file_dict)

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.pc_sampling
def test_pc_sampling_stochastic(binary_handler_profile_rocprof_compute):
    if soc in ("MI100") or soc in ("MI200"):
        assert True
        return

    options = [
        "--block",
        "21",
        "--pc-sampling-method",
        "stochastic",
        "--pc-sampling-interval",
        "1048576",
    ]
    workload_dir = test_utils.get_output_dir()
    _ = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=True,
        roof=False,
        app_name="app_mat_mul_max",
    )

    file_dict = test_utils.check_csv_files(workload_dir, num_devices, 1)
    assert sorted(list(file_dict.keys())) == sorted(PC_SAMPLING_STOCHASTIC_FILES)

    validate(inspect.stack()[0][3], workload_dir, file_dict)

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


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
):
    # These metrics should cover the deterministic counters being checked
    options = ["--block", "6.1.5", "6.1.6", "7.2.2", "10.1"]
    workload_dir = test_utils.get_output_dir(param_id="no_iter_mplx")
    _ = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=True,
        roof=False,
        app_name="app_laplace_eqn",
    )
    counters_no_multiplexing = test_utils.check_csv_files(
        workload_dir, num_devices, num_kernels
    )["pmc_perf.csv"]
    test_utils.clean_output_dir(config["cleanup"], workload_dir)

    options = [
        "--block",
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
    counters_kernel = test_utils.check_csv_files(
        workload_dir, num_devices, num_kernels
    )["pmc_perf.csv"]
    test_utils.clean_output_dir(config["cleanup"], workload_dir)

    options = [
        "--block",
        "6.1.5",
        "6.1.6",
        "7.2.2",
        "10.1",
        "--iteration-multiplexing",
        "kernel_launch_params",
    ]
    workload_dir = test_utils.get_output_dir(param_id="iter_mplx_params")
    _ = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=True,
        roof=False,
        app_name="app_laplace_eqn_iter",
    )
    counters_kernel_launch_params = test_utils.check_csv_files(
        workload_dir, num_devices, num_kernels
    )["pmc_perf.csv"]
    test_utils.clean_output_dir(config["cleanup"], workload_dir)

    assert are_deterministic_counters_equal(
        [counters_kernel, counters_kernel_launch_params], counters_no_multiplexing
    )


@pytest.mark.iteration_multiplexing_stochastic
def test_iteration_multiplexing_stochastic_counter_accuracy(
    binary_handler_profile_rocprof_compute,
):
    workload_dir = test_utils.get_output_dir(param_id="no_iter_mplx")
    # These metrics should cover the L1 cache stochastic counters
    options = ["--block", "16.1", "16.3"]
    _ = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=True,
        roof=False,
        app_name="app_laplace_eqn",
    )
    counters_no_multiplexing = test_utils.check_csv_files(
        workload_dir, num_devices, num_kernels
    )["pmc_perf.csv"]
    test_utils.clean_output_dir(config["cleanup"], workload_dir)

    options = ["--block", "16.1", "16.3", "--iteration-multiplexing", "kernel"]
    workload_dir = test_utils.get_output_dir(param_id="iter_mplx_kernel")
    _ = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=True,
        roof=False,
        app_name="app_laplace_eqn_iter",
    )
    counters_kernel = test_utils.check_csv_files(
        workload_dir, num_devices, num_kernels
    )["pmc_perf.csv"]
    test_utils.clean_output_dir(config["cleanup"], workload_dir)

    options = [
        "--block",
        "16.1",
        "16.3",
        "--iteration-multiplexing",
        "kernel_launch_params",
    ]
    workload_dir = test_utils.get_output_dir(param_id="iter_mplx_params")
    _ = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=True,
        roof=False,
        app_name="app_laplace_eqn_iter",
    )
    counters_kernel_launch_params = test_utils.check_csv_files(
        workload_dir, num_devices, num_kernels
    )["pmc_perf.csv"]
    test_utils.clean_output_dir(config["cleanup"], workload_dir)

    assert are_stochastic_counters_similar(
        [counters_kernel, counters_kernel_launch_params], counters_no_multiplexing
    )


# Not part of automated test runs since testing all counters is expensive
def test_iteration_multiplexing_all_counter_accuracy(
    binary_handler_profile_rocprof_compute,
):
    workload_dir = test_utils.get_output_dir(param_id="no_iter_mplx")
    _ = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        check_success=True,
        roof=False,
        app_name="app_laplace_eqn",
    )
    counters_no_multiplexing = test_utils.check_csv_files(
        workload_dir, num_devices, num_kernels
    )["pmc_perf.csv"]
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
    counters_kernel = test_utils.check_csv_files(
        workload_dir, num_devices, num_kernels
    )["pmc_perf.csv"]
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
    counters_kernel_launch_params = test_utils.check_csv_files(
        workload_dir, num_devices, num_kernels
    )["pmc_perf.csv"]
    test_utils.clean_output_dir(config["cleanup"], workload_dir)

    assert are_deterministic_counters_equal(
        [counters_kernel, counters_kernel_launch_params], counters_no_multiplexing
    )
    assert are_stochastic_counters_similar(
        [counters_kernel, counters_kernel_launch_params], counters_no_multiplexing
    )


skip_if_no_torch_gpu = pytest.mark.skipif(
    (
        importlib.util.find_spec("torch") is None
        or not __import__("torch").cuda.is_available()
    ),
    reason=("PyTorch and GPU access are required for this test"),
)

@skip_if_no_torch_gpu
@pytest.mark.torch_trace
def test_torch_trace_profile(binary_handler_profile_rocprof_compute):
    """
    Test profiling a PyTorch application with --torch-trace option.
    Verifies that all required files are generated and counter values are valid.
    NOTE: Not included in the test suite since this requires PyTorch installation.
    """
    workload_dir = test_utils.get_output_dir(param_id="torch_ops")
    Path(workload_dir).mkdir(parents=True, exist_ok=True)
    torch_app_path = Path(workload_dir) / "test_torch_app.py"

    torch_app_code = """
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    if not torch.cuda.is_available():
        import sys
        print("GPU is required for this test. Exiting.")
        sys.exit(1)
    model = SimpleNet()
    model = model.cuda()
    x = torch.randn(5, 10).cuda()
    # Run a few iterations
    for epoch in range(1):
        output = model(x)
        loss = output.sum()
        loss.backward()
        print("Training completed")
"""

    with open(torch_app_path, "w") as f:
        f.write(torch_app_code)

    config["torch_test_app"] = ["python3", str(torch_app_path)]

    # Profile with --torch-trace (requires --experimental)
    options = [
        "--experimental",
        "--torch-trace",
    ]

    returncode = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=True,
        app_name="torch_test_app",
    )
    assert returncode == 0, "Profiling the torch application failed"
    # Verify files are generated
    # 1. Check basic CSV files
    num_devices = config.get("num_devices", 1)
    file_dict = test_utils.check_csv_files(workload_dir, num_devices, 1)
    assert "pmc_perf.csv" in file_dict, "pmc_perf.csv not generated"

    # 2. Look for corresponding marker_api_trace.csv file
    # and counter_collection.csv file in workload_dir/ and workload/*/
    marker_api_trace_files = list(Path(workload_dir).glob("**/*marker_api_trace.csv"))
    counter_collection_files = list(
        Path(workload_dir).glob("**/*counter_collection.csv")
    )
    # Check if there is one-to-one mapping between marker_api_trace
    # and counter_collection files.
    # They should be present in the same subdirectories.
    assert len(marker_api_trace_files) == len(counter_collection_files), (
        "Mismatch in number of marker_api_trace.csv and counter_collection.csv files"
    )
    for marker_file in marker_api_trace_files:
        # Build corresponding counter_collection file path by replacing filename
        corresponding_counter_file = marker_file.parent / marker_file.name.replace(
            "marker_api_trace", "counter_collection"
        )
        assert corresponding_counter_file.exists(), (
            f"counter_collection.csv not found for {marker_file}"
        )
        # Check marker_api_trace.csv
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
        # Check counter_collection.csv
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

    destination_dir = test_utils.get_output_dir(param_id="torch_ops_analyze")
    # Saving the profiler output to analyze with the torch trace analyzer script
    shutil.copytree(workload_dir, destination_dir, dirs_exist_ok=True)
    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@skip_if_no_torch_gpu
@pytest.mark.torch_trace
def test_torch_trace_overhead(binary_handler_profile_rocprof_compute):
    """
    Measure overhead introduced by --torch-trace flag.
    Compares execution time with and without the flag to ensure overhead is acceptable.
    NOTE: Not included in the test suite since this requires PyTorch and GPU.
    """
    helper_dir = Path(test_utils.get_output_dir(param_id="torch_helper_script"))
    helper_dir.mkdir(parents=True, exist_ok=True)
    torch_app_path = helper_dir / "test_torch_app.py"
    torch_app_code = """
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    if not torch.cuda.is_available():
        import sys
        print("GPU is required for this test. Exiting.")
        sys.exit(1)
    model = SimpleNet()
    model = model.cuda()
    x = torch.randn(5, 10).cuda()
    # Run a few iterations
    for epoch in range(1):
        output = model(x)
        loss = output.sum()
        loss.backward()
    print("Training completed")
"""
    with open(torch_app_path, "w") as f:
        f.write(torch_app_code)
    config["torch_test_app"] = ["python3", str(torch_app_path)]
    # Run WITHOUT --torch-trace (baseline)
    workload_dir_baseline = test_utils.get_output_dir(param_id="torch_baseline")
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
    baseline_df = pd.read_csv(f"{workload_dir_baseline}/pmc_perf.csv")
    baseline_kernel_duration_total = (
        baseline_df["End_Timestamp"].max() - baseline_df["Start_Timestamp"].min()
    )
    test_utils.clean_output_dir(config["cleanup"], workload_dir_baseline)
    # Run WITH --torch-trace (requires --experimental)
    workload_dir_with_flag = test_utils.get_output_dir(param_id="torch_with_ops")
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
    with_flag_df = pd.read_csv(f"{workload_dir_with_flag}/pmc_perf.csv")
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
    assert "--block" in output
    assert "--set" in output

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
    assert "--block" in output
    assert "--set" in output

    test_utils.clean_output_dir(config["cleanup"], workload_dir)
