# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Helpers for integration tests: GPU probing, workload setup, and profile mocks.

Imported as `integration_common` alongside the shared `tests/common.py`, never
star-imported, so the two helper sets stay distinguishable at the call site.
`gpu_soc()` shells out to `rocminfo` and is called lazily, via the `gpu_arch` and
`soc` fixtures, so importing this module never probes the GPU.
"""

import importlib.util
import inspect
import os
import re
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import common
import pandas as pd
import pytest
import yaml
from common import SUPPORTED_ARCHS

from utils.utils_common import canonical_config_arch

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
config["triton_test_app"] = ["python3", "./tests/triton_ffn.py"]
config["torch_compile_test_app"] = ["python3", "./tests/torch_compile_triton.py"]
config["cleanup"] = True
config["METRIC_COMPARE"] = False
config["METRIC_LOGGING"] = False

arch_config = {}

num_kernels = 3
num_devices = 1

attach_detach_interval_msec_no_delay = 1000
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
    "2.1.11": {"absolute": 0, "relative": 8},
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

# SLURM rank/size env var pair used in rank-related tests
SLURM_RANK_VAR, SLURM_SIZE_VAR = "SLURM_PROCID", "SLURM_NTASKS"

# check for parallel resource allocation
common.check_resource_allocation()

# Set default profiler
os.environ["ROCPROF"] = "rocprofiler-sdk"


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

    # results_*.csv is written compressed, so accept either form. read_csv
    # infers gzip from the .gz suffix.
    def is_csv(name):
        return name.endswith(".csv") or name.endswith(".csv.gz")

    # Validate PMC data exists (profile creates pmc_perf_*.csv or results_*.csv)
    has_separate = any(
        f.startswith("pmc_perf_") and is_csv(f) for f in files_in_workload
    )
    has_results = any(f.startswith("results_") and is_csv(f) for f in files_in_workload)

    assert has_separate or has_results, (
        "Expected pmc_perf_*.csv or results_*.csv from profile mode"
    )

    # Validate row counts for PMC files (but don't add to return dict)
    for file in files_in_workload:
        is_pmc = file.startswith("pmc_perf_") or file.startswith("results_")
        if is_pmc and is_csv(file):
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
        # Skip PMC files (already validated above), compressed or not
        if file.startswith("pmc_perf_") or file.startswith("results_"):
            continue

        if file.endswith(".csv"):
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


@lru_cache(maxsize=None)
def gpu_soc():
    """Return (arch, model) from rocminfo, e.g. ('gfx942', 'MI300').

    Both are '' when no supported GPU is detected, including when rocminfo is
    absent from PATH on a CPU-only host.
    """
    # decode with utf-8 to account for rocm-smi changes in latest rocm
    try:
        rocminfo = (
            subprocess
            .run(["rocminfo"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            .stdout.decode("utf-8")
            .split("\n")
        )
    except FileNotFoundError:
        return "", ""
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

    unsupported_socs = {
        "MI100",
        "RDNA35_POINT_1",
        "RDNA35_HALO",
        "RDNA35_POINT_2",
        "RDNA35_GORGON_POINT",
    }
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


def inject_mpirun(command, num_ranks):
    """
    Wrap a command with mpirun for multi-rank execution.
    """
    mpirun_cmd = ["mpirun"]
    # Add --allow-run-as-root only when running as root
    # (needed for OpenMPI in containers)
    # This flag is OpenMPI-specific and would cause errors
    # with other MPI implementations
    if os.geteuid() == 0:
        mpirun_cmd.append("--allow-run-as-root")
    mpirun_cmd.extend(["-n", str(num_ranks)])
    return mpirun_cmd + command


def require_torch(*, gpu: bool = False) -> None:
    """Skip when PyTorch (or, with gpu=True, GPU) is unavailable."""
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")
    try:
        import torch
    except Exception as e:
        pytest.skip(f"PyTorch import failed: {type(e).__name__}: {e}")
    if gpu and not torch.cuda.is_available():
        pytest.skip("torch.cuda.is_available() is False")


def require_triton(*, gpu: bool = False) -> None:
    """Skip when Triton, or the PyTorch/GPU stack it requires, is unavailable."""
    require_torch(gpu=gpu)
    if importlib.util.find_spec("triton") is None:
        pytest.skip("Triton is not installed")
    try:
        import triton  # noqa: F401
    except Exception as e:
        pytest.skip(f"Triton import failed: {type(e).__name__}: {e}")


def get_available_sets_for_arch(gpu_arch):
    """Return available set options for the given GPU arch,
    or [] if gpu_arch is falsy."""
    if not gpu_arch:
        return []
    config_arch = canonical_config_arch(gpu_arch) or gpu_arch
    sets_file = (
        Path(common.SRC)
        / "rocprof_compute_soc"
        / "profile_configs"
        / "sets"
        / f"{config_arch}_sets.yaml"
    )
    if not sets_file.exists():
        return []
    data = yaml.safe_load(sets_file.read_text())
    return [s["set_option"] for s in data.get("sets", []) if s.get("set_option")]


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


def baseline_compare_metric(test_name, workload_dir, args=[]):
    _, soc = gpu_soc()
    baseline_dir = (Path("tests/workloads/vcopy") / soc).resolve()
    if not baseline_dir.exists():
        pytest.skip(f"Skipping test since {baseline_dir} does not exist")

    baseline_dir = str(baseline_dir)

    t = subprocess.Popen(
        [
            sys.executable,
            "src/rocprof_compute",
            "analyze",
            "--path",
            baseline_dir,
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
        if Path(baseline_dir + "/metric_error_log.csv").exists():
            error_df = pd.read_csv(
                baseline_dir + "/metric_error_log.csv",
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
                            error_df.to_csv(baseline_dir + "/metric_error_log.csv")


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
                (
                    (test_counters - test_counters.mean()) / test_counters.std(ddof=0)
                ).abs()
                for test_counters in test_counters_list
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
        return False, "Group keys do not match between baseline and test dataframes"

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

            return (
                False,
                f"{counter_name} is not equal between baseline and test dataframes",
            )

    return True, "All deterministic counters are equal"


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


def clear_rank_env(monkeypatch, *env_vars):
    """Remove the specified environment variables."""
    for key in env_vars:
        monkeypatch.delenv(key, raising=False)


def skip_unsupported_roofline_soc():
    _, soc = gpu_soc()
    if soc == "MI100":
        pytest.skip(f"Roofline is not supported on {soc}")


def is_gfx115x_soc():
    return gpu_soc()[1] in {
        "RDNA35_POINT_1",
        "RDNA35_HALO",
        "RDNA35_POINT_2",
        "RDNA35_GORGON_POINT",
    }


def is_gfx1250_soc():
    return gpu_soc()[1] == "GFX1250_SERIES"


# --
