import inspect
import os
import re
import shutil
import subprocess
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import test_utils

# Globals

# TODO: MI350 What are the gpu models in MI 350 series
SUPPORTED_ARCHS = {
    "gfx906": {"mi50": ["MI50", "MI60"]},
    "gfx908": {"mi100": ["MI100"]},
    "gfx90a": {"mi200": ["MI210", "MI250", "MI250X"]},
    "gfx940": {"mi300": ["MI300A_A0"]},
    "gfx941": {"mi300": ["MI300X_A0"]},
    "gfx942": {"mi300": ["MI300A_A1", "MI300X_A1"]},
    "gfx950": {"mi350": ["MI350"]},
}

CHIP_IDS = {
    "29856": "MI300A_A1",
    "29857": "MI300X_A1",
    "29858": "MI308X",
    "30112": "MI350",
}


# --
# Runtime config options
# --

config = {}
config["kernel_name_1"] = "vecCopy"
config["app_1"] = ["./tests/vcopy", "-n", "1048576", "-b", "256", "-i", "3"]
config["cleanup"] = True
config["COUNTER_LOGGING"] = False
config["METRIC_COMPARE"] = False
config["METRIC_LOGGING"] = False

num_kernels = 3
num_devices = 1

DEFAULT_ABS_DIFF = 15
DEFAULT_REL_DIFF = 50
MAX_REOCCURING_COUNT = 28

ALL_CSVS_MI100 = sorted(
    [
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
        "timestamps.csv",
    ]
)

ALL_CSVS_MI200 = sorted(
    [
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
        "sysinfo.csv",
        "timestamps.csv",
    ]
)
ALL_CSVS_MI300 = sorted(
    [
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
        "sysinfo.csv",
        "timestamps.csv",
    ]
)
ALL_CSVS_MI350 = sorted(
    [
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
        "sysinfo.csv",
    ]
)

ROOF_ONLY_FILES = sorted(
    [
        "empirRoof_gpu-0_FP32.pdf",
        "pmc_perf.csv",
        "pmc_perf_0.csv",
        "pmc_perf_1.csv",
        "pmc_perf_2.csv",
        "roofline.csv",
        "sysinfo.csv",
        "timestamps.csv",
    ]
)

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
                        print(str(idx_1) + "[" + pmc_counter + "] diff is :" + str(diff))
        differences["kernel_name"] = [kernel_name]
        differences["test_name"] = [test_name]
        differences["gpu-id"] = [gpu_id]
        errors_pd = pd.concat([errors_pd, pd.DataFrame.from_dict(differences)])
    return errors_pd


def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if cmd[0] == "amd-smi" and p.returncode == 8:
        print("ERROR: No GPU detected. Unable to load amd-smi")
        assert 0
    return p.stdout.decode("ascii")


def gpu_soc():
    global num_devices
    ## 1) Parse arch details from rocminfo
    rocminfo = str(
        # decode with utf-8 to account for rocm-smi changes in latest rocm
        subprocess.run(
            ["rocminfo"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ).stdout.decode("utf-8")
    )
    rocminfo = rocminfo.split("\n")
    soc_regex = re.compile(r"^\s*Name\s*:\s+ ([a-zA-Z0-9]+)\s*$", re.MULTILINE)
    devices = list(filter(soc_regex.match, rocminfo))
    gpu_arch = devices[0].split()[1]

    if not gpu_arch in SUPPORTED_ARCHS.keys():
        print("Cannot find a supported arch in rocminfo")
        assert 0
    else:
        num_devices = (
            len(devices)
            if not "CI_VISIBLE_DEVICES" in os.environ
            else os.environ["CI_VISIBLE_DEVICES"]
        )

    ## 2) Parse chip id from rocminfo
    chip_id = re.compile(r"^\s*Chip ID:\s+ ([a-zA-Z0-9]+)\s*", re.MULTILINE)
    ids = list(filter(chip_id.match, rocminfo))
    for id in ids:
        chip_id = re.match(r"^[^()]+", id.split()[2]).group(0)

    ## 3) Deduce gpu model name from arch
    gpu_model = list(SUPPORTED_ARCHS[gpu_arch].keys())[0].upper()
    # For testing purposes we only care about gpu model series not the specific model
    # if gpu_model not in ("MI50", "MI100", "MI200"):
    #     if chip_id in CHIP_IDS:
    #         gpu_model = CHIP_IDS[chip_id]

    return gpu_model


soc = gpu_soc()

# Set rocprofv2 as profiler if MI300
if "ROCPROF" not in os.environ.keys():
    if soc == "MI100":
        os.environ["ROCPROF"] = "rocprof"

    else:
        os.environ["ROCPROF"] = "rocprofv3"


def using_v3():
    return "ROCPROF" not in os.environ.keys() or (
        "ROCPROF" in os.environ.keys()
        and (
            os.environ["ROCPROF"].endswith("rocprofv3")
            or os.environ["ROCPROF"] == "rocprofiler-sdk"
        )
    )


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
                + r")(?: *)([()0-9A-Za-z- ]+ )(?: *)([0-9.-]*)(?: *)([0-9.-]*)(?: *)\(([-0-9.]*)%\)(?: *)([-0-9.e]*)",
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

                        new_error = pd.DataFrame.from_dict(
                            {
                                "Index": [metric_idx],
                                "Metric": [metric_name],
                                "Percent Difference": [relative_diff],
                                "Absolute Difference": [absolute_diff],
                                "Baseline": [baseline_val],
                                "Current": [current_val],
                                "Test Name": [test_name],
                            }
                        )
                        error_df = pd.concat([error_df, new_error])
                        counts = error_df.groupby(["Index"]).cumcount()
                        reoccurring_metrics = error_df.loc[counts > MAX_REOCCURING_COUNT]
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


# --
# Start of profiling tests
# --


@pytest.mark.misc
def test_path(binary_handler_profile_rocprof_compute):
    workload_dir = test_utils.get_output_dir()
    binary_handler_profile_rocprof_compute(config, workload_dir)

    file_dict = test_utils.check_csv_files(workload_dir, num_devices, num_kernels)

    if soc == "MI100":
        assert sorted(list(file_dict.keys())) == ALL_CSVS_MI100
    elif soc == "MI200":
        assert sorted(list(file_dict.keys())) == sorted(
            [f for f in ALL_CSVS_MI200 if f != "timestamps.csv"]
            if using_v3()
            else ALL_CSVS_MI200
        )
    elif "MI300" in soc:
        assert sorted(list(file_dict.keys())) == sorted(
            [f for f in ALL_CSVS_MI300 if f != "timestamps.csv"]
            if using_v3()
            else ALL_CSVS_MI300
        )
    elif "MI350" in soc:
        assert sorted(list(file_dict.keys())) == sorted(ALL_CSVS_MI350)
    else:
        print("This test is not supported for {}".format(soc))
        assert 0

    validate(inspect.stack()[0][3], workload_dir, file_dict)

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.misc
def test_roof_kernel_names(binary_handler_profile_rocprof_compute):
    if soc in ("MI100"):
        # roofline is not supported on MI100
        assert True
        # Do not continue testing
        return

    options = ["--device", "0", "--roof-only", "--kernel-names"]
    workload_dir = test_utils.get_output_dir()
    returncode = binary_handler_profile_rocprof_compute(
        config, workload_dir, options, check_success=False, roof=True
    )

    # assert successful run
    assert returncode == 0
    file_dict = test_utils.check_csv_files(workload_dir, 1, num_kernels)
    if soc == "MI100":
        assert sorted(list(file_dict.keys())) == ALL_CSVS_MI100
    else:
        assert sorted(list(file_dict.keys())) == sorted(
            (
                [f for f in ROOF_ONLY_FILES if f != "timestamps.csv"]
                if using_v3()
                else ROOF_ONLY_FILES
            )
            + ["kernelName_legend.pdf"]
        )

    validate(
        inspect.stack()[0][3],
        workload_dir,
        file_dict,
    )

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.misc
def test_device_filter(binary_handler_profile_rocprof_compute):
    options = ["--device", "0"]
    workload_dir = test_utils.get_output_dir()
    binary_handler_profile_rocprof_compute(config, workload_dir, options)

    file_dict = test_utils.check_csv_files(workload_dir, 1, num_kernels)
    if soc == "MI100":
        assert sorted(list(file_dict.keys())) == ALL_CSVS_MI100
    elif soc == "MI200":
        assert sorted(list(file_dict.keys())) == sorted(
            [f for f in ALL_CSVS_MI200 if f != "timestamps.csv"]
            if using_v3()
            else ALL_CSVS_MI200
        )
    elif "MI300" in soc:
        assert sorted(list(file_dict.keys())) == sorted(
            [f for f in ALL_CSVS_MI300 if f != "timestamps.csv"]
            if using_v3()
            else ALL_CSVS_MI300
        )
    elif "MI350" in soc:
        assert sorted(list(file_dict.keys())) == sorted(ALL_CSVS_MI350)
    else:
        print("Testing isn't supported yet for {}".format(soc))
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
        assert sorted(list(file_dict.keys())) == ALL_CSVS_MI100
    elif soc == "MI200":
        assert sorted(list(file_dict.keys())) == sorted(
            [f for f in ALL_CSVS_MI200 if f != "timestamps.csv"]
            if using_v3()
            else ALL_CSVS_MI200
        )
    elif "MI300" in soc:
        assert sorted(list(file_dict.keys())) == sorted(
            [f for f in ALL_CSVS_MI300 if f != "timestamps.csv"]
            if using_v3()
            else ALL_CSVS_MI300
        )
    elif "MI350" in soc:
        assert sorted(list(file_dict.keys())) == sorted(ALL_CSVS_MI350)
    else:
        print("Testing isn't supported yet for {}".format(soc))
        assert 0

    validate(
        inspect.stack()[0][3],
        workload_dir,
        file_dict,
    )

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.block
def test_block_SQ(binary_handler_profile_rocprof_compute):
    options = ["--block", "SQ"]
    workload_dir = test_utils.get_output_dir()
    binary_handler_profile_rocprof_compute(config, workload_dir, options)

    file_dict = test_utils.check_csv_files(workload_dir, num_devices, num_kernels)
    expected_csvs = [
        "SQ_IFETCH_LEVEL.csv",
        "SQ_INST_LEVEL_LDS.csv",
        "SQ_INST_LEVEL_SMEM.csv",
        "SQ_INST_LEVEL_VMEM.csv",
        "SQ_LEVEL_WAVES.csv",
        "pmc_perf.csv",
        "pmc_perf_0.csv",
        "pmc_perf_1.csv",
        "pmc_perf_2.csv",
        "sysinfo.csv",
        "timestamps.csv",
    ]
    if soc == "MI200" or "MI300" in soc:
        expected_csvs = [
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
            "sysinfo.csv",
            "timestamps.csv",
        ]

        if using_v3():
            expected_csvs = [
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
                "sysinfo.csv",
            ]

    if soc == "MI350":
        expected_csvs = [
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
            "sysinfo.csv",
        ]

    assert sorted(list(file_dict.keys())) == sorted(expected_csvs)

    validate(
        inspect.stack()[0][3],
        workload_dir,
        file_dict,
    )

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.block
def test_block_SQC(binary_handler_profile_rocprof_compute):
    options = ["--block", "SQC"]
    workload_dir = test_utils.get_output_dir()
    binary_handler_profile_rocprof_compute(config, workload_dir, options)

    file_dict = test_utils.check_csv_files(workload_dir, num_devices, num_kernels)
    expected_csvs = [
        "pmc_perf.csv",
        "pmc_perf_0.csv",
        "pmc_perf_1.csv",
        "pmc_perf_2.csv",
        "sysinfo.csv",
        "timestamps.csv",
    ]
    if using_v3() or soc == "MI350":
        expected_csvs.remove("timestamps.csv")

    assert sorted(list(file_dict.keys())) == sorted(expected_csvs)

    validate(
        inspect.stack()[0][3],
        workload_dir,
        file_dict,
    )

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.block
def test_block_TA(binary_handler_profile_rocprof_compute):
    options = ["--block", "TA"]
    workload_dir = test_utils.get_output_dir()
    binary_handler_profile_rocprof_compute(config, workload_dir, options)

    file_dict = test_utils.check_csv_files(workload_dir, num_devices, num_kernels)
    expected_csvs = [
        "pmc_perf.csv",
        "pmc_perf_0.csv",
        "pmc_perf_1.csv",
        "pmc_perf_2.csv",
        "pmc_perf_3.csv",
        "pmc_perf_4.csv",
        "pmc_perf_5.csv",
        "pmc_perf_6.csv",
        "pmc_perf_7.csv",
        "sysinfo.csv",
        "timestamps.csv",
    ]
    if using_v3() or soc == "MI350":
        expected_csvs.remove("timestamps.csv")

    assert sorted(list(file_dict.keys())) == sorted(expected_csvs)

    validate(
        inspect.stack()[0][3],
        workload_dir,
        file_dict,
    )

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.block
def test_block_TD(binary_handler_profile_rocprof_compute):
    options = ["--block", "TD"]
    workload_dir = test_utils.get_output_dir()
    binary_handler_profile_rocprof_compute(config, workload_dir, options)

    file_dict = test_utils.check_csv_files(workload_dir, num_devices, num_kernels)
    expected_csvs = [
        "pmc_perf.csv",
        "pmc_perf_0.csv",
        "pmc_perf_1.csv",
        "pmc_perf_2.csv",
        "sysinfo.csv",
        "timestamps.csv",
    ]
    if soc == "MI200" or "MI300" in soc:
        expected_csvs = [
            "pmc_perf.csv",
            "pmc_perf_0.csv",
            "pmc_perf_1.csv",
            "pmc_perf_2.csv",
            "pmc_perf_3.csv",
            "sysinfo.csv",
            "timestamps.csv",
        ]
        if using_v3():
            expected_csvs = [
                "pmc_perf.csv",
                "pmc_perf_0.csv",
                "pmc_perf_1.csv",
                "pmc_perf_2.csv",
                "pmc_perf_3.csv",
                "sysinfo.csv",
            ]
    if soc == "MI350":
        expected_csvs = [
            "pmc_perf.csv",
            "pmc_perf_0.csv",
            "pmc_perf_1.csv",
            "pmc_perf_2.csv",
            "pmc_perf_3.csv",
            "sysinfo.csv",
        ]

    assert sorted(list(file_dict.keys())) == sorted(expected_csvs)

    validate(
        inspect.stack()[0][3],
        workload_dir,
        file_dict,
    )

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.block
def test_block_TCP(binary_handler_profile_rocprof_compute):
    options = ["--block", "TCP"]
    workload_dir = test_utils.get_output_dir()
    binary_handler_profile_rocprof_compute(config, workload_dir, options)

    file_dict = test_utils.check_csv_files(workload_dir, num_devices, num_kernels)
    expected_csvs = [
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
        "sysinfo.csv",
        "timestamps.csv",
    ]

    if using_v3():
        expected_csvs = [
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
            "sysinfo.csv",
        ]

    if soc == "MI100" or soc == "MI200":
        expected_csvs = [
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
            "sysinfo.csv",
            "timestamps.csv",
        ]

        if using_v3():
            expected_csvs = [
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
                "sysinfo.csv",
            ]

    if soc == "MI350":
        expected_csvs.remove("timestamps.csv")

    assert sorted(list(file_dict.keys())) == sorted(expected_csvs)

    validate(
        inspect.stack()[0][3],
        workload_dir,
        file_dict,
    )

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.block
def test_block_TCC(binary_handler_profile_rocprof_compute):
    options = ["--block", "TCC"]
    workload_dir = test_utils.get_output_dir()
    binary_handler_profile_rocprof_compute(config, workload_dir, options)

    file_dict = test_utils.check_csv_files(workload_dir, num_devices, num_kernels)
    expected_csvs = [
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
        "sysinfo.csv",
        "timestamps.csv",
    ]

    if using_v3():
        expected_csvs = [
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
            "sysinfo.csv",
        ]

    if soc == "MI100" or soc == "MI200":
        expected_csvs = [
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
            "sysinfo.csv",
            "timestamps.csv",
        ]

        if using_v3():
            expected_csvs = [
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
                "sysinfo.csv",
            ]

    if soc == "MI350":
        expected_csvs.remove("timestamps.csv")

    assert sorted(list(file_dict.keys())) == sorted(expected_csvs)

    validate(
        inspect.stack()[0][3],
        workload_dir,
        file_dict,
    )

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.block
def test_block_SPI(binary_handler_profile_rocprof_compute):
    options = ["--block", "SPI"]
    workload_dir = test_utils.get_output_dir()
    binary_handler_profile_rocprof_compute(config, workload_dir, options)

    file_dict = test_utils.check_csv_files(workload_dir, num_devices, num_kernels)
    expected_csvs = [
        "pmc_perf.csv",
        "pmc_perf_0.csv",
        "pmc_perf_1.csv",
        "pmc_perf_2.csv",
        "pmc_perf_3.csv",
        "pmc_perf_4.csv",
        "pmc_perf_5.csv",
        "pmc_perf_6.csv",
        "pmc_perf_7.csv",
        "sysinfo.csv",
        "timestamps.csv",
    ]

    if using_v3():
        expected_csvs = [
            "pmc_perf.csv",
            "pmc_perf_0.csv",
            "pmc_perf_1.csv",
            "pmc_perf_2.csv",
            "pmc_perf_3.csv",
            "pmc_perf_4.csv",
            "pmc_perf_5.csv",
            "pmc_perf_6.csv",
            "pmc_perf_7.csv",
            "sysinfo.csv",
        ]

    if soc == "MI350":
        expected_csvs = [
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
            "sysinfo.csv",
        ]

    assert sorted(list(file_dict.keys())) == sorted(expected_csvs)

    validate(
        inspect.stack()[0][3],
        workload_dir,
        file_dict,
    )

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.block
def test_block_CPC(binary_handler_profile_rocprof_compute):
    options = ["--block", "CPC"]
    workload_dir = test_utils.get_output_dir()
    binary_handler_profile_rocprof_compute(config, workload_dir, options)

    file_dict = test_utils.check_csv_files(workload_dir, num_devices, num_kernels)
    expected_csvs = [
        "pmc_perf.csv",
        "pmc_perf_0.csv",
        "pmc_perf_1.csv",
        "pmc_perf_2.csv",
        "pmc_perf_3.csv",
        "pmc_perf_4.csv",
        "sysinfo.csv",
        "timestamps.csv",
    ]

    if using_v3():
        expected_csvs = [
            "pmc_perf.csv",
            "pmc_perf_0.csv",
            "pmc_perf_1.csv",
            "pmc_perf_2.csv",
            "pmc_perf_3.csv",
            "pmc_perf_4.csv",
            "sysinfo.csv",
        ]

    if soc == "MI350":
        expected_csvs = [
            "pmc_perf.csv",
            "pmc_perf_0.csv",
            "pmc_perf_1.csv",
            "pmc_perf_2.csv",
            "pmc_perf_3.csv",
            "pmc_perf_4.csv",
            "pmc_perf_5.csv",
            "pmc_perf_6.csv",
            "pmc_perf_7.csv",
            "sysinfo.csv",
        ]

    assert sorted(list(file_dict.keys())) == sorted(expected_csvs)

    validate(inspect.stack()[0][3], workload_dir, file_dict)

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.block
def test_block_CPF(binary_handler_profile_rocprof_compute):
    options = ["--block", "CPF"]
    workload_dir = test_utils.get_output_dir()
    binary_handler_profile_rocprof_compute(config, workload_dir, options)

    file_dict = test_utils.check_csv_files(workload_dir, num_devices, num_kernels)
    expected_csvs = [
        "pmc_perf.csv",
        "pmc_perf_0.csv",
        "pmc_perf_1.csv",
        "pmc_perf_2.csv",
        "pmc_perf_3.csv",
        "sysinfo.csv",
        "timestamps.csv",
    ]
    if using_v3() or soc == "MI350":
        expected_csvs.remove("timestamps.csv")
    assert sorted(list(file_dict.keys())) == sorted(expected_csvs)

    validate(
        inspect.stack()[0][3],
        workload_dir,
        file_dict,
    )

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.block
def test_block_SQ_CPC(binary_handler_profile_rocprof_compute):
    options = ["--block", "SQ", "CPC"]
    workload_dir = test_utils.get_output_dir()
    binary_handler_profile_rocprof_compute(config, workload_dir, options)

    file_dict = test_utils.check_csv_files(workload_dir, num_devices, num_kernels)
    expected_csvs = [
        "SQ_IFETCH_LEVEL.csv",
        "SQ_INST_LEVEL_LDS.csv",
        "SQ_INST_LEVEL_SMEM.csv",
        "SQ_INST_LEVEL_VMEM.csv",
        "SQ_LEVEL_WAVES.csv",
        "pmc_perf.csv",
        "pmc_perf_0.csv",
        "pmc_perf_1.csv",
        "pmc_perf_2.csv",
        "sysinfo.csv",
        "timestamps.csv",
    ]
    if soc == "MI200" or "MI300" in soc:
        expected_csvs = [
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
            "sysinfo.csv",
            "timestamps.csv",
        ]

        if using_v3():
            expected_csvs = [
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
                "sysinfo.csv",
            ]

    if soc == "MI350":
        expected_csvs = [
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
            "sysinfo.csv",
        ]

    assert sorted(list(file_dict.keys())) == sorted(expected_csvs)

    validate(
        inspect.stack()[0][3],
        workload_dir,
        file_dict,
    )

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.block
def test_block_SQ_TA(binary_handler_profile_rocprof_compute):
    options = ["--block", "SQ", "TA"]
    workload_dir = test_utils.get_output_dir()
    binary_handler_profile_rocprof_compute(config, workload_dir, options)

    file_dict = test_utils.check_csv_files(workload_dir, num_devices, num_kernels)
    expected_csvs = [
        "SQ_IFETCH_LEVEL.csv",
        "SQ_INST_LEVEL_LDS.csv",
        "SQ_INST_LEVEL_SMEM.csv",
        "SQ_INST_LEVEL_VMEM.csv",
        "SQ_LEVEL_WAVES.csv",
        "pmc_perf.csv",
        "pmc_perf_0.csv",
        "pmc_perf_1.csv",
        "pmc_perf_2.csv",
        "sysinfo.csv",
        "timestamps.csv",
    ]
    if soc == "MI200" or "MI300" in soc:
        expected_csvs = [
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
            "sysinfo.csv",
            "timestamps.csv",
        ]

        if using_v3():
            expected_csvs = [
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
                "sysinfo.csv",
            ]

    if soc == "MI350":
        expected_csvs = [
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
            "sysinfo.csv",
        ]

    assert sorted(list(file_dict.keys())) == sorted(expected_csvs)

    validate(inspect.stack()[0][3], workload_dir, file_dict)

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.block
def test_block_SQ_SPI(binary_handler_profile_rocprof_compute):
    options = ["--block", "SQ", "SPI"]
    workload_dir = test_utils.get_output_dir()
    binary_handler_profile_rocprof_compute(config, workload_dir, options)

    file_dict = test_utils.check_csv_files(workload_dir, num_devices, num_kernels)
    expected_csvs = [
        "SQ_IFETCH_LEVEL.csv",
        "SQ_INST_LEVEL_LDS.csv",
        "SQ_INST_LEVEL_SMEM.csv",
        "SQ_INST_LEVEL_VMEM.csv",
        "SQ_LEVEL_WAVES.csv",
        "pmc_perf.csv",
        "pmc_perf_0.csv",
        "pmc_perf_1.csv",
        "pmc_perf_2.csv",
        "sysinfo.csv",
        "timestamps.csv",
    ]
    if soc == "MI200" or "MI300" in soc:
        expected_csvs = [
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
            "sysinfo.csv",
            "timestamps.csv",
        ]

        if using_v3():
            expected_csvs = [
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
                "sysinfo.csv",
            ]

    if soc == "MI350":
        expected_csvs = [
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
            "sysinfo.csv",
        ]

    assert sorted(list(file_dict.keys())) == sorted(expected_csvs)

    validate(
        inspect.stack()[0][3],
        workload_dir,
        file_dict,
    )
    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.block
def test_block_SQ_SQC_TCP_CPC(binary_handler_profile_rocprof_compute):
    options = ["--block", "SQ", "SQC", "TCP", "CPC"]
    workload_dir = test_utils.get_output_dir()
    binary_handler_profile_rocprof_compute(config, workload_dir, options)

    file_dict = test_utils.check_csv_files(workload_dir, num_devices, num_kernels)
    expected_csvs = [
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
        "sysinfo.csv",
        "timestamps.csv",
    ]
    if soc == "MI200" or "MI300" in soc:
        expected_csvs = [
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
            "sysinfo.csv",
            "timestamps.csv",
        ]

        if using_v3():
            expected_csvs = [
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
                "sysinfo.csv",
            ]

    if soc == "MI350":
        expected_csvs = [
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
            "sysinfo.csv",
        ]

    assert sorted(list(file_dict.keys())) == sorted(expected_csvs)

    validate(inspect.stack()[0][3], workload_dir, file_dict)

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.block
def test_block_SQ_SPI_TA_TCC_CPF(binary_handler_profile_rocprof_compute):
    options = ["--block", "SQ", "SPI", "TA", "TCC", "CPF"]
    workload_dir = test_utils.get_output_dir()
    binary_handler_profile_rocprof_compute(config, workload_dir, options)

    file_dict = test_utils.check_csv_files(workload_dir, num_devices, num_kernels)
    expected_csvs = [
        "SQ_IFETCH_LEVEL.csv",
        "SQ_INST_LEVEL_LDS.csv",
        "SQ_INST_LEVEL_SMEM.csv",
        "SQ_INST_LEVEL_VMEM.csv",
        "SQ_LEVEL_WAVES.csv",
        "pmc_perf.csv",
        "pmc_perf_0.csv",
        "pmc_perf_1.csv",
        "pmc_perf_2.csv",
        "sysinfo.csv",
        "timestamps.csv",
    ]

    if soc == "MI100":
        expected_csvs = [
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
            "timestamps.csv",
        ]

    if soc == "MI200" or "MI300" in soc:
        expected_csvs = [
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
            "sysinfo.csv",
            "timestamps.csv",
        ]

        if using_v3():
            expected_csvs = [
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
                "sysinfo.csv",
            ]

    if soc == "MI350":
        expected_csvs = [
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
            "sysinfo.csv",
        ]

    assert sorted(list(file_dict.keys())) == sorted(expected_csvs)

    validate(
        inspect.stack()[0][3],
        workload_dir,
        file_dict,
    )

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.dispatch
def test_dispatch_0(binary_handler_profile_rocprof_compute):
    options = ["--dispatch", "0"]
    workload_dir = test_utils.get_output_dir()
    binary_handler_profile_rocprof_compute(config, workload_dir, options)

    file_dict = test_utils.check_csv_files(workload_dir, num_devices, 1)
    if soc == "MI100":
        assert sorted(list(file_dict.keys())) == ALL_CSVS_MI100
    elif soc == "MI200":
        assert sorted(list(file_dict.keys())) == sorted(
            [f for f in ALL_CSVS_MI200 if f != "timestamps.csv"]
            if using_v3()
            else ALL_CSVS_MI200
        )
    elif "MI300" in soc:
        assert sorted(list(file_dict.keys())) == sorted(
            [f for f in ALL_CSVS_MI300 if f != "timestamps.csv"]
            if using_v3()
            else ALL_CSVS_MI300
        )
    elif "MI350" in soc:
        assert sorted(list(file_dict.keys())) == sorted(ALL_CSVS_MI350)
    else:
        print("Testing isn't supported yet for {}".format(soc))
        assert 0

    validate(
        inspect.stack()[0][3],
        workload_dir,
        file_dict,
        [
            "--dispatch",
            "0",
        ],
    )

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.dispatch
def test_dispatch_0_1(binary_handler_profile_rocprof_compute):
    options = ["--dispatch", "0:2"]
    workload_dir = test_utils.get_output_dir()
    binary_handler_profile_rocprof_compute(config, workload_dir, options)

    file_dict = test_utils.check_csv_files(workload_dir, num_devices, 2)
    if soc == "MI100":
        assert sorted(list(file_dict.keys())) == ALL_CSVS_MI100
    elif soc == "MI200":
        assert sorted(list(file_dict.keys())) == sorted(
            [f for f in ALL_CSVS_MI200 if f != "timestamps.csv"]
            if using_v3()
            else ALL_CSVS_MI200
        )
    elif "MI300" in soc:
        assert sorted(list(file_dict.keys())) == sorted(
            [f for f in ALL_CSVS_MI300 if f != "timestamps.csv"]
            if using_v3()
            else ALL_CSVS_MI300
        )
    elif "MI350" in soc:
        assert sorted(list(file_dict.keys())) == sorted(ALL_CSVS_MI350)
    else:
        print("Testing isn't supported yet for {}".format(soc))
        assert 0

    validate(
        inspect.stack()[0][3],
        workload_dir,
        file_dict,
        ["--dispatch", "0", "1"],
    )

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.dispatch
def test_dispatch_2(binary_handler_profile_rocprof_compute):
    options = ["--dispatch", "0"]
    workload_dir = test_utils.get_output_dir()
    binary_handler_profile_rocprof_compute(config, workload_dir, options)

    file_dict = test_utils.check_csv_files(workload_dir, num_devices, 1)
    if soc == "MI100":
        assert sorted(list(file_dict.keys())) == ALL_CSVS_MI100
    elif soc == "MI200":
        assert sorted(list(file_dict.keys())) == sorted(
            [f for f in ALL_CSVS_MI200 if f != "timestamps.csv"]
            if using_v3()
            else ALL_CSVS_MI200
        )
    elif "MI300" in soc:
        assert sorted(list(file_dict.keys())) == sorted(
            [f for f in ALL_CSVS_MI300 if f != "timestamps.csv"]
            if using_v3()
            else ALL_CSVS_MI300
        )
    elif "MI350" in soc:
        assert sorted(list(file_dict.keys())) == sorted(ALL_CSVS_MI350)
    else:
        print("Testing isn't supported yet for {}".format(soc))
        assert 0

    validate(
        inspect.stack()[0][3],
        workload_dir,
        file_dict,
        [
            "--dispatch",
            "0",
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
        assert sorted(list(file_dict.keys())) == ALL_CSVS_MI100
    elif soc == "MI200":
        assert sorted(list(file_dict.keys())) == sorted(
            [f for f in ALL_CSVS_MI200 if f != "timestamps.csv"]
            if using_v3()
            else ALL_CSVS_MI200
        )
    elif "MI300" in soc:
        assert sorted(list(file_dict.keys())) == sorted(
            [f for f in ALL_CSVS_MI300 if f != "timestamps.csv"]
            if using_v3()
            else ALL_CSVS_MI300
        )
    elif "MI350" in soc:
        assert sorted(list(file_dict.keys())) == sorted(ALL_CSVS_MI350)
    else:
        print("Testing isn't supported yet for {}".format(soc))
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
        assert sorted(list(file_dict.keys())) == ALL_CSVS_MI100
    elif soc == "MI200":
        assert sorted(list(file_dict.keys())) == sorted(
            [f for f in ALL_CSVS_MI200 if f != "timestamps.csv"]
            if using_v3()
            else ALL_CSVS_MI200
        )
    elif "MI300" in soc:
        assert sorted(list(file_dict.keys())) == sorted(
            [f for f in ALL_CSVS_MI300 if f != "timestamps.csv"]
            if using_v3()
            else ALL_CSVS_MI300
        )
    elif "MI350" in soc:
        assert sorted(list(file_dict.keys())) == sorted(ALL_CSVS_MI350)
    else:
        print("Testing isn't supported yet for {}".format(soc))
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
    assert (
        sorted(list(file_dict.keys()))
        == [f for f in ROOF_ONLY_FILES if f != "timestamps.csv"]
        if using_v3()
        else ROOF_ONLY_FILES
    )

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

    assert (
        sorted(list(file_dict.keys()))
        == [f for f in ROOF_ONLY_FILES if f != "timestamps.csv"]
        if using_v3()
        else ROOF_ONLY_FILES
    )

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

    assert (
        sorted(list(file_dict.keys()))
        == [f for f in ROOF_ONLY_FILES if f != "timestamps.csv"]
        if using_v3()
        else ROOF_ONLY_FILES
    )

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

    assert (
        sorted(list(file_dict.keys()))
        == [f for f in ROOF_ONLY_FILES if f != "timestamps.csv"]
        if using_v3()
        else ROOF_ONLY_FILES
    )

    validate(
        inspect.stack()[0][3],
        workload_dir,
        file_dict,
    )

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.section
def test_instmix_section(binary_handler_profile_rocprof_compute):
    options = ["--block", "10"]
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
        "'10': metric_id", f"{workload_dir}/profiling_config.yaml"
    )
    assert test_utils.check_file_pattern(
        "TA_FLAT_WAVEFRONTS", f"{workload_dir}/pmc_perf.csv"
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
        "'10': metric_id", f"{workload_dir}/profiling_config.yaml"
    )
    assert test_utils.check_file_pattern(
        "'3': metric_id", f"{workload_dir}/profiling_config.yaml"
    )
    assert test_utils.check_file_pattern(
        "TA_FLAT_WAVEFRONTS", f"{workload_dir}/pmc_perf.csv"
    )
    assert test_utils.check_file_pattern(
        "SQC_TC_DATA_READ_REQ", f"{workload_dir}/pmc_perf.csv"
    )
    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.section
def test_instmix_section_TA_block(binary_handler_profile_rocprof_compute):
    options = ["--block", "10", "TA"]
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
        "'10': metric_id", f"{workload_dir}/profiling_config.yaml"
    )
    assert test_utils.check_file_pattern(
        "TA: hardware_block", f"{workload_dir}/profiling_config.yaml"
    )
    assert test_utils.check_file_pattern(
        "TA_FLAT_WAVEFRONTS", f"{workload_dir}/pmc_perf.csv"
    )
    assert not test_utils.check_file_pattern(
        "SQC_TC_DATA_READ_REQ", f"{workload_dir}/pmc_perf.csv"
    )
    assert test_utils.check_file_pattern("", f"{workload_dir}/pmc_perf.csv")
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
        "'10': metric_id", f"{workload_dir}/profiling_config.yaml"
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
    options = ["--list-metrics"]
    workload_dir = test_utils.get_output_dir()
    _ = binary_handler_profile_rocprof_compute(
        config, workload_dir, options, check_success=True, roof=False
    )
    # workload dir should be empty
    assert not os.listdir(workload_dir)
    test_utils.clean_output_dir(config["cleanup"], workload_dir)
