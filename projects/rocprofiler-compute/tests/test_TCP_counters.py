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
import re
import subprocess
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest
import test_utils

rocprof_compute = SourceFileLoader(
    "rocprof-compute", "src/rocprof-compute"
).load_module()

config = {}
config["vseq"] = ["./tests/vsequential_access"]
config["vrand"] = ["./tests/vrandom_access"]
config["cleanup"] = True
config["COUNTER_LOGGING"] = False
config["METRIC_COMPARE"] = False
config["METRIC_LOGGING"] = False

SUPPORTED_ARCHS = {
    "gfx940": {"mi300": ["MI300A_A0"]},
    "gfx941": {"mi300": ["MI300X_A0"]},
    "gfx942": {"mi300": ["MI300A_A1", "MI300X_A1"]},
}

MI300_CHIP_IDS = {
    "29856": "MI300A_A1",
    "29857": "MI300X_A1",
    "29858": "MI308X",
}


def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if cmd[0] == "amd-smi" and p.returncode == 8:
        print("ERROR: No GPU detected. Unable to load amd-smi")
        assert 0
    return p.stdout.decode("ascii")


def gpu_soc():
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
        return None

    ## 2) Parse chip id from rocminfo
    chip_id = re.compile(r"^\s*Chip ID:\s+ ([a-zA-Z0-9]+)\s*", re.MULTILINE)
    ids = list(filter(chip_id.match, rocminfo))
    for id in ids:
        chip_id = re.match(r"^[^()]+", id.split()[2]).group(0)

    ## 3) Deduce gpu model name from arch
    gpu_model = list(SUPPORTED_ARCHS[gpu_arch].keys())[0].upper()
    # For testing purposes we only care about gpu model series not the specific model
    # if gpu_model == "MI300":
    #     if chip_id in MI300_CHIP_IDS:
    #         gpu_model = MI300_CHIP_IDS[chip_id]
    # else:
    #     return None

    return gpu_model


def load_metrics(csv_file_path):
    """
    Reads the CSV file into a dictionary of dictionaries:
        {
            "Metric_1": {
                    "Avg": value,
                    "Min": value,
                    "Max": value,
                    "Unit": "unit"
                },
            "Metric_2": { ... },
            ...
        }
    """
    metrics_data = {}
    with open(csv_file_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)  # reads header from first line

        for row in reader:
            metric_name = row["Metric"].strip()
            metrics_data[metric_name] = {
                "Avg": float(row["Avg"]) if row["Avg"] else None,
                "Min": float(row["Min"]) if row["Min"] else None,
                "Max": float(row["Max"]) if row["Max"] else None,
                "Unit": row["Unit"].strip() if row["Unit"] else None,
            }
    return metrics_data


soc = gpu_soc()


@pytest.mark.L1_cache
def test_L1_cache_counters(
    binary_handler_profile_rocprof_compute, binary_handler_analyze_rocprof_compute
):
    if not soc or "MI300" not in soc:
        pytest.skip("Skipping L1 cache test for non-mi300 socs.")

    # set up two apps: sequential and random access
    app_names = ["vseq", "vrand"]
    options = ["-b", "16"]

    result = {}
    metrics = ["Read Req", "Write Req", "Cache Hit Rate"]
    base = Path(test_utils.get_output_dir())

    for app_name in app_names:
        workload_dir = str(base / app_name)

        # 1. profile the app
        return_code = binary_handler_profile_rocprof_compute(
            config,
            workload_dir,
            options,
            check_success=False,
            roof=False,
            app_name=app_name,
        )
        assert return_code == 0

        # 2. analyze the results
        return_code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            workload_dir,
            "-b",
            "16.3",
            "--save-dfs",
            workload_dir,
        ])
        assert return_code == 0

        # 3. save results in local

        # FIXME: customize file name to avoid hardcode
        csv_path = workload_dir + "/16.3_vL1D_cache_access_metrics.csv"
        data = load_metrics(csv_path)

        for metric in metrics:
            if app_name not in result or not isinstance(result[app_name], dict):
                result[app_name] = {}
            result[app_name][metric] = data[metric]["Avg"]

        # 4. clean local output
        test_utils.clean_output_dir(config["cleanup"], workload_dir)
    test_utils.clean_output_dir(config["cleanup"], base)

    # 5. check results are expected

    # FIXME: use a range for comparison to account for different results
    assert result["vseq"]["Cache Hit Rate"] >= result["vrand"]["Cache Hit Rate"]
    assert result["vseq"]["Read Req"] <= result["vrand"]["Read Req"]
    assert result["vseq"]["Write Req"] <= result["vrand"]["Write Req"]
