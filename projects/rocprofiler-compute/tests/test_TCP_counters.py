# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

from pathlib import Path

import common
import pandas as pd
import pytest

config = {}
config["vseq"] = ["./tests/vsequential_access"]
config["vrand"] = ["./tests/vrandom_access"]
config["cleanup"] = True
config["COUNTER_LOGGING"] = False
config["METRIC_COMPARE"] = False
config["METRIC_LOGGING"] = False


def load_metrics(csv_file_path):
    """Read workload_metric.csv and return {metric_name: {value_name: value}}."""
    df = pd.read_csv(csv_file_path)
    return df.pivot(index="metric_name", columns="value_name", values="value").to_dict(
        orient="index"
    )


_, soc = common.gpu_soc()


@pytest.mark.L1_cache
def test_L1_cache_counters(
    binary_handler_profile_rocprof_compute, binary_handler_analyze_rocprof_compute
):
    if not soc or "MI300" not in soc:
        pytest.skip("Skipping L1 cache test for non-mi300 socs.")

    # set up two apps: sequential and random access
    app_names = ["vseq", "vrand"]
    # Scope to the relevant metric section and multiplex counters across
    # kernel iterations so the app runs only once.
    options = ["-b", "16.3", "--iteration-multiplexing"]

    result = {}
    metrics = ["Read Req", "Write Req", "Cache Hit Rate"]
    base = Path(common.get_output_dir())

    for app_name in app_names:
        workload_dir = f"{base}/{app_name}"
        workload_dir_output = f"{base}_{app_name}"

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
            "--output-format",
            "csv",
            "--output-name",
            workload_dir_output,
        ])
        assert return_code == 0

        # 3. save results in local
        csv_path = workload_dir_output + "/workload_metric.csv"
        data = load_metrics(csv_path)

        for metric in metrics:
            if app_name not in result or not isinstance(result[app_name], dict):
                result[app_name] = {}
            result[app_name][metric] = data[metric]["Avg"]

        # 4. clean local output
        common.clean_output_dir(config["cleanup"], workload_dir)
        common.clean_output_dir(config["cleanup"], workload_dir_output)
    common.clean_output_dir(config["cleanup"], base)

    # 5. check results are expected

    # FIXME: use a range for comparison to account for different results
    assert result["vseq"]["Cache Hit Rate"] >= result["vrand"]["Cache Hit Rate"]
    assert result["vseq"]["Read Req"] <= result["vrand"]["Read Req"]
    assert result["vseq"]["Write Req"] <= result["vrand"]["Write Req"]
