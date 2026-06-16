# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import importlib
import pkgutil
from pathlib import Path
from typing import Any, Union

import utils.utils_profile_csv as csv_ops
from orchestrator.common import ProfilerOptions
from orchestrator.common import is_live_attach as _is_live_attach
from utils.logger import (
    console_debug,
    console_error,
    console_log,  # noqa: F401
    console_warning,  # noqa: F401
    demarcate,
)
from utils.utils_common import get_rocprof_cmd


def is_live_attach(
    profiler_options: ProfilerOptions,
) -> bool:
    """Return True if the profiler options indicate a live-attach (pid) mode."""
    return _is_live_attach(profiler_options)


def run_prof(
    fnames: Union[list[str], str],
    profiler_options: ProfilerOptions,
    workload_dir: str,
    loglevel: int,
    format_rocprof_output: str,
    torch_trace_enabled: bool = False,
    retain_rocpd_output: bool = False,
) -> None:
    multiple_files = isinstance(fnames, list)
    if multiple_files and (
        (
            isinstance(profiler_options, dict)
            and profiler_options.get("ROCPROF_ITERATION_MULTIPLEXING") is None
        )
        or (
            isinstance(profiler_options, list)
            and "--iteration-multiplexing" not in profiler_options
        )
    ):
        console_error(
            "Multiple pmc files detected but ROCPROF_ITERATION_MULTIPLEXING is not set."
        )
        return

    fpath = Path(fnames[0]) if multiple_files else Path(fnames)
    if multiple_files:
        console_debug(f"pmc files: {', '.join([Path(fname).name for fname in fnames])}")
    else:
        console_debug(f"pmc file: {fpath.name}")

    if get_rocprof_cmd() == "rocprofiler-sdk":
        from orchestrator.rocprofiler_sdk import RocprofilerSdkProfileOrchestrator

        orchestrator = RocprofilerSdkProfileOrchestrator()
    else:
        from orchestrator.rocprofv3 import Rocprofv3ProfileOrchestrator

        orchestrator = Rocprofv3ProfileOrchestrator()

    orchestrator.run_pass(
        fnames=fnames,
        profiler_options=profiler_options,
        workload_dir=workload_dir,
        format_rocprof_output=format_rocprof_output,
        torch_trace_enabled=torch_trace_enabled,
        retain_rocpd_output=retain_rocpd_output,
    )


@demarcate
def gen_sysinfo(
    workload_dir: str,
    app_cmd: str,
    skip_roof: bool,
    mspec: Any,  # noqa: ANN401
    soc: Any,  # noqa: ANN401
) -> None:
    data = mspec.get_class_members()

    # Append workload information to machine specs
    data["command"] = app_cmd
    data["workload_path"] = workload_dir

    blocks = ["SQ", "LDS", "SQC", "TA", "TD", "TCP", "TCC", "SPI", "CPC", "CPF"]
    if not skip_roof:
        blocks.append("roofline")
    data["ip_blocks"] = "|".join(blocks)

    csv_ops.write_csv_from_dicts(workload_dir + "/" + "sysinfo.csv", [data])


def get_submodules(package_name: str) -> list[str]:
    """List all submodules for a target package"""

    submodules: list[str] = []

    # walk all submodules in target package
    package = importlib.import_module(package_name)
    for _, name, _ in pkgutil.walk_packages(package.__path__):
        pretty_name = name.split("_", 1)[1].replace("_", "")
        # ignore base submodule, add all other
        if pretty_name != "base":
            submodules.append(pretty_name)

    return submodules


def v3_counter_csv_to_v2_csv(
    counter_file: str, agent_info_filepath: str, converted_csv_file: str
) -> None:
    """Compatibility wrapper for CSV profile data conversion."""
    from interface.csv_data import v3_counter_csv_to_v2_csv as convert_counter_csv

    convert_counter_csv(counter_file, agent_info_filepath, converted_csv_file)


def convert_native_counter_collection_csv(workload_dir: str) -> None:
    """Compatibility wrapper for native counter CSV conversion."""
    from interface.csv_data import convert_native_counter_collection_csv as convert_csv

    convert_csv(workload_dir)


def process_rocprofv3_output(workload_dir: str, using_native_tool: bool) -> list[str]:
    """Compatibility wrapper for rocprofv3 CSV profile data processing."""
    from interface.csv_data import process_rocprofv3_output as process_output

    return process_output(workload_dir, using_native_tool)


@demarcate
def save_torch_trace_inputs(
    workload_dir: str,
    fbase: str,
    output_format: str = "rocpd",
) -> None:
    """Compatibility wrapper for torch trace data retention."""
    from interface.csv_data import save_torch_trace_inputs as save_inputs

    save_inputs(workload_dir, fbase, output_format)


@demarcate
def process_kokkos_trace_output(workload_dir: str, fbase: str) -> None:
    """Compatibility wrapper for Kokkos trace data processing."""
    from interface.csv_data import process_kokkos_trace_output as process_output

    process_output(workload_dir, fbase)
