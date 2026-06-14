# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import importlib
import os
import pkgutil
import re
import shlex
import shutil
import time
from pathlib import Path
from typing import Any, Union, cast

import config
import utils.utils_profile_csv as csv_ops
from interface.factory import create_profile_artifact_writer
from interface.profile_artifacts import ProfilePassContext
from utils.logger import (
    console_debug,
    console_error,
    console_log,  # noqa: F401
    console_warning,  # noqa: F401
    demarcate,
)
from utils.utils_common import (
    capture_subprocess_output,
    create_temp_rocprofiler_metrics_path,
    get_rocprof_cmd,
    parse_pmc_perf,
    perform_attach_detach,
)
from vendored import yaml

_PROFILER_INTERNAL_RE = re.compile(
    r"^\[rocprofiler"  # rocprofiler-sdk and rocprofiler-compute tool messages
    r"|^[WI]\d{8}\s"  # glog-style timestamps (W/I followed by YYYYMMDD)
)


def _is_live_attach(
    profiler_options: Union[list[str], dict[str, Union[str, list[str]]]],
) -> bool:
    """Return True if the profiler options indicate a live-attach (pid) mode."""
    return (isinstance(profiler_options, list) and "--pid" in profiler_options) or (
        isinstance(profiler_options, dict)
        and profiler_options.get("ROCPROF_ATTACH_PID") is not None
    )


def _classify_output_line(line: str) -> None:
    """Log a subprocess output line at the appropriate level.

    Profiler-internal messages go to DEBUG (visible with -v).
    Everything else goes to ERROR (always visible on failure).
    """
    if _PROFILER_INTERNAL_RE.match(line):
        console_debug(line)
    else:
        console_error(line, exit=False)


def run_prof(
    fnames: Union[list[str], str],
    profiler_options: Union[list[str], dict[str, Union[str, list[str]]]],
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
    fbase = fpath.stem
    if multiple_files:
        console_debug(f"pmc files: {', '.join([Path(fname).name for fname in fnames])}")
    else:
        console_debug(f"pmc file: {fpath.name}")

    # standard rocprof options
    if get_rocprof_cmd() == "rocprofiler-sdk":
        options = cast(dict[str, Union[str, list[str]]], profiler_options).copy()
        if multiple_files:
            options["ROCPROF_COUNTERS"] = ", ".join([
                f"pmc: {' '.join(parse_pmc_perf(fname))}" for fname in fnames
            ])
        else:
            options["ROCPROF_COUNTERS"] = f"pmc: {' '.join(parse_pmc_perf(fnames))}"
        options["ROCPROF_AGENT_INDEX"] = "absolute"
    else:
        if multiple_files:
            console_error(
                "Multiple pmc files detected but rocprofv3 does not "
                "support multiple input files."
            )
            return
        default_options = ["-i", fnames]
        options = default_options + cast(list[str], profiler_options)
        options = ["-A", "absolute"] + options

    new_env = os.environ.copy()

    # Counter definitions
    with open(
        config.rocprof_compute_home
        / "rocprof_compute_soc"
        / "profile_configs"
        / "sdk_config.yaml",
        encoding="utf-8",
    ) as filename:
        sdk_config = yaml.safe_load(filename)
    # Extra counter definitions
    for fname in fnames if multiple_files else [fnames]:
        fname_path = Path(fname)
        counter_def_fname = fname_path.parent / (
            "counter_def_" + fname_path.name[len("pmc_perf_") :]
        )
        if counter_def_fname.exists():
            with open(Path(counter_def_fname), encoding="utf-8") as file:
                sdk_config["rocprofiler-sdk"]["counters"].extend(
                    yaml.safe_load(file)["rocprofiler-sdk"]["counters"]
                )
    # Set counter definitions
    new_env["ROCPROFILER_METRICS_PATH"] = create_temp_rocprofiler_metrics_path(
        sdk_config
    )
    console_debug(
        "Adding env var for counter definitions: "
        f"ROCPROFILER_METRICS_PATH={new_env['ROCPROFILER_METRICS_PATH']}"
    )

    time_1 = time.time()

    output_path = Path(workload_dir + "/out/pmc_1")
    output_path.mkdir(parents=True, exist_ok=True)

    if get_rocprof_cmd() == "rocprofiler-sdk":
        app_cmd = options.pop("APP_CMD") if "APP_CMD" in options else None
        for key, value in options.items():
            new_env[key] = value
        # Log only the os.environ delta to avoid leaking secrets in shared logs.
        env_delta = {k: v for k, v in new_env.items() if os.environ.get(k) != v}
        console_debug(f"rocprof sdk env vars: {env_delta}")

        if _is_live_attach(profiler_options):
            perform_attach_detach(new_env, options)
        else:
            if app_cmd is None:
                console_error(
                    "APP_CMD, the workload's execuatble must be provided "
                    "when not in live attach mode"
                )

            console_debug(f"rocprof sdk user provided command: {app_cmd}")
            success, output = capture_subprocess_output(
                app_cmd, new_env=new_env, profileMode=True
            )
    else:
        # print in readable format using shlex
        console_debug(f"rocprof command: {shlex.join([get_rocprof_cmd()] + options)}")
        # profile the app
        success, output = capture_subprocess_output(
            [get_rocprof_cmd()] + options, new_env=new_env, profileMode=True
        )

    time_2 = time.time()
    console_debug(
        f"Finishing subprocess of pmc file(s), the time taken is "
        f"{int((time_2 - time_1) / 60)} m {str((time_2 - time_1) % 60)} sec "
    )

    if get_rocprof_cmd() != "rocprofiler-sdk":
        # rocprofv3 with yaml input file can write out/pass_1 instead of out/pmc_1
        # Move files from out/pass_1 to out/pmc_1 if pass_1 exists
        pass_1 = Path(workload_dir) / "out" / "pass_1"
        if pass_1.exists():
            shutil.copytree(
                pass_1, Path(workload_dir) / "out" / "pmc_1", dirs_exist_ok=True
            )

    # Delete counter definition temporary directory
    if new_env.get("ROCPROFILER_METRICS_PATH"):
        shutil.rmtree(new_env["ROCPROFILER_METRICS_PATH"], ignore_errors=True)

    if (not _is_live_attach(profiler_options)) and (not success):
        for line in output.splitlines():
            stripped = line.strip()
            if stripped:
                _classify_output_line(stripped)
        console_error("Profiling execution failed.")

    pass_context = _build_profile_pass_context(
        workload_dir=workload_dir,
        fbase=fbase,
        options=options,
        torch_trace_enabled=torch_trace_enabled,
        retain_rocpd_output=retain_rocpd_output,
    )
    if format_rocprof_output not in ("rocpd", "csv"):
        console_error(f"Unknown format_rocprof_output: {format_rocprof_output}")
        return

    create_profile_artifact_writer(format_rocprof_output).finalize_pass(pass_context)


def _build_profile_pass_context(
    workload_dir: str,
    fbase: str,
    options: Union[list[str], dict[str, Any]],
    torch_trace_enabled: bool,
    retain_rocpd_output: bool,
) -> ProfilePassContext:
    profiler_command = get_rocprof_cmd()
    return ProfilePassContext(
        workload_dir=Path(workload_dir),
        fbase=fbase,
        profiler_command=profiler_command,
        using_native_tool=_uses_native_counter_collection(profiler_command, options),
        torch_trace_enabled=torch_trace_enabled,
        retain_rocpd_output=retain_rocpd_output,
        kokkos_trace_enabled=_has_kokkos_trace(options),
    )


def _uses_native_counter_collection(
    profiler_command: str,
    options: Union[list[str], dict[str, Any]],
) -> bool:
    return (
        profiler_command == "rocprofiler-sdk"
        and isinstance(options, dict)
        and options.get("ROCPROF_COUNTER_COLLECTION") == "0"
    )


def _has_kokkos_trace(options: Union[list[str], dict[str, Any]]) -> bool:
    return isinstance(options, list) and "--kokkos-trace" in options


def pc_sampling_prof(
    profiler_options: Union[list[str], dict[str, Union[str, list[str]]]],
    method: str,
    interval: int,
    workload_dir: str,
) -> None:
    """
    Run rocprof with pc sampling. Current support v3 only.
    """
    # Todo:
    #   - precheck with rocprofv3 –-list-avail

    unit = "time" if method == "host_trap" else "cycles"

    if get_rocprof_cmd() == "rocprofiler-sdk":
        options = cast(dict[str, Union[str, list[str]]], profiler_options).copy()
        options.update({
            # no counter collection for pc sampling
            "ROCPROF_COUNTER_COLLECTION": "0",
            "ROCPROF_KERNEL_TRACE": "1",
            "ROCPROF_OUTPUT_FORMAT": "csv,json",
            "ROCPROF_OUTPUT_PATH": workload_dir,
            "ROCPROF_OUTPUT_FILE_NAME": "ps_file",
            "ROCPROFILER_PC_SAMPLING_BETA_ENABLED": "1",
            "ROCPROF_PC_SAMPLING_UNIT": unit,
            "ROCPROF_PC_SAMPLING_INTERVAL": str(interval),
            "ROCPROF_PC_SAMPLING_METHOD": method,
        })
        app_cmd = options.pop("APP_CMD") if "APP_CMD" in options else None
        new_env = os.environ.copy()
        for key, value in options.items():
            new_env[key] = value
        # Log only the os.environ delta to avoid leaking secrets in shared logs.
        env_delta = {k: v for k, v in new_env.items() if os.environ.get(k) != v}
        console_debug(f"pc sampling rocprof sdk env vars: {env_delta}")

        if _is_live_attach(profiler_options):
            perform_attach_detach(new_env, options)
        else:
            if app_cmd is None:
                console_error(
                    "APP_CMD, the workload's executable must be provided "
                    "when not in live attach mode"
                )

            console_debug(f"pc sampling rocprof sdk user provided command: {app_cmd}")
            success, output = capture_subprocess_output(
                app_cmd, new_env=new_env, profileMode=True
            )
            if not success:
                console_error("PC sampling failed.")
    else:
        profiler_options_list = cast(list[str], profiler_options)

        options = [
            "--kernel-trace",
            "--pc-sampling-beta-enabled",
            "--pc-sampling-method",
            method,
            "--pc-sampling-unit",
            unit,
            "--output-format",
            "csv",
            "json",
            "--pc-sampling-interval",
            str(interval),
            "-d",
            workload_dir,
            "-o",
            "ps_file",  # TODO: sync up with the name from source in 2100_.yaml
        ]

        if _is_live_attach(profiler_options):
            try:
                pid_idx = profiler_options_list.index("--pid")
                options += ["--pid", profiler_options_list[pid_idx + 1]]
                if "--attach-duration-msec" in profiler_options_list:
                    dur_idx = profiler_options_list.index("--attach-duration-msec")
                    options += [
                        "--attach-duration-msec",
                        profiler_options_list[dur_idx + 1],
                    ]
            except (ValueError, IndexError):
                console_error(
                    "--pid or --attach-duration-msec option not found in "
                    "profiler arguments for live attach mode"
                )
        else:
            try:
                app_cmd_with_separator = profiler_options_list[
                    profiler_options_list.index("--") :
                ]
                options += app_cmd_with_separator
            except ValueError:
                console_error(
                    "APP_CMD, the workload's executable must be provided "
                    "when not in live attach mode"
                )

        console_debug(f"rocprof command: {shlex.join([get_rocprof_cmd()] + options)}")
        # profile the app
        success, output = capture_subprocess_output(
            [get_rocprof_cmd()] + options, new_env=os.environ.copy(), profileMode=True
        )
        if not success:
            console_error("PC sampling failed.")


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
    """Compatibility wrapper for CSV artifact conversion."""
    from interface.csv_data import v3_counter_csv_to_v2_csv as convert_counter_csv

    convert_counter_csv(counter_file, agent_info_filepath, converted_csv_file)


def convert_native_counter_collection_csv(workload_dir: str) -> None:
    """Compatibility wrapper for native counter CSV conversion."""
    from interface.csv_data import convert_native_counter_collection_csv as convert_csv

    convert_csv(workload_dir)


def process_rocprofv3_output(workload_dir: str, using_native_tool: bool) -> list[str]:
    """Compatibility wrapper for rocprofv3 CSV artifact processing."""
    from interface.csv_data import process_rocprofv3_output as process_output

    return process_output(workload_dir, using_native_tool)


@demarcate
def save_torch_trace_inputs(
    workload_dir: str,
    fbase: str,
    output_format: str = "rocpd",
) -> None:
    """Compatibility wrapper for torch trace artifact retention."""
    from interface.csv_data import save_torch_trace_inputs as save_inputs

    save_inputs(workload_dir, fbase, output_format)


@demarcate
def process_kokkos_trace_output(workload_dir: str, fbase: str) -> None:
    """Compatibility wrapper for Kokkos trace artifact processing."""
    from interface.csv_data import process_kokkos_trace_output as process_output

    process_output(workload_dir, fbase)
