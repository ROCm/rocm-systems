# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import fcntl
import importlib
import os
import pkgutil
import re
import shlex
import shutil
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional, Union, cast

import config
import utils.utils_profile_csv as csv_ops
from utils import rocpd_data
from utils.inject_roctx.constants import KNOWN_ML_API_BACKENDS
from utils.logger import (
    console_debug,
    console_error,
    console_log,
    console_warning,
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

ProfilerOptions = Union[list[str], dict[str, Union[str, list[str]]]]

# inject_roctx appends a trailing "|<backend>" suffix to marker names.
_UNKNOWN_BACKEND = "unknown"
_BACKEND_SUFFIX_RE = re.compile(
    r"\|(" + "|".join(re.escape(b) for b in KNOWN_ML_API_BACKENDS) + r")$"
)


def is_live_attach(
    profiler_options: ProfilerOptions,
) -> bool:
    """Return True if the profiler options indicate a live-attach (pid) mode."""
    return (isinstance(profiler_options, list) and "--pid" in profiler_options) or (
        isinstance(profiler_options, dict)
        and profiler_options.get("ROCPROF_ATTACH_PID") is not None
    )


def pc_sampling_unit(method: str) -> str:
    """Map a PC sampling method to its sampling unit."""
    return "time" if method == "host_trap" else "cycles"


@contextmanager
def file_lock(
    lock_path: Path,
    wait_message: str = "",
    acquired_message: str = "",
) -> Generator[None, None, None]:
    """Hold an exclusive advisory lock on a shared, multi-user lock file."""
    fd, mode = _open_shared_lock_fd(lock_path)
    with os.fdopen(fd, mode, encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            if wait_message:
                print(wait_message, flush=True)
            fcntl.flock(lock_handle, fcntl.LOCK_EX)  # blocking wait
            if acquired_message:
                print(acquired_message, flush=True)
        yield


def _open_shared_lock_fd(lock_path: Path) -> tuple[int, str]:
    """Open a shared world-rw lock file, creating it if needed.

    flock advisory locks do not require write access, so a read-only fd is
    enough to keep a legacy file owned by another user lockable.
    """
    nofollow = getattr(os, "O_NOFOLLOW", 0)  # don't open through a symlink
    cloexec = getattr(os, "O_CLOEXEC", 0)
    create_flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | nofollow | cloexec
    try:
        fd = os.open(lock_path, create_flags, 0o666)
        os.fchmod(fd, 0o666)  # fchmod defeats umask -> world-rw
        return fd, "r+"
    except FileExistsError:
        pass  # already published; open the existing file below

    try:
        return os.open(lock_path, os.O_RDWR | nofollow | cloexec), "r+"
    except PermissionError:
        pass  # foreign-owned legacy file; fall back to read-only
    except OSError as e:
        raise RuntimeError(f"Cannot open lock file {lock_path}: {e}.") from e

    try:
        return os.open(lock_path, os.O_RDONLY | nofollow | cloexec), "r"
    except OSError as e:
        raise RuntimeError(
            f"Cannot open lock file {lock_path}: {e}. A stale lock file owned "
            "by another user may exist; remove it and retry."
        ) from e


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
    profiler_options: ProfilerOptions,
    workload_dir: str,
    loglevel: int,
    ml_api_trace_enabled: bool = False,
    retain_rocpd_output: bool = False,
    extra_env: Optional[dict[str, str]] = None,
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
    if extra_env:
        new_env.update(extra_env)

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

        if is_live_attach(profiler_options):
            perform_attach_detach(new_env, options)
        else:
            if app_cmd is None:
                console_error(
                    "APP_CMD, the workload's executable must be provided "
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

    if (not is_live_attach(profiler_options)) and (not success):
        for line in output.splitlines():
            stripped = line.strip()
            if stripped:
                _classify_output_line(stripped)
        console_error("Profiling execution failed.")

    # If using native tool for counter collection
    if (
        get_rocprof_cmd() == "rocprofiler-sdk"
        and options["ROCPROF_COUNTER_COLLECTION"] == "0"
    ):
        for db_name in (Path(workload_dir) / "out/pmc_1").glob("*/*.db"):
            pid = db_name.stem.split("_")[0]
            counter_csv = (
                Path(workload_dir)
                / "out"
                / "pmc_1"
                / f"{pid}_native_counter_collection.csv"
            )
            if not counter_csv.is_file():
                console_debug(
                    f"No native counter CSV for pid {pid}; "
                    f"skipping rocpd update for {db_name}."
                )
                continue
            rocpd_data.update_rocpd_pmc_events(
                str(counter_csv),
                str(db_name),
            )
            console_debug(f"Updated rocpd db {db_name} with native tool counters.")
    # Write results_fbase.csv
    rocpd_data.convert_dbs_to_csv(
        [str(p) for p in (Path(workload_dir) / "out/pmc_1").glob("*/*.db")],
        workload_dir + f"/out/pmc_1/{fbase}_counter_collection.csv",
        workload_dir + f"/out/pmc_1/{fbase}_marker_api_trace.csv",
    )
    # Subprocess succeeded but may have dispatched zero GPU kernels,
    # in which case the CSV is missing or has no data rows.
    try:
        combined_rows, _ = csv_ops.read_csv_as_dicts(
            workload_dir + f"/out/pmc_1/{fbase}_counter_collection.csv"
        )
    except (FileNotFoundError, ValueError):
        combined_rows = []
    if not combined_rows:
        console_warning(
            "No GPU kernel data collected. "
            "The workload may not have dispatched any GPU kernels."
        )
        shutil.rmtree(f"{workload_dir}/out", ignore_errors=True)
        return
    # Reset Dispatch_ID based on PID, Kernel_Name, Grid_Size,
    # Workgroup_Size, LDS_Per_Workgroup, Start_Timestamp, End_Timestamp
    csv_ops.assign_group_ids(
        combined_rows,
        [
            "PID",
            "Kernel_Name",
            "Grid_Size",
            "Workgroup_Size",
            "LDS_Per_Workgroup",
            "Start_Timestamp",
            "End_Timestamp",
        ],
        "Dispatch_ID",
    )
    # Reset Kernel_ID based on Kernel_Name, Grid_Size,
    # Workgroup_Size, LDS_Per_Workgroup
    csv_ops.assign_group_ids(
        combined_rows,
        ["Kernel_Name", "Grid_Size", "Workgroup_Size", "LDS_Per_Workgroup"],
        "Kernel_ID",
    )
    # Drop PID since its not required
    csv_ops.drop_column_from_rows(combined_rows, "PID")
    # Write back to CSV
    csv_ops.write_csv_from_dicts(
        workload_dir + f"/out/pmc_1/{fbase}_counter_collection.csv",
        combined_rows,
    )
    csv_ops.write_csv_from_dicts(workload_dir + f"/results_{fbase}.csv", combined_rows)
    console_warning(
        "Intermediate results_*.csv generation from rocpd databases is "
        "deprecated and will be replaced with automatic .db file "
        "retention in a future release."
    )
    if ml_api_trace_enabled:
        # move counter collection and marker trace to workload dir
        save_ml_api_trace_inputs(workload_dir, fbase)
    if retain_rocpd_output:
        console_warning(
            "--retain-rocpd-output is deprecated and will be removed in "
            "a future release. .db files will be retained automatically."
        )
        for db_path in (Path(workload_dir) / "out/pmc_1").glob("*/*.db"):
            pid = db_path.stem.split("_")[0]
            shutil.copyfile(
                db_path,
                workload_dir + f"/{fbase}_{pid}.db",
            )
            console_warning(
                f"Retaining large raw rocpd database: {workload_dir}/{fbase}_{pid}.db"
            )
    # Remove temp directory
    shutil.rmtree(workload_dir + "/" + "out")


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


def _parse_function_backend(function_value: Optional[str]) -> tuple[str, str]:
    """Return (clean_function, backend) for one Function cell.

    Values with no recognized backend suffix return "unknown".
    """
    if function_value is None:
        return "", _UNKNOWN_BACKEND
    raw = str(function_value)
    match = _BACKEND_SUFFIX_RE.search(raw)
    if match is None:
        return raw, _UNKNOWN_BACKEND
    return raw[: match.start()], match.group(1)


def _augment_marker_rows(
    rows: list[dict], fieldnames: list[str]
) -> tuple[list[dict], list[str], int, list[str]]:
    """Move the wire backend suffix from the Function column into a Backend
    column.

    Returns the rows, the field names including Backend, the count of rows whose
    Function has no recognized backend suffix, and up to three sample Function
    values from those rows.
    """
    augmented_fieldnames = list(fieldnames)
    if "Backend" not in augmented_fieldnames:
        augmented_fieldnames.append("Backend")
    unknown_samples: list[str] = []
    unknown_count = 0
    for row in rows:
        clean_function, backend = _parse_function_backend(row.get("Function", ""))
        row["Function"] = clean_function
        row["Backend"] = backend
        if backend == _UNKNOWN_BACKEND:
            unknown_count += 1
            sample = clean_function or "<empty>"
            if len(unknown_samples) < 3 and sample not in unknown_samples:
                unknown_samples.append(sample)
    return rows, augmented_fieldnames, unknown_count, unknown_samples


def _augment_marker_csv(src_marker: str, dst_marker: str) -> None:
    """Copy src_marker to dst_marker, moving the wire backend suffix out of
    Function into a dedicated Backend column. Rows whose Function has no
    recognized backend suffix are tagged Backend="unknown".
    """
    rows, fieldnames = csv_ops.read_csv_as_dicts(src_marker)
    if "Function" not in fieldnames:
        # Unrecognized schema: copy verbatim.
        console_warning(
            "ml api trace",
            f"{dst_marker} has no 'Function' column (columns: {fieldnames}); "
            "copying verbatim without backend augmentation.",
        )
        shutil.copyfile(src_marker, dst_marker)
        return
    rows, augmented_fieldnames, unknown_count, unknown_samples = _augment_marker_rows(
        rows, fieldnames
    )
    csv_ops.write_csv_from_dicts(dst_marker, rows, fieldnames=augmented_fieldnames)
    if unknown_count:
        console_warning(
            "ml api trace",
            f"{unknown_count} marker row(s) in {src_marker} have no recognized "
            f"|<backend> suffix and were tagged Backend='{_UNKNOWN_BACKEND}'. "
            f"Sample Function values: {unknown_samples}.",
        )


@demarcate
def save_ml_api_trace_inputs(
    workload_dir: str,
    fbase: str,
) -> None:
    """
    Move counter_collection and marker_api_trace data to workload_dir,
    for creation of ML API trace in Analyze mode.

    Marker CSVs are augmented on copy: the trailing ``|<backend>`` suffix
    written by inject_roctx is split off Function and surfaced as a
    dedicated Backend column (torch, triton, ...).
    """
    src_dir = Path(workload_dir) / "out" / "pmc_1"
    # Only one pair expected
    src_counter = src_dir / f"{fbase}_counter_collection.csv"
    src_marker = src_dir / f"{fbase}_marker_api_trace.csv"
    dst_counter = Path(workload_dir) / f"ml_api_trace_{fbase}_counter_collection.csv"
    dst_marker = Path(workload_dir) / f"ml_api_trace_{fbase}_marker_api_trace.csv"
    # These files are expected to exist.
    shutil.copyfile(src_counter, dst_counter)
    _augment_marker_csv(str(src_marker), str(dst_marker))
    console_log(
        "ml api trace",
        "Moved counter collection and marker trace files "
        "to workload dir for ML API trace creation.",
    )
    console_log("Counter Collection: ", str(dst_counter))
    console_log("Marker API Trace: ", str(dst_marker))
