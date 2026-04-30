# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import argparse
import re
import shlex
import shutil
import sys
import time
from abc import abstractmethod
from pathlib import Path
from typing import Any, Optional, Union

from rocprof_compute_soc.soc_base import OmniSoC_Base
from utils.logger import (
    console_debug,
    console_error,
    console_log,
    console_warning,
    demarcate,
)
from utils.native_tool_finder import NativeToolFinder
from utils.utils_common import (
    format_time,
    get_job_rank_and_size,
    is_only_pc_sampling,
    print_status,
)
from utils.utils_exceptions import (
    ExecutableNotFoundError,
    NoScriptInCommandError,
    PythonScriptNotFoundError,
)
from utils.utils_profile import gen_sysinfo, pc_sampling_prof, run_prof
from vendored import yaml


def _find_python_script_index(argv: list[str]) -> tuple[Optional[int], Optional[str]]:
    """Locate the script argument in a Python command, skipping interpreter flags.

    Returns (script_index, skip_flag).  skip_flag is the flag string ("-c"/"-m")
    when injection should be skipped, or None when a script position was found
    (or no arguments remain).
    """
    skip_next = False
    for i, token in enumerate(argv[1:], start=1):
        if skip_next:
            skip_next = False
            continue
        if token in ("-c", "-m"):
            return None, token
        if token in ("-W", "-X"):
            skip_next = True
            continue
        if token.startswith("-"):
            continue
        return i, None
    return None, None


def _prepare_torch_trace_injection(
    remaining: list[str],
    resolved_exec_path: Path,
    is_python: bool,
    script_index: Optional[int],
    skip_flag: Optional[str],
) -> None:
    """Rewrite the workload command to inject ROCTX markers for --torch-trace.

    Mutates *remaining* in-place.  Three cases:
      1. Explicit Python interpreter  — insert inject_roctx.py before the script.
      2. Direct .py script execution  — prepend sys.executable + inject_roctx.py.
      3. Non-Python binary            — warn and leave the command untouched.
    """
    inject_script = Path(__file__).parent.parent / "utils" / "inject_roctx.py"
    if not inject_script.exists():
        console_error(
            f"Cannot find inject_roctx.py at {inject_script}. "
            "Please verify your installation."
        )

    if is_python:
        if skip_flag:
            console_warning(
                f"Cannot inject ROCTX markers into 'python {skip_flag}' "
                "invocations. Launching workload as-is; "
                "--torch-trace may have no effect."
            )
        elif not Path(remaining[script_index]).is_file():
            raise PythonScriptNotFoundError(remaining[script_index])
        else:
            remaining.insert(script_index, str(inject_script))
    elif resolved_exec_path.suffix in (".py", ".pyw", ".pyc", ".pyo"):
        remaining.insert(0, str(inject_script))
        remaining.insert(0, sys.executable)
    else:
        console_warning(
            "Command does not look like a Python entry point, "
            "skipping ROCTX auto-injection and launching workload as-is. "
            "Ensure the binary already initializes PyTorch/ROCTX markers, "
            "otherwise --torch-trace will have no effect."
        )

    if (resolved_exec_path.parent / "_internal").is_dir():
        console_warning(
            "Workload appears to be a self-contained binary. "
            "Such bundles typically ship private ROCm/HSA libraries, which "
            "prevents --torch-trace from collecting data. "
            "Rebuild without packaging libhsa/libhip or "
            "adjust LD_LIBRARY_PATH to /opt/rocm before profiling."
        )


class RocProfCompute_Base:
    def __init__(
        self,
        args: argparse.Namespace,
        profiler_mode: str,
        soc: OmniSoC_Base,
    ) -> None:
        self.__args = args
        self.__profiler = profiler_mode
        self._soc = soc  # OmniSoC obj

    def get_args(self) -> argparse.Namespace:
        return self.__args

    def get_profiler_options(self) -> Union[list[str], dict[str, Any]]:
        """Fetch any version specific arguments required by profiler"""
        # assume no SoC specific options and return empty list by default
        return []

    @demarcate
    def sanitize(self) -> None:
        """Perform sanitization of inputs"""
        args = self.get_args()

        if (
            sum((
                bool(args.filter_blocks),
                bool(args.set_selected),
                bool(args.roof_only),
            ))
            > 1
        ):
            console_error(
                "--block, --set, and --roof-only are mutually exclusive options. "
                "Please use only one of them."
            )

        if args.no_native_tool and args.iteration_multiplexing is not None:
            console_error(
                "--no-native-tool cannot be used with --iteration-multiplexing. "
                "Please remove one of these options."
            )

        if args.attach_pid and args.iteration_multiplexing is not None:
            console_error(
                "--attach-pid cannot be used with --iteration-multiplexing. "
                "Please remove one of these options."
            )

        if getattr(args, "torch_trace", False):
            if args.attach_pid:
                console_error(
                    "--torch-trace cannot be used with --attach-pid. "
                    "Torch trace requires injecting ROCTX markers into the "
                    "workload at launch; already-running processes cannot be "
                    "instrumented. Please remove one of these options."
                )

            if args.attach_duration_msec:
                console_error(
                    "--torch-trace cannot be used with --attach-duration-msec. "
                    "--attach-duration-msec only applies to --attach-pid, which "
                    "is incompatible with --torch-trace. Please remove one of "
                    "these options."
                )

            if args.spatial_multiplexing is not None:
                console_error(
                    "--torch-trace does not yet support multi-node profiling "
                    "via --spatial-multiplexing. Please remove one of these "
                    "options."
                )

        # verify correct formatting for application binary
        args.remaining = args.remaining[1:]
        resolved_exec_path: Optional[Path] = None

        if args.remaining:
            # Validate that MPI launchers are not used after --
            MPI_LAUNCHERS = {"mpirun", "mpiexec", "srun", "orterun"}
            if Path(args.remaining[0]).name in MPI_LAUNCHERS:
                console_error(
                    f"MPI launcher '{args.remaining[0]}' cannot be used after '--'.\n"
                    "Instead, wrap rocprof-compute with the MPI launcher:\n\n"
                    f"    {args.remaining[0]} -n <ranks> rocprof-compute profile "
                    "[options] -- ./your_application\n\n"
                    "See documentation for multi-rank profiling."
                )

            # Ensure that command points to an executable
            exec_candidate = shutil.which(args.remaining[0])
            if not exec_candidate:
                raise ExecutableNotFoundError(args.remaining[0])
            resolved_exec_path = Path(exec_candidate).resolve()

            # Detect bare Python interpreter (no script, no -c/-m) regardless
            # of --torch-trace — this always hangs the profiler.
            is_python = re.match(r"^python[0-9.]*$", resolved_exec_path.name)
            script_index: Optional[int] = None
            skip_flag: Optional[str] = None
            if is_python:
                script_index, skip_flag = _find_python_script_index(args.remaining)
                if script_index is None and skip_flag is None:
                    raise NoScriptInCommandError(args.remaining)

            if getattr(args, "torch_trace", False):
                _prepare_torch_trace_injection(
                    args.remaining,
                    resolved_exec_path,
                    bool(is_python),
                    script_index,
                    skip_flag,
                )
            args.remaining = shlex.join(args.remaining)
        elif not args.attach_pid:
            console_error(
                "Profiling command required. Pass application executable after -- "
                "at the end of options.\n"
                "\ti.e. rocprof-compute profile -n vcopy -- "
                "./vcopy -n 1048576 -b 256"
            )

    # ----------------------------------------------------
    # Required methods to be implemented by child classes
    # ----------------------------------------------------
    @abstractmethod
    def pre_processing(self) -> None:
        """Perform any pre-processing steps prior to profiling."""
        args = self.get_args()
        console_debug("profiling", f"pre-processing using {self.__profiler} profiler")

        if args.attach_pid:
            args.remaining = ""

        self._filter_blocks = self._soc.profiling_setup()

        # Write profiling configuration as yaml file
        with open(f"{self.__args.output_directory}/profiling_config.yaml", "w") as f:
            args_dict = vars(self.__args)
            # Override filter_blocks when writing profiling config yaml
            args_dict["filter_blocks"] = self._filter_blocks
            args_dict["config_dir"] = str(args_dict["config_dir"])
            yaml.dump(args_dict, f)

        # verify soc compatibility
        if self.__profiler not in self._soc.get_compatible_profilers():
            console_error(
                f"{self._soc.get_arch()} is not enabled in {self.__profiler}. "
                f"Available profilers include: {self._soc.get_compatible_profilers()}"
            )

        gen_sysinfo(
            workload_dir=args.output_directory,
            app_cmd=args.remaining,
            skip_roof=args.no_roof,
            mspec=self._soc._mspec,
            soc=self._soc,
        )

    def profile(
        self,
        fnames: Union[list[Path], Path],
        options: Union[list[str], dict[str, Any]],
        total_runs: int = 1,
    ) -> float:
        args = self.get_args()

        if isinstance(fnames, list):
            console_log(
                "profiling", f"Current input files: {', '.join(map(str, fnames))}"
            )
            str_fnames = [str(fname) for fname in fnames]
        else:
            console_log("profiling", f"Current input file: {fnames}")
            str_fnames = str(fnames)

        start_time = time.time()

        if self.__profiler == "rocprofv3" or self.__profiler == "rocprofiler-sdk":
            # Only 1-run case is permitted for attach/detach
            if (isinstance(options, list) and "--pid" in options) or (
                isinstance(options, dict)
                and (options.get("ROCPROF_ATTACH_PID") is not None)
            ):
                if total_runs > 1:
                    console_error(
                        f"Cannot attach process for profiling as the requested "
                        f"performance counters exceed the collection capacity of "
                        f"single pass counter collection. The current setup of "
                        f"requested counter blocks needs {total_runs} number of "
                        f'passes. Please use "--block" or "--set" '
                        f"to adjust or reduce the requested performance metrics!"
                    )
            console_debug(f"Sending profiler options to run_prof: {options}")

            run_prof(
                fnames=str_fnames,
                profiler_options=options,
                workload_dir=args.output_directory,
                loglevel=args.loglevel,
                format_rocprof_output=args.format_rocprof_output,
                torch_trace_enabled=getattr(args, "torch_trace", False),
                retain_rocpd_output=args.retain_rocpd_output,
            )

            end_time = time.time()
            duration = end_time - start_time

            console_debug(
                f"The time of run_prof of {str_fnames} is {int(duration / 60)} min"
                f" {duration % 60} sec"
            )
            return duration
        else:
            console_error("Profiler not supported")
            return 0.0

    @abstractmethod
    def run_profiling(self, version: str, prog: str) -> None:
        """Run profiling."""
        console_debug(
            "profiling", f"performing profiling using {self.__profiler} profiler"
        )
        args = self.get_args()

        # log basic info
        console_log(f"{str(prog).title()} version: {version}")
        console_log(f"Profiler choice: {self.__profiler}")
        console_log(
            f"Output directory: "
            f"{Path(self.__args.output_directory).absolute().resolve()}"
        )
        console_log(f"Target: {self._soc._mspec.gpu_model}")
        console_log(f"Command: {args.remaining}")
        console_log(f"Kernel Selection: {args.kernel}")
        console_log(f"Dispatch Selection: {args.dispatch}")
        if self._filter_blocks:
            console_log(f"Filtered sections: {str(self._filter_blocks)}")
        else:
            console_log("Filtered sections: All")

        # Run profiling on each input file
        input_files = sorted(
            Path(args.output_directory).glob("perfmon/pmc_perf_*.yaml")
        )
        total_runs = len(input_files)

        if total_runs == 0 and is_only_pc_sampling(args.filter_blocks):
            console_log(
                "profiling",
                "No performance counters to collect -- PC sampling only mode",
            )

        msg = "Collecting Performance Counters"
        status_msg = f"{msg} (Roofline Only)" if self.__args.roof_only else msg
        print_status(status_msg)

        native_tool_path = self.__get_native_tool_path(args)
        if self.__profiler == "rocprofiler-sdk":
            options = self.get_profiler_options(native_tool_path=native_tool_path)
        else:
            options = self.get_profiler_options()

        # Compute total workload runs including PC sampling for warning check
        total_workload_runs = total_runs
        if any(block in ["21", "pc_sampling"] for block in args.filter_blocks):
            total_workload_runs += 1

        # Warn about multi-rank profiling when multiple workload runs are needed
        # Skip warning when iteration multiplexing is enabled (single application run)
        _, total_ranks = get_job_rank_and_size()
        if (
            total_workload_runs > 1
            and total_ranks is not None
            and total_ranks >= 2
            and args.iteration_multiplexing is None
        ):
            console_warning(
                "Multi-rank application detected. Application replay mode "
                "(running the workload multiple times) may fail to collect "
                "data for workloads with MPI communication. "
                "Consider using single-pass modes:\n"
                "  --iteration-multiplexing  : Collect all counters in a "
                "single application run\n"
                "  --set <name>              : Profile a predefined counter set\n"
                "See documentation for more information."
            )

        total_profiling_time = 0.0

        if args.iteration_multiplexing is not None:
            if native_tool_path is None:
                console_error(
                    "Native tool is not supported which is required for "
                    "iteration multiplexing."
                )
            console_log(
                "profiling", f"Iteration multiplexing: {args.iteration_multiplexing}"
            )
            if args.iteration_multiplexing == "kernel":
                console_warning(
                    "profiling",
                    (
                        "Each kernel should be called atleast "
                        f"{len(input_files)} times to collect all counters."
                    ),
                )
            elif args.iteration_multiplexing == "kernel_launch_params":
                console_warning(
                    "profiling",
                    (
                        "Each kernel should be called atleast "
                        f"{len(input_files)} times with the exact launch parameters "
                        "to collect all counters."
                    ),
                )

            self.profile(input_files, options)
        else:
            console_log("profiling", "Iteration multiplexing: Disabled")

            total_runs = len(input_files)
            total_profiling_time = 0.0

            for i, fname in enumerate(input_files):
                run_number = i + 1

                # Log progress and time estimation
                if i > 0:
                    avg_time = total_profiling_time / i
                    time_left_seconds = (total_runs - run_number) * avg_time
                    time_left = format_time(time_left_seconds)
                    console_log(
                        f"[Run {run_number}/{total_runs}]"
                        f"[Approximate profiling time left: {time_left}]..."
                    )
                else:
                    console_log(
                        f"[Run {run_number}/{total_runs}]"
                        "[Approximate profiling time left: "
                        "pending first measurement...]"
                    )

                duration = self.profile(fname, options, total_runs)
                total_profiling_time += duration

        # PC sampling data is only collected when block "21" is specified
        if not "21" in args.filter_blocks:
            console_warning(
                "PC sampling data collection skipped as block 21 is not specified."
            )
            return

        total_runs = len(
            list(Path(args.output_directory).glob("perfmon/pmc_perf_*.yaml"))
        )

        console_log(f"[Run {total_runs + 1}/{total_runs + 1}][PC sampling profile run]")

        start_time = time.time()
        # No native tool for pc sampling
        options = self.get_profiler_options()

        if (
            is_only_pc_sampling(args.filter_blocks)
            and self.__profiler == "rocprofiler-sdk"
            and (rocprof_output_path := getattr(options, "ROCPROF_OUTPUT_PATH", None))
            is not None
        ):
            rocprof_output_path = Path(rocprof_output_path)
            if rocprof_output_path.exists():
                shutil.rmtree(rocprof_output_path, ignore_errors=True)
                console_debug(
                    f"Removed existing ROCProf output path: {rocprof_output_path}"
                )

        pc_sampling_prof(
            profiler_options=options,
            method=args.pc_sampling_method,
            interval=args.pc_sampling_interval,
            workload_dir=args.output_directory,
        )
        end_time = time.time()

        duration = end_time - start_time
        console_debug(
            "profiling",
            f"The time of pc sampling profiling is {int(duration / 60)} m "
            f"{duration % 60} sec",
        )

    def __get_native_tool_path(self, args: argparse.Namespace) -> str | None:
        try:
            if (
                self.__is_native_tool_requested(args)  # noqa: E501
                and self.__is_native_tool_supported(args)
            ):
                compute_root_path = Path(__file__).resolve().parents[1]
                native_tool_finder = NativeToolFinder(compute_root_path)
                return str(native_tool_finder.get_collector_library_path())
            return None
        except Exception:
            console_error(
                "Failed to use native counter collection tool.\n"
                "Please ensure the native tool library is installed "
                "or source files are present."
            )

    def __is_native_tool_requested(self, args: argparse.Namespace) -> bool:
        return self.__profiler == "rocprofiler-sdk" and not args.no_native_tool

    def __is_native_tool_supported(self, args: argparse.Namespace) -> bool:
        # Native counter collection tool is only compatible with
        # rocprofiler-sdk public API for ROCm version >= 7.x.x

        # PC sampling only profile does not need native tool

        # Do not use native tool in attach
        # mode until we figure out how multiple tools can attach
        # TODO: Figure out how multiple tools can attach
        return (
            int(self._soc._mspec.rocm_version.split(".")[0]) >= 7
            and not args.attach_pid
            and not is_only_pc_sampling(args.filter_blocks)
        )

    @abstractmethod
    def post_processing(self) -> None:
        """Perform any post-processing steps prior to profiling."""
        console_debug(
            "profiling", f"performing post-processing using {self.__profiler} profiler"
        )
        self._soc.post_profiling()
