##############################################################################
# MIT License
#
# Copyright (c) 2021 - 2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

import argparse
import shlex
import shutil
import sys
import tempfile
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
from utils.utils_common import (
    capture_subprocess_output,
    format_time,
    get_rank,
    is_only_pc_sampling,
    print_status,
)
from utils.utils_profile import gen_sysinfo, pc_sampling_prof, run_prof
from vendored import yaml


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

        # verify not accessing parent directories
        if ".." in str(args.path):
            console_error(
                "Access denied. Cannot access parent directories in path (i.e. ../)"
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
                console_error(
                    f"Your command {args.remaining[0]} doesn't point to a executable. "
                    "Please verify."
                )
            resolved_exec_path = Path(exec_candidate).resolve()

            # Appending a wrapper for injecting roctx-markers
            if getattr(args, "torch_trace", False):
                # Find the inject_roctx.py script in src/utils
                inject_script = (
                    Path(__file__).parent.parent / "utils" / "inject_roctx.py"
                )
                if not inject_script.exists():
                    console_error(
                        f"Cannot find inject_roctx.py at {inject_script}. "
                        "Please verify your installation."
                    )

                # Case 1: Explicit python command (python, python3, etc.)
                if args.remaining[0].startswith("python"):
                    # Insert inject_roctx.py after the python interpreter
                    args.remaining.insert(1, str(inject_script))
                # Case 2: Direct Python script execution (./main.py, /path/to/script.py)
                elif args.remaining[0].endswith((".py", ".pyw", ".pyc", ".pyo")):
                    # Use current Python interpreter
                    args.remaining.insert(0, str(inject_script))
                    args.remaining.insert(0, sys.executable)
                else:
                    console_warning(
                        "Command does not look like a Python entry point, "
                        "skipping ROCTX auto-injection and launching workload as-is."
                    )
                    console_warning(
                        "Ensure the binary already initializes PyTorch/ROCTX markers, "
                        "otherwise --torch-trace will have no effect."
                    )

                if (
                    resolved_exec_path
                    and (resolved_exec_path.parent / "_internal").is_dir()
                ):
                    console_warning(
                        "Workload appears to be a self-contained binary. "
                        "Such bundles typically ship private ROCm/HSA libraries, which "
                        "prevents --torch-trace from collecting data."
                        "Rebuild without packaging libhsa/libhip or "
                        "adjust LD_LIBRARY_PATH to /opt/rocm) before profiling."
                    )
            args.remaining = " ".join(args.remaining)
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
        with open(f"{self.__args.path}/profiling_config.yaml", "w") as f:
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
            workload_name=args.name,
            workload_dir=args.path,
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
                workload_dir=args.path,
                mspec=self._soc._mspec,
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
        console_log(f"Path: {Path(self.__args.path).absolute().resolve()}")
        console_log(f"Target: {self._soc._mspec.gpu_model}")
        console_log(f"Command: {args.remaining}")
        console_log(f"Kernel Selection: {args.kernel}")
        console_log(f"Dispatch Selection: {args.dispatch}")
        if self._filter_blocks:
            console_log(f"Filtered sections: {str(self._filter_blocks)}")
        else:
            console_log("Filtered sections: All")

        # Run profiling on each input file
        input_files = sorted(Path(args.path).glob("perfmon/*.txt"))
        total_runs = len(input_files)

        if total_runs == 0 and is_only_pc_sampling(args.filter_blocks):
            console_log(
                "profiling",
                "No performance counters to collect -- PC sampling only mode",
            )

        msg = "Collecting Performance Counters"
        status_msg = f"{msg} (Roofline Only)" if self.__args.roof_only else msg
        print_status(status_msg)

        native_tool_path = None
        # Native counter collection tool is only compatible with
        # rocprofiler-sdk public API for ROCm version >= 7.x.x
        # Do not use native tool in attach
        # mode until we figure out how multiple tools can attach
        # TODO: Figure out how multiple tools can attach
        if (
            self.__profiler == "rocprofiler-sdk"
            and not args.no_native_tool
            and int(self._soc._mspec.rocm_version.split(".")[0]) >= 7
            and not args.attach_pid
        ):
            # Use native counter collection tool
            # Use lib* glob pattern to handle CMAKE_INSTALL_LIBDIR variations
            # (lib, lib64, lib32, etc. depending on distribution)
            script_path = Path(sys.argv[0]).resolve()
            native_tool_base_path = (
                script_path.parents[2] if len(script_path.parents) >= 3 else Path()
            )
            native_tool_glob_pattern = (
                "lib*/rocprofiler-compute/librocprofiler-compute-tool.so"
            )
            try:
                native_tool_path = str(
                    next(native_tool_base_path.glob(native_tool_glob_pattern))
                )
            except Exception as e:
                console_debug(
                    f"Could not find pre-built native tool: {e}.\n"
                    f"Search path: {native_tool_base_path}\n"
                    f"Glob pattern: {native_tool_glob_pattern}\n"
                    "Building native tool now."
                )
                native_tool_path = None
            if not (native_tool_path and Path(native_tool_path).is_file()):
                # Build native counter collection tool if not exists
                native_tool_path = str(
                    Path(
                        tempfile.mkdtemp(prefix="rocprofiler-compute-tool-", dir="/tmp")
                    )
                    / "librocprofiler-compute-tool.so"
                )
                native_tool_cpp_path = Path(__file__).resolve().parents[1] / "lib"
                link_libraries = ("rocprofiler-sdk",)
                build_command = (
                    # Create shared object
                    "hipcc -shared -fPIC "
                    # Link with dependant libraries
                    + " ".join(f"-l{lib}" for lib in link_libraries)
                    + " "
                    # Compliler flags
                    "-std=c++17 -W -Wall -Wextra -Wshadow -O2 "
                    # rocprofiler sdk library path
                    f"-L {str(Path(args.rocprofiler_sdk_tool_path).parent.parent)} "
                    # native tool source files (tool.cpp and helper.cpp)
                    f"{native_tool_cpp_path}/"
                    "rocprofiler_compute_tool.cpp "
                    f"{native_tool_cpp_path}/"
                    "helper.cpp "
                    # temporary shared object for native tool
                    f"-o {native_tool_path}"
                )
                console_debug(f"Building native tool using command: {build_command}")
                success, output = capture_subprocess_output(shlex.split(build_command))
                console_debug(f"Build output: {output}")
                if not success:
                    console_error(
                        "Failed to use native counter collection tool.\n"
                        "Could not find pre-built .so file at: "
                        f"{native_tool_base_path / native_tool_glob_pattern}\n"
                        "Could not find source .cpp files in folder: "
                        f"{native_tool_cpp_path}\n"
                        "Please ensure the native tool library is installed "
                        "or source files are present."
                    )

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
        if (
            total_workload_runs > 1
            and get_rank() is not None
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

        for fname in input_files:
            # Kernel filtering (in-place replacement)
            if not args.kernel == None:
                success, output = capture_subprocess_output([
                    "sed",
                    "-i",
                    "-r",
                    f"s%^(kernel:).*%kernel: {','.join(self.__args.kernel)}%g",
                    str(fname),
                ])
                # log output from profile filtering
                if not success:
                    console_error(output)
                else:
                    console_debug(output)

            # Dispatch filtering (inplace replacement)
            if args.dispatch is not None:
                success, output = capture_subprocess_output([
                    "sed",
                    "-i",
                    "-r",
                    f"s%^(range:).*%range: {' '.join(self.__args.dispatch)}%g",
                    str(fname),
                ])
                # log output from profile filtering
                if not success:
                    console_error(output)
                else:
                    console_debug(output)

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

        # Delete temporary native tool if created
        if native_tool_path and native_tool_path.startswith("/tmp"):
            shutil.rmtree(Path(native_tool_path).parent, ignore_errors=True)

        # PC sampling data is only collected when block "21" is specified
        if not "21" in args.filter_blocks:
            console_warning(
                "PC sampling data collection skipped as block 21 is not specified."
            )
            return

        total_runs = len(list(Path(args.path).glob("perfmon/*.txt")))

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
            workload_dir=args.path,
        )
        end_time = time.time()

        duration = end_time - start_time
        console_debug(
            "profiling",
            f"The time of pc sampling profiling is {int(duration / 60)} m "
            f"{duration % 60} sec",
        )

    @abstractmethod
    def post_processing(self) -> None:
        """Perform any post-processing steps prior to profiling."""
        console_debug(
            "profiling", f"performing post-processing using {self.__profiler} profiler"
        )
        self._soc.post_profiling()
