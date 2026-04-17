# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT
import shlex
import tempfile
from pathlib import Path

from utils.logger import console_debug, console_error
from utils.utils_common import capture_subprocess_output


class NativeTool:
    def __init__(self) -> None:
        pass

    def get_collector_library_path(
        self, rocprof_compute_script_path: Path, rocprofiler_sdk_tool_path: Path
    ) -> None:
        native_tool_base_path = (
            rocprof_compute_script_path.parents[2]
            if len(rocprof_compute_script_path.parents) >= 3
            else Path()
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
                Path(tempfile.mkdtemp(prefix="rocprofiler-compute-tool-", dir="/tmp"))
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
                f"-L {str(Path(rocprofiler_sdk_tool_path).parent.parent)} "
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
