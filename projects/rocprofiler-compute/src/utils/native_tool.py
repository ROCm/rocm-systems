# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT
import shlex
from pathlib import Path

from utils.logger import console_debug, console_error
from utils.utils_common import capture_subprocess_output


class NativeTool:
    sources_dir_name = "lib"
    sources_build_subdir_name = "_build"
    sources_bin_subdir_name = "bin"
    lib_name = "librocprofiler-compute-tool.so"
    lib_relative_path = (
        sources_dir_name
        + sources_build_subdir_name
        + sources_bin_subdir_name
        + lib_name
    )

    def __init__(self) -> None:
        pass

    def get_collector_library_path(
        self, compute_script_path: Path, rocprofiler_sdk_tool_path: Path
    ) -> Path:
        native_tool_path = self.__find_existing_collector(
            compute_script_path, rocprofiler_sdk_tool_path
        )
        if not native_tool_path:
            native_tool_path = self.__build_collector(
                compute_script_path, rocprofiler_sdk_tool_path
            )
        return Path(native_tool_path)

    def __find_existing_collector(
        self, compute_script_path: Path, rocprofiler_sdk_tool_path: Path
    ) -> Path | None:
        collector_path = self.__find_built_collector(compute_script_path)
        if not collector_path:
            collector_path = self.__find_installed_collector(rocprofiler_sdk_tool_path)
        return collector_path

    def __find_installed_collector(
        self, rocprofiler_sdk_tool_path: Path
    ) -> Path | None:
        rocm_root_path = self.__get_installed_rocm_root_path(rocprofiler_sdk_tool_path)
        pattern = f"lib*/rocprofiler-compute/{self.lib_name}"
        console_debug(f"Searching {rocm_root_path} by {pattern} for native collector")
        return self.__find_file_by_glob_pattern(rocm_root_path, pattern)

    def __get_installed_rocm_root_path(self, rocprofiler_sdk_tool_path: Path) -> Path:
        assert rocprofiler_sdk_tool_path.stem == "rocprofv3"
        native_tool_base_path = (
            rocprofiler_sdk_tool_path.parents[1]
            if len(rocprofiler_sdk_tool_path.parents) > 1
            else Path()
        )
        return native_tool_base_path

    def __find_built_collector(self, compute_script_path: Path) -> Path | None:
        source_root = self.__get_source_root(compute_script_path)
        pattern = self.lib_relative_path
        console_debug(f"Searching {source_root} by {pattern} for native collector")
        return self.__find_file_by_glob_pattern(source_root, pattern)

    def __get_source_root(self, compute_script_path: Path) -> Path:
        assert compute_script_path.stem == "rocprof-compute"
        return compute_script_path.parent

    def __find_file_by_glob_pattern(self, base_path: Path, pattern: str) -> Path | None:
        match = next(base_path.glob(pattern), None)
        return Path(match) if match is not None else None

    def __build_collector(
        self, native_tool_path: Path, rocprofiler_sdk_tool_path: Path
    ) -> Path:
        # Build native counter collection tool if not exists
        build_command = ""
        console_debug(f"Building native tool using command: {build_command}")
        success, output = capture_subprocess_output(shlex.split(build_command))
        console_debug(f"Build output: {output}")
        if not success:
            console_error(
                "Failed to use native counter collection tool.\n"
                "Please ensure the native tool library is installed "
                "or source files are present."
            )
        return native_tool_path
