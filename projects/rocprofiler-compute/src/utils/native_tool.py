# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT
import shlex
from pathlib import Path

from utils.logger import console_debug, console_error, console_log
from utils.utils_common import capture_subprocess_output


class NativeTool:
    sources_dir_name = "lib"
    sources_build_subdir_name = "_build"
    sources_bin_subdir_name = "lib"
    lib_name = "librocprofiler-compute-tool.so"
    lib_relative_path = "/".join([
        sources_dir_name,
        sources_build_subdir_name,
        sources_bin_subdir_name,
        lib_name,
    ])

    def __init__(self) -> None:
        pass

    def get_collector_library_path(
        self, compute_script_path: Path, rocprofiler_sdk_tool_path: Path
    ) -> Path | None:
        console_debug("Searching for native collector.")
        console_debug(f"Compute script path: {compute_script_path}")
        console_debug(f"rocprofiler_sdk_tool_path: {rocprofiler_sdk_tool_path}")

        native_tool_path = self.__find_existing_collector(
            compute_script_path, rocprofiler_sdk_tool_path
        )
        if not native_tool_path:
            native_tool_path = self.__build_collector(compute_script_path)
        console_log(f"Using native collector: {native_tool_path}")
        return native_tool_path

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
        assert rocprofiler_sdk_tool_path.stem == "librocprofiler-sdk-tool"
        native_tool_base_path = (
            rocprofiler_sdk_tool_path.parents[2]
            if len(rocprofiler_sdk_tool_path.parents) > 2
            else Path()
        )
        return native_tool_base_path

    def __find_built_collector(self, compute_script_path: Path) -> Path | None:
        source_root = self.__get_source_root(compute_script_path)
        pattern = self.lib_relative_path
        console_log(f"Searching {source_root} by {pattern} for native collector")
        return self.__find_file_by_glob_pattern(source_root, pattern)

    def __get_source_root(self, compute_script_path: Path) -> Path:
        assert compute_script_path.stem == "rocprof-compute"
        return compute_script_path.parent

    def __find_file_by_glob_pattern(self, base_path: Path, pattern: str) -> Path | None:
        match = next(base_path.glob(pattern), None)
        return Path(match) if match is not None else None

    def __build_collector(self, compute_script_path: Path) -> Path | None:
        self._generate_cmake_project(compute_script_path.parent)
        self._build_cmake_project(compute_script_path.parent)
        return self.__find_built_collector(compute_script_path)

    def _generate_cmake_project(self, src_path: Path) -> bool:
        build_command = (
            "cmake "
            + f"-S {src_path}/{self.sources_dir_name} "
            + f"-B {src_path}/{self.sources_dir_name}/{self.sources_build_subdir_name}"
        )
        console_debug(f"Building native tool using command: {build_command}")
        return self.__execute_command(build_command)

    def _build_cmake_project(self, src_path: Path) -> bool:
        generate_command = (
            "cmake --build "
            + f"{src_path}/{self.sources_dir_name}/{self.sources_build_subdir_name}"
        )
        console_debug(
            f"Generating native tool project using command: {generate_command}"
        )
        return self.__execute_command(generate_command)

    def __execute_command(self, command: str) -> bool:
        success, output = capture_subprocess_output(shlex.split(command))
        console_debug(f"Build output: {output}")
        if not success:
            console_error("Failed to execute command: {command}")
            return False
        return True
