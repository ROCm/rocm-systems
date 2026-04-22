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

    def __init__(self, compute_script_path: Path, sdk_tool_path: Path) -> None:
        assert sdk_tool_path.stem == "librocprofiler-sdk-tool"
        assert compute_script_path.stem == "rocprof-compute"
        console_debug("Searching for native collector.")
        console_debug(f"Compute script path: {compute_script_path}")
        console_debug(f"ROCm Profiler SDK Tool path: {sdk_tool_path}")

        self.compute_script_dir = compute_script_path.parent
        self.sdk_tool_path = sdk_tool_path
        pass

    def get_collector_library_path(self) -> Path | None:
        native_tool_path = self.__find_existing_collector()
        if not native_tool_path:
            native_tool_path = self.__build_collector()
        console_log(f"Using native collector: {native_tool_path}")
        return native_tool_path

    def __find_existing_collector(self) -> Path | None:
        if self.__is_run_from_rocm_installation_folder():
            collector_path = self.__find_installed_collector()
            if not collector_path:
                raise RuntimeError("Failed to find installed native collector")
        else:
            collector_path = self.__find_built_collector()
        return collector_path

    def __is_run_from_rocm_installation_folder(self) -> bool:
        return self.compute_script_dir.parent == self.sdk_tool_path.parents[2]

    def __find_installed_collector(self) -> Path | None:
        rocm_root_path = self.__get_installed_rocm_root_path()
        pattern = f"lib*/rocprofiler-compute/{self.lib_name}"
        console_debug(f"Searching {rocm_root_path} by {pattern} for native collector")
        return self.__find_file_by_glob_pattern(rocm_root_path, pattern)

    def __get_installed_rocm_root_path(self) -> Path:
        native_tool_base_path = (
            self.sdk_tool_path.parents[2]
            if len(self.sdk_tool_path.parents) > 2
            else Path()
        )
        return native_tool_base_path

    def __find_built_collector(self) -> Path | None:
        pattern = self.lib_relative_path
        console_log(
            f"Searching {self.compute_script_dir} by {pattern} for native collector"
        )
        return self.__find_file_by_glob_pattern(self.compute_script_dir, pattern)

    def __find_file_by_glob_pattern(self, base_path: Path, pattern: str) -> Path | None:
        match = next(base_path.glob(pattern), None)
        return Path(match) if match is not None else None

    def __build_collector(self) -> Path | None:
        self._generate_cmake_project(self.compute_script_dir)
        self._build_cmake_project(self.compute_script_dir)
        return self.__find_built_collector()

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
