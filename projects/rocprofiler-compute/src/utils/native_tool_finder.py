# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT
import shlex
from pathlib import Path

from utils.logger import console_debug, console_log
from utils.utils_common import capture_subprocess_output


class NativeToolFinder:
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

    def __init__(self, root_path: Path) -> None:
        console_debug("Searching for native collector.")
        console_debug(f"ROCm Compute root directory: {root_path}")

        self.root_path = root_path
        pass

    def get_collector_library_path(self) -> Path:
        collector_path = self.__find_installed_collector()
        if not collector_path:
            collector_path = self.__build_collector()
        if not collector_path:
            raise RuntimeError("Failed to find or build collector")
        console_log(f"Using native collector: {collector_path}")
        return collector_path

    def __find_installed_collector(self) -> Path | None:
        rocm_root_path = self.__get_installed_rocm_root_path()
        # lib* glob pattern is used to handle CMAKE_INSTALL_LIBDIR variations
        pattern = f"lib*/rocprofiler-compute/{self.lib_name}"
        console_debug(f"Searching {rocm_root_path} by {pattern} for native collector")
        return self.__find_file_by_glob_pattern(rocm_root_path, pattern)

    def __get_installed_rocm_root_path(self) -> Path:
        native_tool_base_path = (
            self.root_path.parents[1] if len(self.root_path.parents) > 1 else Path()
        )
        return native_tool_base_path

    def __find_file_by_glob_pattern(self, base_path: Path, pattern: str) -> Path | None:
        match = next(base_path.glob(pattern), None)
        return Path(match) if match is not None else None

    def __build_collector(self) -> Path | None:
        self._generate_cmake(self.root_path)
        self._build_cmake(self.root_path)
        return self.__find_built_collector()

    def __find_built_collector(self) -> Path | None:
        pattern = self.lib_relative_path
        console_log(f"Searching {self.root_path} by {pattern} for native collector")
        return self.__find_file_by_glob_pattern(self.root_path, pattern)

    def _generate_cmake(self, src_path: Path) -> None:
        generate_command = (
            "cmake "
            f"-S {src_path}/{self.sources_dir_name} "
            f"-B {src_path}/{self.sources_dir_name}/{self.sources_build_subdir_name}"
        )
        console_log(f"Generating native tool project using command: {generate_command}")
        self.__execute_command(generate_command)

    def _build_cmake(self, src_path: Path) -> None:
        build_command = (
            "cmake --build "
            f"{src_path}/{self.sources_dir_name}/{self.sources_build_subdir_name} "
            "--parallel"
        )
        console_log(f"Building native tool using command: {build_command}")
        self.__execute_command(build_command)

    def __execute_command(self, command: str) -> None:
        # Output is logged when enable_logging=False is not provided
        success, _ = capture_subprocess_output(shlex.split(command))
        if not success:
            msg = f"Failed to execute command: {command}"
            raise RuntimeError(msg)
