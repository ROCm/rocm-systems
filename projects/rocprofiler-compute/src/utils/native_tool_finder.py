# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT
import shlex
from pathlib import Path

from utils.logger import console_debug, console_log
from utils.utils_common import capture_subprocess_output


class NativeToolFinder:
    """Locate an artifact of the native tool project, building it from the source
    tree when no installed copy is present.
    """

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

    def __init__(
        self,
        root_path: Path,
        *,
        artifact_name: str = lib_name,
        build_target: str | None = None,
        build_path: Path | None = None,
        configure_options: tuple[str, ...] = (),
        cmake_executable: str = "cmake",
        search_installed: bool = True,
    ) -> None:
        console_debug(f"Searching for {artifact_name}.")
        console_debug(f"ROCm Compute root directory: {root_path}")

        self.root_path = root_path
        self.artifact_name = artifact_name
        self.build_target = build_target
        self.sources_path = root_path / self.sources_dir_name
        self.build_path = (
            self.sources_path / self.sources_build_subdir_name
            if build_path is None
            else build_path
        )
        self.configure_options = tuple(configure_options)
        self.cmake_executable = cmake_executable
        self.search_installed = search_installed

    def get_artifact_path(self) -> Path:
        artifact_path = (
            self.__find_installed_artifact() if self.search_installed else None
        )
        if not artifact_path:
            artifact_path = self.__build_artifact()
        if not artifact_path:
            raise RuntimeError(f"Failed to find or build {self.artifact_name}")
        console_log(f"Using {self.artifact_name}: {artifact_path}")
        return artifact_path

    def __find_installed_artifact(self) -> Path | None:
        rocm_root_path = self.__get_installed_rocm_root_path()
        # lib* glob pattern is used to handle CMAKE_INSTALL_LIBDIR variations
        pattern = f"lib*/rocprofiler-compute/{self.artifact_name}"
        console_debug(f"Searching {rocm_root_path} for {pattern}")
        return self.__find_file_by_glob_pattern(rocm_root_path, pattern)

    def __get_installed_rocm_root_path(self) -> Path:
        native_tool_base_path = (
            self.root_path.parents[1] if len(self.root_path.parents) > 1 else Path()
        )
        return native_tool_base_path

    def __find_file_by_glob_pattern(self, base_path: Path, pattern: str) -> Path | None:
        match = next(base_path.glob(pattern), None)
        return Path(match) if match is not None else None

    def __build_artifact(self) -> Path | None:
        self._generate_cmake()
        self._build_cmake()
        return self.__find_built_artifact()

    def __find_built_artifact(self) -> Path | None:
        artifact_path = (
            self.build_path / self.sources_bin_subdir_name / self.artifact_name
        )
        console_debug(f"Built artifact expected at {artifact_path}")
        return artifact_path if artifact_path.is_file() else None

    def _generate_cmake(self) -> None:
        command = [
            self.cmake_executable,
            "-S",
            str(self.sources_path),
            "-B",
            str(self.build_path),
            *self.configure_options,
        ]
        console_log(
            f"Generating native tool project using command: {shlex.join(command)}"
        )
        self.__execute_command(command)

    def _build_cmake(self) -> None:
        command = [self.cmake_executable, "--build", str(self.build_path), "--parallel"]
        if self.build_target is not None:
            command += ["--target", self.build_target]
        console_log(f"Building native tool using command: {shlex.join(command)}")
        self.__execute_command(command)

    def __execute_command(self, command: list[str]) -> None:
        # Output is logged when enable_logging=False is not provided
        success, output = capture_subprocess_output(command)
        if not success:
            msg = f"Failed to execute command: {shlex.join(command)}"
            tail = self.__output_tail(output)
            raise RuntimeError(f"{msg}\n{tail}" if tail else msg)

    @staticmethod
    def __output_tail(output: str, max_lines: int = 20) -> str:
        lines = output.strip().splitlines()
        return "\n".join(lines[-max_lines:])
