# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT
import os
import shlex
from pathlib import Path
from typing import List, Optional, Tuple

from utils.logger import console_debug, console_log
from utils.utils_common import capture_subprocess_output

# Lines of cmake output kept when a command fails.
_ERROR_OUTPUT_LINES = 20

_INSTALLED_LIB_ARTIFACT_GLOB = "lib*/rocprofiler-compute/{artifact_name}"


def find_installed_artifacts(root_path: Path, artifact_name: str) -> List[Path]:
    """Return ``lib*/rocprofiler-compute/`` files matching ``artifact_name``."""
    search_root = root_path.parents[1] if len(root_path.parents) > 1 else Path()
    pattern = _INSTALLED_LIB_ARTIFACT_GLOB.format(artifact_name=artifact_name)
    console_debug(f"Searching {search_root} for {pattern}")
    return sorted(path for path in search_root.glob(pattern) if path.is_file())


class NativeToolFinder:
    """Locate an artifact of the native tool project, building it from the source
    tree when no installed copy is present.

    Set ``reuse_built_artifact`` only for artifacts whose name encodes the
    sources they were built from; a fixed-name artifact would be reused stale.
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
        build_target: Optional[str] = None,
        build_path: Optional[Path] = None,
        configure_options: Tuple[str, ...] = (),
        cmake_executable: str = "cmake",
        search_installed: bool = True,
        reuse_built_artifact: bool = False,
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
        self.reuse_built_artifact = reuse_built_artifact

    def get_artifact_path(self) -> Path:
        artifact_path = (
            self._find_installed_artifact() if self.search_installed else None
        )
        if not artifact_path and self.reuse_built_artifact:
            artifact_path = self._find_built_artifact()
        if not artifact_path:
            artifact_path = self._build_artifact()
        if not artifact_path:
            raise RuntimeError(f"Failed to find or build {self.artifact_name}")
        console_log(f"Using {self.artifact_name}: {artifact_path}")
        return artifact_path

    def _find_installed_artifact(self) -> Optional[Path]:
        matches = find_installed_artifacts(self.root_path, self.artifact_name)
        return matches[0] if matches else None

    def _build_artifact(self) -> Optional[Path]:
        self._generate_cmake()
        self._build_cmake()
        return self._find_built_artifact()

    def _find_built_artifact(self) -> Optional[Path]:
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
        self._execute_command(command)

    def _build_cmake(self) -> None:
        command = [self.cmake_executable, "--build", str(self.build_path), "--parallel"]
        if self.build_target is not None:
            command += ["--target", self.build_target]
        console_log(f"Building native tool using command: {shlex.join(command)}")
        self._execute_command(command)

    def _execute_command(self, command: List[str]) -> None:
        # Output is logged when enable_logging=False is not provided
        success, output = capture_subprocess_output(command)
        if success:
            return
        # The tail carries the cmake diagnostic the loader classifies.
        tail = "\n".join(output.strip().splitlines()[-_ERROR_OUTPUT_LINES:])
        message = f"Failed to execute command: {shlex.join(command)}"
        raise RuntimeError(f"{message}\n{tail}" if tail else message)


def _source_library_output_dirs(root_path: Path) -> List[Path]:
    directories: List[Path] = []
    cmake_binary_dir = os.environ.get("CMAKE_BINARY_DIR")
    if cmake_binary_dir:
        directories.append(
            Path(cmake_binary_dir) / NativeToolFinder.sources_bin_subdir_name
        )
    directories.append(
        root_path.parent / "build" / NativeToolFinder.sources_bin_subdir_name
    )
    directories.append(
        root_path
        / NativeToolFinder.sources_dir_name
        / NativeToolFinder.sources_build_subdir_name
        / NativeToolFinder.sources_bin_subdir_name
    )
    return directories


def find_prebuilt_artifacts(root_path: Path, artifact_name: str) -> List[Path]:
    """Return matching artifacts from installed and source-tree locations."""
    artifacts: List[Path] = list(find_installed_artifacts(root_path, artifact_name))
    for directory in _source_library_output_dirs(root_path):
        if not directory.is_dir():
            continue
        artifacts.extend(
            sorted(path for path in directory.glob(artifact_name) if path.is_file())
        )
    unique_artifacts: List[Path] = []
    seen_paths = set()
    for path in artifacts:
        resolved = path.resolve()
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        unique_artifacts.append(path)
    return unique_artifacts
