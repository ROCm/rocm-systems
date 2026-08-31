# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT
import shlex
from pathlib import Path
from typing import List, Optional

from utils.logger import console_debug, console_log
from utils.utils_common import capture_subprocess_output

_INSTALLED_LIB_ARTIFACT_GLOB = "lib*/rocprofiler-compute/{artifact_name}"


def find_installed_artifacts(root_path: Path, artifact_name: str) -> List[Path]:
    """Return ``lib*/rocprofiler-compute/`` files matching ``artifact_name``."""
    search_root = root_path.parents[1] if len(root_path.parents) > 1 else Path()
    pattern = _INSTALLED_LIB_ARTIFACT_GLOB.format(artifact_name=artifact_name)
    console_debug(f"Searching {search_root} for {pattern}")
    return sorted(path for path in search_root.glob(pattern) if path.is_file())


def find_prebuilt_artifacts(root_path: Path, artifact_name: str) -> List[Path]:
    """Return matching artifacts from installed and source-tree locations."""
    source_dirs = [
        root_path.parent / "build" / NativeToolFinder.sources_bin_subdir_name,
        root_path
        / NativeToolFinder.sources_dir_name
        / NativeToolFinder.sources_build_subdir_name
        / NativeToolFinder.sources_bin_subdir_name,
    ]
    artifacts: List[Path] = list(find_installed_artifacts(root_path, artifact_name))
    for directory in source_dirs:
        artifacts.extend(
            sorted(path for path in directory.glob(artifact_name) if path.is_file())
        )
    return list(dict.fromkeys(path.resolve() for path in artifacts))


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

    def __init__(self, root_path: Path) -> None:
        console_debug(f"Searching for {self.lib_name}.")
        console_debug(f"ROCm Compute root directory: {root_path}")

        self.root_path = root_path
        self.sources_path = root_path / self.sources_dir_name
        self.build_path = self.sources_path / self.sources_build_subdir_name

    def get_artifact_path(self) -> Path:
        artifact_path = self._find_installed_artifact()
        if not artifact_path:
            artifact_path = self._build_artifact()
        if not artifact_path:
            raise RuntimeError(f"Failed to find or build {self.lib_name}")
        console_log(f"Using {self.lib_name}: {artifact_path}")
        return artifact_path

    def _find_installed_artifact(self) -> Optional[Path]:
        matches = find_installed_artifacts(self.root_path, self.lib_name)
        return matches[0] if matches else None

    def _build_artifact(self) -> Optional[Path]:
        self._generate_cmake()
        self._build_cmake()
        return self._find_built_artifact()

    def _find_built_artifact(self) -> Optional[Path]:
        artifact_path = self.build_path / self.sources_bin_subdir_name / self.lib_name
        console_debug(f"Built artifact expected at {artifact_path}")
        return artifact_path if artifact_path.is_file() else None

    def _generate_cmake(self) -> None:
        command = [
            "cmake",
            "-S",
            str(self.sources_path),
            "-B",
            str(self.build_path),
        ]
        console_log(
            f"Generating native tool project using command: {shlex.join(command)}"
        )
        self._execute_command(command)

    def _build_cmake(self) -> None:
        command = ["cmake", "--build", str(self.build_path), "--parallel"]
        console_log(f"Building native tool using command: {shlex.join(command)}")
        self._execute_command(command)

    def _execute_command(self, command: List[str]) -> None:
        # Output is logged when enable_logging=False is not provided
        success, _ = capture_subprocess_output(command)
        if not success:
            raise RuntimeError(f"Failed to execute command: {shlex.join(command)}")
