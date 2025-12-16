# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import tempfile
from typing import Optional
import re

@dataclass
class RocprofsysConfig:
    """Configuration for rocprofiler-systems test execution

    Contains necessary paths to configure tests for build or for install modes.

        Attributes:
        - rocprofsys_root_dir: Path to either the source or install directory
        - rocprofsys_build_dir: Path to either the build or install directory
        - rocprofsys_instrument: Path to rocprof-sys-instrument executable
        - rocprofsys_run: Path to rocprof-sys-run executable
        - rocprofsys_sample: Path to rocprof-sys-sample executable
        - rocprofsys_causal: Path to rocprof-sys-causal executable
        - rocprofsys_avail: Path to rocprof-sys-avail executable
        - rocm_path: Path to ROCm installation directory
        - rocprofsys_lib_dir: Path to rocprofsys library directory
        - rocprofsys_bin_dir: Path to rocprofsys binary directory
        - rocprofsys_examples_bin_dir: Path to rocprofsys examples binary directory
        - rocprofsys_tests_bin_dir: Path to rocprofsys tests binary directory
        - rocpd_validation_rules: Path to rocprofiler-systems rocpd validation rules directory
        - is_installed: Whether this is an installed configuration
    """

    rocprofsys_root_dir: Path
    rocprofsys_build_dir: Path
    rocprofsys_instrument: Path
    rocprofsys_run: Path
    rocprofsys_sample: Path
    rocprofsys_causal: Path
    rocprofsys_avail: Path
    rocm_path: Path
    rocprofsys_lib_dir: Path
    rocprofsys_bin_dir: Path
    rocprofsys_examples_dir: Path
    rocprofsys_tests_dir: Path
    rocpd_validation_rules: Path
    test_output_dir: Path
    mpiexec: Path
    is_installed: bool = False

    def get_llvm_lib_paths(self) -> list[Path]:
        """Get list of found ROCm LLVM lib paths.

        Returns:
            List of existing LLVM lib paths found, empty list if none found.
        """
        found_paths = []
        if self.rocm_path:
            # Match discover_llvm_libdir_for_ompt() logic
            candidates = [
                self.rocm_path / "llvm" / "lib",
                self.rocm_path / "lib" / "llvm" / "lib",
            ]
            for candidate in candidates:
                if candidate.exists():
                    found_paths.append(candidate)
        return found_paths

    def get_library_path(self) -> str:
        """Get LD_LIBRARY_PATH including rocprofiler-systems libraries."""
        paths = [str(self.rocprofsys_lib_dir)]

        # Add ROCm LLVM lib if available
        for llvm_path in self.get_llvm_lib_paths():
            paths.append(str(llvm_path))

        # Append existing LD_LIBRARY_PATH
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        if existing:
            paths.append(existing)

        return ":".join(paths)

    def get_target_executable(self, name: str) -> Path:
        """Get path to a test target executable.

        When is_installed is True, searches in the following order:
        1. rocprofsys_root_dir/name (build directory layout)
        2. rocprofsys_examples_dir/name/name (build directory layout)
        3. PATH lookup via shutil.which

        When is_installed is False, searches in the following order:
        1. rocprofsys_examples_dir/name
        2. rocprofsys_bin_dir/name
        3. PATH lookup via shutil.which

        Args:
            name: Name of the target executable

        Returns:
            Path to the executable

        Raises:
            FileNotFoundError: If the executable is not found
        """

        if self.is_installed:
            # examples directory layout
            exe = self.rocprofsys_examples_dir / name
            if exe.exists() and exe.is_file():
                return exe

            # binary directory
            exe = self.rocprofsys_bin_dir / name
            if exe.exists() and exe.is_file():
                return exe

            # PATH lookup via shutil.which
            exe = shutil.which(name)
            if exe:
                return Path(exe)

            raise FileNotFoundError(
                f"Target executable '{name}' not found. Searched in:\n"
                f"  - {self.rocprofsys_examples_dir}/{name}\n"
                f"  - {self.rocprofsys_bin_dir}/{name}\n"
                f"  - PATH"
            )

        else:
            # Build directory mode
            exe = self.rocprofsys_examples_dir / name
            if exe.exists() and exe.is_file():
                return exe

            exe = self.rocprofsys_examples_dir / "examples" / name / name
            if exe.exists() and exe.is_file():
                return exe

            # binary directory
            exe = self.rocprofsys_bin_dir / name
            if exe.exists() and exe.is_file():
                return exe

            # PATH lookup via shutil.which
            exe = shutil.which(name)
            if exe:
                return Path(exe)

            raise FileNotFoundError(
                f"Target executable '{name}' not found. Searched in:\n"
                f"  - {self.rocprofsys_examples_dir}/{name}\n"
                f"  - {self.rocprofsys_examples_dir}/examples/{name}/{name}\n"
                f"  - {self.rocprofsys_bin_dir}/{name}\n"
                f"  - PATH"
            )

    def get_base_environment(self) -> dict[str, str]:
        """Get base environment variables for test execution."""
        return {
            "ROCPROFSYS_CI": "ON",
            "ROCPROFSYS_CONFIG_FILE": "",
            "ROCPROFSYS_TRACE": "ON",
            "ROCPROFSYS_PROFILE": "ON",
            "ROCPROFSYS_USE_SAMPLING": "ON",
            "ROCPROFSYS_USE_PROCESS_SAMPLING": "ON",
            "ROCPROFSYS_TIME_OUTPUT": "OFF",
            "ROCPROFSYS_FILE_OUTPUT": "ON",
            "ROCPROFSYS_USE_PID": "OFF",
            "ROCPROFSYS_VERBOSE": "1",
            "ROCPROFSYS_SAMPLING_FREQ": "300",
            "ROCPROFSYS_SAMPLING_DELAY": "0.05",
            "OMP_PROC_BIND": "spread",
            "OMP_PLACES": "threads",
            "OMP_NUM_THREADS": "2",
            "LD_LIBRARY_PATH": self.get_library_path(),
        }

    def get_base_binary_environment(self) -> dict[str, str]:
        """Get base environment variables for rocprof-sys binary test execution."""
        return {
            "ROCPROFSYS_TRACE": "ON",
            "ROCPROFSYS_PROFILE": "ON",
            "ROCPROFSYS_USE_SAMPLING": "ON",
            "ROCPROFSYS_TIME_OUTPUT": "OFF",
            "LD_LIBRARY_PATH": self.get_library_path(),
            "ROCPROFSYS_CI": "ON",
            "ROCPROFSYS_CI_TIMEOUT": "300",
            "ROCPROFSYS_CONFIG_FILE": "",
        }

def _find_rocm_path() -> Optional[Path]:
    """Find ROCm installation path."""
    for candidate in [
        os.environ.get("ROCM_PATH"),
        "/opt/rocm",
        "/usr/local/rocm",
    ]:
        if candidate and Path(candidate).exists():
            return Path(candidate)
    return None


def _get_rocm_version() -> Optional[tuple[int, int, int]]:
    """Get the installed ROCm version as a tuple (major, minor, patch).

    Returns:
        Tuple of (major, minor, patch) or None if ROCm not found or version undetectable.
    """
    rocm_path = _find_rocm_path()
    if not rocm_path:
        return None

    # Check .info/version file (standard location)
    version_file = rocm_path / ".info" / "version"
    if not version_file.exists():
        # Try alternative location
        version_file = rocm_path / "share" / "rocm" / "version"

    if version_file.exists():
        try:
            version_str = version_file.read_text().strip()
            # Parse version like "6.2.0" or "6.2.0-12345"
            match = re.match(r"(\d+)\.(\d+)\.(\d+)", version_str)
            if match:
                return (int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except (OSError, ValueError):
            pass

    return None


def _check_rocm_version(min_version: str) -> bool:
    """Check if installed ROCm version meets minimum requirement.

    Args:
        min_version: Minimum version string like "7.0" or "6.2.1"

    Returns:
        True if ROCm version >= min_version, False otherwise.
    """
    current = _get_rocm_version()
    if current is None:
        return False

    # Parse min_version
    parts = min_version.split(".")
    min_tuple = tuple(int(p) for p in parts)
    # Pad with zeros if needed (e.g., "7.0" -> (7, 0, 0))
    while len(min_tuple) < 3:
        min_tuple = min_tuple + (0,)

    return current >= min_tuple

def _find_mpiexec() -> Optional[Path]:
    """Find MPI laucnher executable."""
    for candidate in ["mpiexec", "mpirun"]:
        path = shutil.which(candidate)
        if path:
            return Path(path)
    return None

def _find_executable(name: str, search_paths: list[Path]) -> Optional[Path]:
    """Find an executable in search paths or via PATH."""
    for search_dir in search_paths:
        exe = search_dir / name
        if exe.exists() and exe.is_file():
            return exe

    # Fallback to PATH
    path_exe = shutil.which(name)
    if path_exe:
        return Path(path_exe)

    return None


def discover_install_config(
    install_dir: Optional[Path] = None,
) -> RocprofsysConfig:
    """Discover rocprofiler-systems installation configuration.

    Creates configuration for testing against installed binaries.

    Args:
        install_dir: Installation prefix (e.g., /opt/rocm or /usr/local)

    Returns:
        RocprofsysConfig configured for installed binaries

    Raises:
        FileNotFoundError: If installation cannot be found
    """

    if install_dir is None:
        env_install = os.environ.get("ROCPROFSYS_INSTALL_DIR")
        if env_install:
            install_dir = Path(env_install)
        else:
            for candidate in [
                _find_rocm_path(),
                Path("/usr/local"),
                Path("/usr"),
                Path("/opt/rocprofiler-systems"), # Standard install location from README.md
            ]:
                if (candidate
                    and (candidate / "share" / "rocprofiler-systems" / "tests").is_dir()
                    and (candidate / "share" / "rocprofiler-systems" / "examples").is_dir()):
                    install_dir = candidate
                    break

    if install_dir is None:
        raise FileNotFoundError(
            "Could not find a suitable rocprofiler-systems installation. Set ROCPROFSYS_INSTALL_DIR "
            "environment variable."
            "A suitable installation is one that has the following directory: share/rocprofiler-systems/examples "
            "and share/rocprofiler-systems/tests"
        )

    install_dir = install_dir.resolve()

    # Determine directory layout
    bin_dir = install_dir / "bin"
    lib_dir = install_dir / "lib"

    # For lib64 systems
    if not lib_dir.exists() and (install_dir / "lib64").exists():
        lib_dir = install_dir / "lib64"

    examples_dir = install_dir / "share" / "rocprofiler-systems" / "examples"
    tests_dir = install_dir / "share" / "rocprofiler-systems" / "tests"
    rocpd_validation_rules = tests_dir / "rocpd-validation-rules"

    # Create a temporary directory for test outputs if needed
    output_dir = Path(tempfile.gettempdir()) / "rocprof-sys-pytest-output"
    output_dir.mkdir(parents=True, exist_ok=True)

    rocm_path = _find_rocm_path()
    mpiexec = _find_mpiexec()
    search_paths = [bin_dir]

    rocprof_instrument = _find_executable("rocprof-sys-instrument", search_paths)
    rocprof_sample = _find_executable("rocprof-sys-sample", search_paths)
    rocprof_run = _find_executable("rocprof-sys-run", search_paths)
    rocprof_causal = _find_executable("rocprof-sys-causal", search_paths)
    rocprof_avail = _find_executable("rocprof-sys-avail", search_paths)

    # If any of the executables are not found, raise an error
    required_executables = {
        "rocprof-sys-instrument": rocprof_instrument,
        "rocprof-sys-sample": rocprof_sample,
        "rocprof-sys-run": rocprof_run,
        "rocprof-sys-causal": rocprof_causal,
        "rocprof-sys-avail": rocprof_avail,
    }

    missing = [name for name, path in required_executables.items() if path is None]
    if missing:
        raise FileNotFoundError(
            f"Required executables not found: {', '.join(missing)}. "
            f"Searched in: {search_paths}"
        )

    return RocprofsysConfig(
        rocprofsys_root_dir=install_dir,
        rocprofsys_build_dir=install_dir,
        rocprofsys_instrument=rocprof_instrument,
        rocprofsys_run=rocprof_run,
        rocprofsys_sample=rocprof_sample,
        rocprofsys_causal=rocprof_causal,
        rocprofsys_avail=rocprof_avail,
        rocm_path=rocm_path,
        rocprofsys_lib_dir=lib_dir,
        rocprofsys_bin_dir=bin_dir,
        rocprofsys_examples_dir=examples_dir,
        rocprofsys_tests_dir=tests_dir,
        rocpd_validation_rules=rocpd_validation_rules,
        test_output_dir=output_dir,
        mpiexec=mpiexec,
        is_installed=True,
    )

def discover_build_config(
    build_dir: Optional[Path] = None,
    source_dir: Optional[Path] = None,
) -> RocprofsysConfig:
    """Discover rocprofiler-systems build configuration.

    Attempts to find the build directory and source directory automatically
    if not provided, checking common locations and environment variables.

    If no build directory is found but an installation is available,
    falls back to discover_install_config().

    Args:
        build_dir: Explicit build directory path
        source_dir: Explicit source directory path

    Returns:
        RocprofsysConfig with discovered paths

    Raises:
        FileNotFoundError: If neither build directory nor installation found
    """

    # Explicit install directory check
    if os.environ.get("ROCPROFSYS_INSTALL_DIR"):
        return discover_install_config()

    if build_dir is None:
        env_build = os.environ.get("ROCPROFSYS_BUILD_DIR")
        if env_build:
            build_dir = Path(env_build)
        else:
            test_dir = Path(__file__).parent.parent.parent.parent
            for candidate in [
                test_dir / "rocprof-sys-build",
                test_dir / "build" / "debug",
                test_dir / "build" / "release",
                test_dir / "build",
                Path.cwd() / "rocprof-sys-build",
                Path.cwd() / "build" / "debug",
                Path.cwd() / "build" / "release",
                Path.cwd() / "build",
            ]:
                if candidate.exists() and (candidate / "bin").exists():
                    build_dir = candidate
                    break

    if build_dir is None or not build_dir.exists():
        raise FileNotFoundError(
            "Could not find build directory or installation. Set one of:\n"
            "  - ROCPROFSYS_BUILD_DIR: Path to build directory\n"
            "  - ROCPROFSYS_INSTALL_DIR: Path to installation prefix"
        )

    if source_dir is None:
        env_source = os.environ.get("ROCPROFSYS_SOURCE_DIR")
        if env_source:
            source_dir = Path(env_source)
        else:
            cmake_cache = build_dir / "CMakeCache.txt"
            if cmake_cache.exists():
                content = cmake_cache.read_text()
                match = re.search(
                    r"CMAKE_HOME_DIRECTORY:INTERNAL=(.+)", content
                )
                if match:
                    source_dir = Path(match.group(1))

            if source_dir is None:
                # Walk up from build_dir
                source_dir = build_dir
                # If their source dir is higher, build_dir should be specified
                for _ in range(2):
                    parent = source_dir.parent
                    if parent == source_dir:
                        break
                    if source_dir.name in ("build", "debug", "release", "rocprof-sys-build"):
                        source_dir = parent
                    else:
                        break

            # Validate that we found a valid source directory
            if not (source_dir / "CMakeLists.txt").exists():
                raise FileNotFoundError(
                    f"Could not find source directory. Detected '{source_dir}' but it does not "
                    f"contain CMakeLists.txt. Set ROCPROFSYS_SOURCE_DIR environment variable."
                )

    source_dir = source_dir.resolve()

    rocm_path = _find_rocm_path()
    mpiexec = _find_mpiexec()

    bin_dir = build_dir / "bin"
    lib_dir = build_dir / "lib"

    search_paths = [bin_dir]

    rocprof_instrument = _find_executable("rocprof-sys-instrument", search_paths)
    rocprof_sample = _find_executable("rocprof-sys-sample", search_paths)
    rocprof_run = _find_executable("rocprof-sys-run", search_paths)
    rocprof_causal = _find_executable("rocprof-sys-causal", search_paths)
    rocprof_avail = _find_executable("rocprof-sys-avail", search_paths)

    # If any of the executables are not found, raise an error
    required_executables = {
        "rocprof-sys-instrument": rocprof_instrument,
        "rocprof-sys-sample": rocprof_sample,
        "rocprof-sys-run": rocprof_run,
        "rocprof-sys-causal": rocprof_causal,
        "rocprof-sys-avail": rocprof_avail,
    }

    missing = [name for name, path in required_executables.items() if path is None]
    if missing:
        raise FileNotFoundError(
            f"Required executables not found: {', '.join(missing)}. "
            f"Searched in: {search_paths}"
        )

    return RocprofsysConfig(
        rocprofsys_root_dir=source_dir,
        rocprofsys_build_dir=build_dir,
        rocprofsys_instrument=rocprof_instrument,
        rocprofsys_run=rocprof_run,
        rocprofsys_sample=rocprof_sample,
        rocprofsys_causal=rocprof_causal,
        rocprofsys_avail=rocprof_avail,
        rocm_path=rocm_path,
        rocprofsys_lib_dir=lib_dir,
        rocprofsys_bin_dir=bin_dir,
        rocprofsys_examples_dir=build_dir, # Example binaries are in root of build directory
        rocprofsys_tests_dir=source_dir / "tests",
        rocpd_validation_rules=source_dir / "tests" / "rocpd-validation-rules",
        test_output_dir=build_dir / "rocprof-sys-pytest-output",
        mpiexec=mpiexec,
        is_installed=False,
    )
