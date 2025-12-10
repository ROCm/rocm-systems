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


@dataclass
class RocprofsysConfig:
    """Configuration for rocprofiler-systems test execution

    Contains necessary paths to configure tests for build or for install modes.

    Attributes:
        build_dir: Path to the build or install directory (used for output)
        rocprofsys_instrument: Path to rocprof-sys-instrument executable
        rocprofsys_run: Path to rocprof-sys-run executable
        rocprofsys_sample: Path to rocprof-sys-sample executable
        rocprofsys_causal: Path to rocprof-sys-causal executable
        rocprofsys_avail: Path to rocprof-sys-avail executable
        lib_dir: Path to library directory
        bin_dir: Path to binary directory
        is_installed: Whether this is an installed configuration
    """

    build_dir: Path
    rocprofsys_instrument: Path
    rocprofsys_run: Path
    rocprofsys_sample: Path
    rocprofsys_causal: Path
    rocprofsys_avail: Path
    rocpd_validation_rules: Path
    rocm_path: Path
    lib_dir: Path
    bin_dir: Path
    installed_examples_dir: Path
    installed_tests_dir: Path
    is_installed: bool = True

    def get_library_path(self) -> str:
        """Get LD_LIBRARY_PATH including rocprofiler-systems libraries."""
        paths = [str(self.lib_dir)]

        # Add ROCm LLVM lib if available
        if self.rocm_path:
            llvm_lib = self.rocm_path / "lib" / "llvm" / "lib"
            if llvm_lib.exists():
                paths.append(str(llvm_lib))

        # Append existing LD_LIBRARY_PATH
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        if existing:
            paths.append(existing)

        return ":".join(paths)

    def get_target_executable(self, name: str) -> Path:
        """Get path to a test target executable.

        Args:
            name: Name of the target executable

        Returns:
            Path to the executable

        Raises:
            FileNotFoundError: If the executable is not found
        """

        # Installed executable
        if self.installed_examples_dir:
            exe = self.installed_examples_dir / name
            if exe.exists() and exe.is_file():
                return exe

        raise FileNotFoundError(
            f"Target executable '{name}' not found. Searched in:\n"
            f"  - {self.installed_examples_dir}/{name}"
        )

    def get_base_environment(self) -> dict[str, str]:
        """Get base environment variables for test execution."""
        return {
            "ROCPROFSYS_CI": "ON",
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
            # Try common installation locations
            for candidate in [
                _find_rocm_path(),
                Path("/opt/rocprofiler-systems"), # TODO: Reminder here
                Path("/usr/local"),
                Path("/usr"),
            ]:
                if candidate and (candidate / "share" / "rocprofiler-systems" / "examples").is_dir():
                    install_dir = candidate
                    break

    if install_dir is None:
        raise FileNotFoundError(
            "Could not find a suitable rocprofiler-systems installation. Set ROCPROFSYS_INSTALL_DIR "
            "environment variable."
            "A suitable installation is one that has the following directory: share/rocprofiler-systems/examples"
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

    # Executable time :D
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
        build_dir=output_dir,
        rocprofsys_instrument=rocprof_instrument,
        rocprofsys_run=rocprof_run,
        rocprofsys_sample=rocprof_sample,
        rocprofsys_causal=rocprof_causal,
        rocprofsys_avail=rocprof_avail,
        rocpd_validation_rules=rocpd_validation_rules,
        rocm_path=rocm_path,
        lib_dir=lib_dir,
        bin_dir=bin_dir,
        is_installed=True,
        installed_examples_dir=examples_dir,
        installed_tests_dir=tests_dir,
    )
