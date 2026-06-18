# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations
from dataclasses import dataclass, field
import getpass
import os
from pathlib import Path
import shutil
import tempfile
from typing import Optional
import re
import subprocess

from .capabilities import SystemCapabilities


@dataclass
class RocprofsysConfig:
    """Configuration for rocprofiler-systems test execution

    Contains necessary paths to configure tests for build or for install modes.

        Attributes:
        - rocprofsys_build_dir: Path to either the build or install directory
        - rocprofsys_instrument: Path to rocprof-sys-instrument executable
        - rocprofsys_run: Path to rocprof-sys-run executable
        - rocprofsys_sample: Path to rocprof-sys-sample executable
        - rocprofsys_causal: Path to rocprof-sys-causal executable
        - rocprofsys_avail: Path to rocprof-sys-avail executable
        - rocm_path: Path to ROCm installation directory
        - rocprofsys_lib_dir: Path to rocprofsys library directory
        - rocprofsys_bin_dir: Path to rocprofsys binary directory
        - rocprofsys_examples_dir: Path
            In build mode, this is the root of the build directory (most examples lie here).
            In install mode, this is the examples/ directory.
        - rocprofsys_tests_dir: Path to rocprofsys tests directory
        - test_output_dir: Path to test output directory
        - rocpd_validation_rules: Path to rocprofiler-systems rocpd validation rules directory
        - rocm_version: Tuple of (major, minor, patch) of the installed ROCm version
        - is_installed: Whether this is an installed configuration
        - rocprofsys_python: Path to rocprof-sys-python executable
        - rocprofsys_site_packages: Path to rocprofsys site-packages directory
        - _python_versions_hint: List of python versions available (hint for pytest_generate_tests)
        - _python_root_dirs_hint: List of python root directories available (hint for pytest_generate_tests)
        - Capabilities: SystemCapabilities instance
    """

    rocprofsys_build_dir: Path
    rocprofsys_instrument: Path
    rocprofsys_run: Path
    rocprofsys_sample: Path
    rocprofsys_causal: Path
    rocprofsys_avail: Path
    rocm_path: Optional[Path]
    rocprofsys_lib_dir: Path
    rocprofsys_bin_dir: Path
    rocprofsys_examples_dir: Path
    rocprofsys_tests_dir: Path
    rocpd_validation_rules: Path
    test_output_dir: Path
    rocprofsys_python: Optional[Path] = None
    rocprofsys_site_packages: Optional[Path] = None
    is_installed: bool = False
    rocm_version: Optional[tuple[int, int, int]] = None
    _python_versions_hint: Optional[list[str]] = field(default=None, repr=False)
    _python_root_dirs_hint: Optional[list[Path]] = field(default=None, repr=False)
    _capabilities: Optional[SystemCapabilities] = field(
        default=None, init=False, repr=False
    )

    @property
    def capabilities(self) -> SystemCapabilities:
        """Lazy-initialized system capabilities"""
        if self._capabilities is None:
            self._capabilities = SystemCapabilities.from_config(self)
        return self._capabilities

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
        """Get LD_LIBRARY_PATH including rocprofiler-systems libraries.

        Returns:
            LD_LIBRARY_PATH string with rocprofiler-systems libraries
        """
        paths = [str(self.rocprofsys_lib_dir.resolve())]

        # Where libraries for the examples live
        if self.is_installed:
            examples_lib = self.rocprofsys_examples_dir / "lib"
            if examples_lib.is_dir():
                paths.append(str(examples_lib.resolve()))

        existing = os.environ.get("LD_LIBRARY_PATH", "")
        if existing:
            paths.append(existing)

        # Add ROCm LLVM lib as fallback
        for llvm_path in self.get_llvm_lib_paths():
            paths.append(str(llvm_path))

        return ":".join(paths)

    def get_target_executable(
        self, name: str, python_version: Optional[str] = None
    ) -> Path:
        """Get path to a test target executable.

        When is_installed is True, searches in the following order:
        1. rocprofsys_build_dir/name (build directory layout)
        2. rocprofsys_examples_dir/name/name (build directory layout)
        3. PATH lookup

        When is_installed is False, searches in the following order:
        1. rocprofsys_examples_dir/name
        2. rocprofsys_bin_dir/name
        3. PATH lookup

        Args:
            name: Name of the target executable
            python_version: Optional Python version string

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
            # Python check
            exe = self.rocprofsys_examples_dir / "examples" / "python" / name
            if exe.exists() and exe.is_file():
                return exe

            # examples directory layout
            exe = self.rocprofsys_examples_dir / name
            if exe.exists() and exe.is_file():
                return exe

            # code-coverage.py lies in the code-coverage directory
            if name == "code-coverage.py":
                exe = self.rocprofsys_examples_dir / "examples" / "code-coverage" / name
                if exe.exists() and exe.is_file():
                    return exe

            exe = self.rocprofsys_examples_dir / "examples" / name / name
            if exe.exists() and exe.is_file():
                return exe

            # rccl tests lie in their own directory
            exe = self.rocprofsys_examples_dir / "examples" / "rccl" / name
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
                f"  - {self.rocprofsys_examples_dir}/examples/python/{name}\n"
                f"  - {self.rocprofsys_examples_dir}/{name}\n"
                f"  - {self.rocprofsys_examples_dir}/examples/code-coverage/{name}\n"
                f"  - {self.rocprofsys_examples_dir}/examples/rccl/{name}\n"
                f"  - {self.rocprofsys_examples_dir}/examples/{name}/{name}\n"
                f"  - {self.rocprofsys_bin_dir}/{name}\n"
                f"  - PATH"
            )

    def get_fundamental_environment(self) -> dict[str, str]:
        """Get fundamental environment variables inherited from parent process."""
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", ""),
            "USER": os.environ.get("USER", ""),
            "SHELL": os.environ.get("SHELL", ""),
            "TERM": os.environ.get("TERM", ""),
            "LANG": os.environ.get("LANG", ""),
        }
        # To maintain a stable environment, only inherit OMPI_ and ROCPROFSYS_ env vars
        for key, value in os.environ.items():
            if key.startswith(("OMPI_", "ROCPROFSYS_")):
                env[key] = value

        # Forward sanitizer runtime options so pytest-launched binaries honor
        # the suppression files / exitcode set by the CI workflow.
        for key in (
            "ASAN_OPTIONS",
            "LSAN_OPTIONS",
            "UBSAN_OPTIONS",
            "TSAN_OPTIONS",
            "ASAN_SYMBOLIZER_PATH",
        ):
            if key in os.environ:
                env[key] = os.environ[key]

        # When the address sanitizer is in use the example binaries are not
        # built with -fsanitize=address, so libasan only enters the process via
        # librocprof-sys-dl.so as a transitive DT_NEEDED. Asan refuses to
        # initialize unless its runtime is first in the link order, so prepend
        # libasan to LD_PRELOAD; rocprof-sys-run later appends librocprof-sys-dl
        # via update_mode::APPEND, preserving "asan first".
        asan_library = os.environ.get("ASAN_LIBRARY")
        if asan_library:
            existing = os.environ.get("LD_PRELOAD", "")
            env["LD_PRELOAD"] = f"{asan_library}:{existing}" if existing else asan_library

        return env

    def get_base_environment(self) -> dict[str, str]:
        """Get base environment variables for test execution."""
        return {
            "ROCPROFSYS_DEFAULT_MIN_INSTRUCTIONS": "64",
            "ROCPROFSYS_CI": "ON",
            "ROCPROFSYS_CI_TIMEOUT": os.environ.get("ROCPROFSYS_CI_TIMEOUT", "300"),
            "ROCPROFSYS_CONFIG_FILE": "",
            "ROCPROFSYS_TRACE": "ON",
            "ROCPROFSYS_PROFILE": "ON",
            "ROCPROFSYS_USE_SAMPLING": "ON",
            "ROCPROFSYS_USE_PROCESS_SAMPLING": "ON",
            "ROCPROFSYS_TIME_OUTPUT": "OFF",
            "ROCPROFSYS_FILE_OUTPUT": "ON",
            "ROCPROFSYS_USE_PID": "OFF",
            "ROCPROFSYS_LOG_LEVEL": "info",
            "ROCPROFSYS_SAMPLING_FREQ": "300",
            "ROCPROFSYS_SAMPLING_DELAY": "0.05",
            "ROCPROFSYS_SAMPLING_GPUS": "all",
            "OMP_PROC_BIND": "spread",
            "OMP_PLACES": "threads",
            "OMP_NUM_THREADS": "2",
            "LD_LIBRARY_PATH": self.get_library_path(),
        }

    def get_base_binary_environment(self) -> dict[str, str]:
        """Get base environment variables for rocprof-sys binary test execution."""
        return {
            "ROCPROFSYS_CI": "ON",
            "ROCPROFSYS_CI_TIMEOUT": os.environ.get("ROCPROFSYS_CI_TIMEOUT", "300"),
            "ROCPROFSYS_TRACE": "ON",
            "ROCPROFSYS_PROFILE": "ON",
            "ROCPROFSYS_USE_SAMPLING": "ON",
            "ROCPROFSYS_TIME_OUTPUT": "OFF",
            "ROCPROFSYS_USE_PID": "OFF",
            "ROCPROFSYS_LOG_LEVEL": "info",
            "LD_LIBRARY_PATH": self.get_library_path(),
            "ROCPROFSYS_CONFIG_FILE": "",
        }

    def get_base_python_environment(self) -> dict[str, str]:
        return {
            "ROCPROFSYS_CI": "ON",
            "ROCPROFSYS_CI_TIMEOUT": os.environ.get("ROCPROFSYS_CI_TIMEOUT", "300"),
            "ROCPROFSYS_TRACE": "ON",
            "ROCPROFSYS_PROFILE": "ON",
            "ROCPROFSYS_USE_SAMPLING": "OFF",
            "ROCPROFSYS_USE_PROCESS_SAMPLING": "ON",
            "ROCPROFSYS_TIME_OUTPUT": "OFF",
            "ROCPROFSYS_TREE_OUTPUT": "OFF",
            "ROCPROFSYS_USE_PID": "OFF",
            "ROCPROFSYS_TIMEMORY_COMPONENTS": "wall_clock,trip_count",
            "PYTHONPATH": (
                str(self.rocprofsys_site_packages)
                if self.rocprofsys_site_packages
                else ""
            ),
            "ROCPROFSYS_CONFIG_FILE": "",
            "LD_LIBRARY_PATH": self.get_library_path(),
        }

    def get_base_causal_environment(self) -> dict[str, str]:
        return {
            "ROCPROFSYS_CI": "ON",
            "ROCPROFSYS_CI_TIMEOUT": os.environ.get("ROCPROFSYS_CI_TIMEOUT", "300"),
            "ROCPROFSYS_USE_PID": "OFF",
            "ROCPROFSYS_THREAD_POOL_SIZE": "0",
            "ROCPROFSYS_VERBOSE": "1",
            "ROCPROFSYS_LOG_LEVEL": "info",
            "ROCPROFSYS_DL_VERBOSE": "0",
            "ROCPROFSYS_DEBUG_SETTINGS": "0",
            "LD_LIBRARY_PATH": self.get_library_path(),
            "ROCPROFSYS_CONFIG_FILE": "",
        }


def _find_rocm_path(optional: bool = False) -> Optional[Path]:
    """Find ROCm installation path.

    Args:
        optional: If True, return None instead of raising when ROCm is not found.
    """
    for candidate in [
        os.environ.get("ROCM_PATH"),
        "/opt/rocm",
        "/usr/local/rocm",
    ]:
        if candidate and Path(candidate).exists():
            return Path(candidate).resolve()
    if optional:
        return None
    raise FileNotFoundError(
        "Could not find ROCm installation. Set ROCM_PATH environment variable."
    )


def _get_rocm_version(rocm_optional: bool = False) -> Optional[tuple[int, int, int]]:
    """Get the installed ROCm version as a tuple (major, minor, patch).

    Returns:
        Tuple of (major, minor, patch) or None if ROCm not found or version undetectable.
    """
    rocm_path = _find_rocm_path(optional=rocm_optional)
    if rocm_path is None:
        return None

    # Check .info/version file
    version_file = rocm_path / ".info" / "version"
    if not version_file.exists():
        # Try alternative location
        version_file = rocm_path / "share" / "rocm" / "version"

    if version_file.exists():
        try:
            version_str = version_file.read_text().strip()
            match = re.match(r"(\d+)\.(\d+)\.(\d+)", version_str)
            if match:
                return (int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except (OSError, ValueError):
            pass

    return None


def _find_executable(name: str, search_paths: list[Path]) -> Optional[Path]:
    """Find an executable in search paths or via PATH."""
    for search_dir in search_paths:
        exe = search_dir / name
        if exe.exists() and exe.is_file():
            return exe.resolve()

    # Fallback to PATH
    path_exe = shutil.which(name)
    if path_exe:
        return Path(path_exe)

    return None


def _find_rocprofsys_core_executables(
    search_paths: list[Path],
) -> dict[str, Optional[Path]]:
    """Return a dictionary of rocprofiler-systems executables and their paths.

    Throws a FileNotFoundError if any of the executables are not found.
    """
    rocprof_instrument = _find_executable("rocprof-sys-instrument", search_paths)
    rocprof_sample = _find_executable("rocprof-sys-sample", search_paths)
    rocprof_run = _find_executable("rocprof-sys-run", search_paths)
    rocprof_causal = _find_executable("rocprof-sys-causal", search_paths)
    rocprof_avail = _find_executable("rocprof-sys-avail", search_paths)

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

    return required_executables


def _find_rocprofsys_python(
    search_paths: list[Path], rocprofsys_build_dir: Path
) -> tuple[Optional[Path], Optional[Path]]:
    """Return the Python executable and rocprofsys site-package path.

    Both must be found, otherwise (None, None) is returned.
    """
    rocprof_python = _find_executable("rocprof-sys-python", search_paths)
    if not rocprof_python:
        return None, None

    # It is either in the agnostic path or in the versioned path
    agnostic_path = rocprofsys_build_dir / "lib" / "python" / "site-packages"
    if (agnostic_path / "rocprofsys").is_dir():
        return rocprof_python, agnostic_path

    # Only one versioned path will exist
    # Otherwise, it will be in the agnostic path
    for child in (rocprofsys_build_dir / "lib").iterdir():
        if child.is_dir() and re.match(r"python\d+\.\d+", child.name):
            site_packages = child / "site-packages"
            if (site_packages / "rocprofsys").is_dir():
                return rocprof_python, site_packages

    return None, None


def _merge_python_root_dirs(
    explicit_dirs: Optional[list[Path]],
) -> Optional[list[Path]]:
    """Merge explicit --python-root-dirs with ROCPROFSYS_PYTHON_HINTS env var.

    ROCPROFSYS_PYTHON_HINTS entries may point to bin/ directories or their parents;
    both are normalized to the parent directory for pythonX.Y lookup.
    """
    result: list[Path] = list(explicit_dirs or [])
    env_hints = os.environ.get("ROCPROFSYS_PYTHON_HINTS", "")
    if env_hints:
        for hint in env_hints.split(";"):
            hint = hint.strip()
            if hint:
                hint_path = Path(hint)
                parent = hint_path.parent if hint_path.name == "bin" else hint_path
                if parent not in result:
                    result.append(parent)
    return result or None


def discover_install_config(
    install_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    python_versions: Optional[list[str]] = None,
    python_root_dirs: Optional[list[Path]] = None,
    rocm_optional: bool = False,
) -> RocprofsysConfig:
    """Discover rocprofiler-systems installation configuration.

    Creates configuration for testing against installed binaries.

    Args:
        install_dir: Installation prefix (e.g., /opt/rocm or /usr/local)

    Returns:
        RocprofsysConfig configured for installed binaries

    Raises:
        FileNotFoundError: If build/installation dirs and executables are not found
    """

    if install_dir is None:
        env_install = os.environ.get("ROCPROFSYS_INSTALL_DIR")
        if env_install:
            install_dir = Path(env_install).resolve()
        else:
            _rocm_candidate = _find_rocm_path(optional=True)
            _install_candidates = [
                Path("/usr/local"),
                Path("/usr"),
                Path(
                    "/opt/rocprofiler-systems"
                ),  # Standard install location from README.md
            ]
            if _rocm_candidate is not None:
                _install_candidates.insert(0, _rocm_candidate)
            for candidate in _install_candidates:
                if (
                    candidate
                    and (candidate / "share" / "rocprofiler-systems" / "tests").is_dir()
                    and (
                        candidate / "share" / "rocprofiler-systems" / "examples"
                    ).is_dir()
                ):
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

    # Create a temporary directory for test outputs
    try:
        username = getpass.getuser()
    except Exception:
        username = str(os.getuid())

    if output_dir is None:
        output_dir = Path(tempfile.gettempdir()) / username / "rocprof-sys-pytest-output"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    rocm_path = _find_rocm_path(optional=rocm_optional)

    search_paths = [bin_dir]
    sys_execs = _find_rocprofsys_core_executables(search_paths)
    rocprofsys_python, rocprofsys_site_packages = _find_rocprofsys_python(
        search_paths, install_dir
    )

    return RocprofsysConfig(
        rocprofsys_build_dir=install_dir,
        rocprofsys_instrument=sys_execs["rocprof-sys-instrument"],
        rocprofsys_run=sys_execs["rocprof-sys-run"],
        rocprofsys_sample=sys_execs["rocprof-sys-sample"],
        rocprofsys_causal=sys_execs["rocprof-sys-causal"],
        rocprofsys_avail=sys_execs["rocprof-sys-avail"],
        rocm_path=rocm_path,
        rocprofsys_lib_dir=lib_dir,
        rocprofsys_bin_dir=bin_dir,
        rocprofsys_examples_dir=examples_dir,
        rocprofsys_tests_dir=tests_dir,
        rocpd_validation_rules=rocpd_validation_rules,
        test_output_dir=output_dir,
        rocm_version=_get_rocm_version(rocm_optional=rocm_optional),
        is_installed=True,
        rocprofsys_python=rocprofsys_python,
        rocprofsys_site_packages=rocprofsys_site_packages,
        _python_versions_hint=python_versions,
        _python_root_dirs_hint=_merge_python_root_dirs(python_root_dirs),
    )


def discover_build_config(
    build_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    python_versions: Optional[list[str]] = None,
    python_root_dirs: Optional[list[Path]] = None,
    rocm_optional: bool = False,
) -> RocprofsysConfig:
    """Discover rocprofiler-systems build configuration.

    Attempts to find the build directory and source directory automatically
    if not provided, checking common locations and environment variables.

    If no build directory is found but an installation is available,
    falls back to discover_install_config().

    Args:
        build_dir: Explicit build directory path

    Returns:
        RocprofsysConfig with discovered paths

    Raises:
        FileNotFoundError: If build/installation dirs and executables are not found
    """

    # Explicit install directory check
    if os.environ.get("ROCPROFSYS_INSTALL_DIR"):
        return discover_install_config(
            output_dir=output_dir,
            python_versions=python_versions,
            python_root_dirs=python_root_dirs,
            rocm_optional=rocm_optional,
        )

    # When running from pyz package (extracted to /tmp), fall back to install config
    # The pyz extracts to paths like /tmp/rocprofsys-tests-*/tests/rocprofsys/config.py
    current_file = Path(__file__).resolve()
    if str(current_file).startswith(tempfile.gettempdir()):
        return discover_install_config(
            python_versions=python_versions,
            python_root_dirs=python_root_dirs,
            rocm_optional=rocm_optional,
        )

    # All files should be in the build directory
    if build_dir is None:
        env_build = os.environ.get("ROCPROFSYS_BUILD_DIR")
        if env_build:
            build_dir = Path(env_build).resolve()
        else:
            build_dir = Path(__file__).resolve().parent.parent.parent.parent.parent.parent

    if build_dir is None or not build_dir.exists():
        raise FileNotFoundError(
            "Could not find build directory or installation. Set one of:\n"
            "  - ROCPROFSYS_BUILD_DIR: Path to build directory\n"
            "  - ROCPROFSYS_INSTALL_DIR: Path to installation prefix"
        )

    rocm_path = _find_rocm_path(optional=rocm_optional)

    bin_dir = build_dir / "bin"
    lib_dir = build_dir / "lib"

    search_paths = [bin_dir]
    sys_execs = _find_rocprofsys_core_executables(search_paths)
    rocprofsys_python, rocprofsys_site_packages = _find_rocprofsys_python(
        search_paths, build_dir
    )

    share_path = build_dir / "share" / "rocprofiler-systems"

    if output_dir is None:
        output_dir = build_dir / "rocprof-sys-pytest-output"
    else:
        output_dir = Path(output_dir)

    tests_dir = share_path / "tests"

    return RocprofsysConfig(
        rocprofsys_build_dir=build_dir,
        rocprofsys_instrument=sys_execs["rocprof-sys-instrument"],
        rocprofsys_run=sys_execs["rocprof-sys-run"],
        rocprofsys_sample=sys_execs["rocprof-sys-sample"],
        rocprofsys_causal=sys_execs["rocprof-sys-causal"],
        rocprofsys_avail=sys_execs["rocprof-sys-avail"],
        rocm_path=rocm_path,
        rocprofsys_lib_dir=lib_dir,
        rocprofsys_bin_dir=bin_dir,
        rocprofsys_examples_dir=build_dir,  # Example binaries are (almost always) in root of build directory
        rocprofsys_tests_dir=tests_dir,
        rocpd_validation_rules=tests_dir / "rocpd-validation-rules",
        test_output_dir=output_dir,
        rocm_version=_get_rocm_version(rocm_optional=rocm_optional),
        is_installed=False,
        rocprofsys_python=rocprofsys_python,
        rocprofsys_site_packages=rocprofsys_site_packages,
        _python_versions_hint=python_versions,
        _python_root_dirs_hint=_merge_python_root_dirs(python_root_dirs),
    )
