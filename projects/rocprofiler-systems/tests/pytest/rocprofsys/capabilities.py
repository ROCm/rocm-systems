# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Optional
import os
import shutil
import subprocess
import re


@dataclass
class SystemCapabilities:
    """
    Class that can be used to check various capabilities of the
    system. Primarily used to determine if tests should be ran.
    Tied to a RocprofsysConfig instance.
    """

    rocprofsys_build_dir: Path
    rocprofsys_tests_dir: Path
    rocprofsys_examples_dir: Path
    rocprofsys_avail: Path
    is_installed: bool
    rocm_path: Optional[Path]
    rocprofsys_site_packages: Optional[Path]
    _python_versions_hint: Optional[list[str]] = field(default=None, repr=False)
    _python_root_dirs_hint: Optional[list[Path]] = field(default=None, repr=False)

    @classmethod
    def from_config(cls, config) -> SystemCapabilities:
        """Create SystemCapabilities from RocprofsysConfig.

        Args:
            config: The RocprofsysConfig instance to extract paths from.

        Returns:
            A new SystemCapabilities instance with paths from config.
        """
        return cls(
            rocm_path=config.rocm_path,
            rocprofsys_build_dir=config.rocprofsys_build_dir,
            rocprofsys_tests_dir=config.rocprofsys_tests_dir,
            rocprofsys_examples_dir=config.rocprofsys_examples_dir,
            rocprofsys_avail=config.rocprofsys_avail,
            rocprofsys_site_packages=config.rocprofsys_site_packages,
            _python_versions_hint=config._python_versions_hint,
            _python_root_dirs_hint=config._python_root_dirs_hint,
            is_installed=config.is_installed,
        )

    @cached_property
    def mpi_implementation(self) -> str:
        """Get the name of the MPI implementation."""
        mpicc = shutil.which("mpicc")
        if not mpicc:
            return "unknown"

        def _get_include_path(args: list[str]) -> str:
            try:
                result = subprocess.run(
                    [mpicc] + args,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                return result.stdout if result.returncode == 0 else ""
            except (subprocess.SubprocessError, OSError):
                return ""

        include_paths = _get_include_path(
            ["--showme:compile"]
        ) or _get_include_path(  # OpenMPI-style
            ["-show"]
        )  # MPICH-style
        # Check include paths for implementation markers
        if "openmpi" in include_paths.lower():
            return "openmpi"
        elif "mpich" in include_paths.lower():
            return "mpich"

        return "unknown"

    @cached_property
    def default_nic(self) -> Optional[str]:
        """Get the name of the default NIC

        Returns: Result of executing the get_default_nic.sh script
        """
        get_default_nic_script = self.rocprofsys_tests_dir / "get_default_nic.sh"
        if not get_default_nic_script.exists():
            return None
        try:
            result = subprocess.run(
                [get_default_nic_script],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return None
            return result.stdout.strip()
        except (subprocess.SubprocessError, OSError):
            return None

    @cached_property
    def ai_nic_devices(self) -> list[str]:
        """Get the unique AI NIC device names reported by AMD SMI.

        Runs ``amd-smi static`` and extracts every distinct NETDEV value.
        Returns an empty list when AMD SMI is unavailable or reports no NICs.

        Example output line from ``amd-smi static``:
        ``NETDEV: enp137s0np0``
        """
        amd_smi = shutil.which("amd-smi")
        if not amd_smi:
            return []
        try:
            result = subprocess.run(
                [amd_smi, "static"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode != 0:
                return []
            seen: set[str] = set()
            devices: list[str] = []
            for line in result.stdout.splitlines():
                if "netdev" in line.lower():
                    colon_idx = line.find(":")
                    if colon_idx != -1:
                        name = line[colon_idx + 1 :].strip()
                        if name and name not in seen:
                            seen.add(name)
                            devices.append(name)
            return devices
        except (subprocess.SubprocessError, OSError, subprocess.TimeoutExpired):
            return []

    @cached_property
    def papi_nic_events(self) -> Optional[str]:
        """Get the list of all events that we want PAPI to record.

        Returns: Result of executing the generate_papi_nic_events.sh script
        """
        generate_papi_nic_events_script = (
            self.rocprofsys_tests_dir / "generate_papi_nic_events.sh"
        )
        if not generate_papi_nic_events_script.exists():
            return None
        try:
            result = subprocess.run(
                [generate_papi_nic_events_script],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return None
            return result.stdout.strip()
        except (subprocess.SubprocessError, OSError):
            return None

    @cached_property
    def ucx_availability(self) -> bool:
        mpiexec_exec = self.mpiexec_exec
        if mpiexec_exec is None:
            return False
        mpi_send_recv = self.rocprofsys_examples_dir / "mpi-send-recv"
        if not mpi_send_recv.exists():
            return False

        # Force OpenMPI to use UCX transport
        ucx_env = os.environ.copy()
        ucx_env.update(
            {
                "OMPI_MCA_pml": "ucx",
                "OMPI_MCA_osc": "ucx",
                "OMPI_MCA_pml_ucx_tls": "tcp,self",
                "OMPI_MCA_pml_ucx_devices": "any",
            }
        )

        try:
            result = subprocess.run(
                [mpiexec_exec, "-n", "2", mpi_send_recv],
                capture_output=True,
                text=True,
                timeout=10,
                env=ucx_env,
            )
            if result.returncode != 0:
                return False
        except (subprocess.SubprocessError, OSError):
            return False

        fail_regex = [
            r"PML ucx cannot be selected",
            r"UCX is not available",
            r"No UCX support found",
            r"Failed to select",
            r"No components were able to be opened in the pml framework",
        ]

        combined_output = (result.stdout or "") + (result.stderr or "")
        for regex in fail_regex:
            if re.search(regex, combined_output):
                return False

        return True

    @cached_property
    def num_procs(self) -> int:
        """Get the number of available processors."""
        num_procs_real = os.cpu_count()
        if num_procs_real is None:
            return 2
        return num_procs_real

    @cached_property
    def ptrace_scope(self) -> int:
        """Get the value of the ptrace_scope kernel parameter."""
        if not Path("/proc/sys/kernel/yama/ptrace_scope").exists():
            return 3
        try:
            return int(Path("/proc/sys/kernel/yama/ptrace_scope").read_text().strip())
        except (OSError, ValueError):
            return 3

    @cached_property
    def perf_event_paranoid(self) -> int:
        """Get the value of the perf_event_paranoid kernel parameter."""
        if not Path("/proc/sys/kernel/perf_event_paranoid").exists():
            return 4
        try:
            return int(Path("/proc/sys/kernel/perf_event_paranoid").read_text().strip())
        except (OSError, ValueError):
            return 4

    @cached_property
    def cap_sys_admin(self) -> bool:
        """Get the value of the CAP_SYS_ADMIN capability."""
        capchk = self.rocprofsys_tests_dir / "rocprof-sys-capchk"
        if not capchk.exists():
            return False
        try:
            result = subprocess.run(
                [capchk, "CAP_SYS_ADMIN", "effective"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return False
            return result.stdout.strip() == "1"
        except (subprocess.SubprocessError, OSError):
            return False

    @cached_property
    def cap_perfmon(self) -> bool:
        """Get the value of the CAP_PERFMON capability."""
        capchk = self.rocprofsys_tests_dir / "rocprof-sys-capchk"
        if not capchk.exists():
            return False
        try:
            result = subprocess.run(
                [capchk, "CAP_PERFMON", "effective"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return False

            return result.stdout.strip() == "1"
        except (subprocess.SubprocessError, OSError):
            return False

    @cached_property
    def papi_availability(self) -> bool:
        """Check if PAPI is built into rocprofiler-systems.

        Returns True only if both papi_array and papi_vector components are available.
        """
        try:
            result = subprocess.run(
                [str(self.rocprofsys_avail), "--components"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return False
            output = result.stdout
            papi_array_available = False
            papi_vector_available = False

            for line in output.splitlines():
                if "papi_array" in line and "true" in line.lower():
                    papi_array_available = True
                if "papi_vector" in line and "true" in line.lower():
                    papi_vector_available = True

            return papi_array_available and papi_vector_available
        except (subprocess.SubprocessError, OSError, subprocess.TimeoutExpired):
            return False

    @cached_property
    def _supported_python_versions_and_executables(
        self,
    ) -> tuple[Optional[list[str]], Optional[list[Path]]]:
        """Return the list of supported python versions and executables"""
        versions, executables = _get_supported_python_versions_and_executables(
            self.rocprofsys_site_packages,
            self._python_versions_hint,
            self._python_root_dirs_hint,
        )
        return versions, executables

    @cached_property
    def supported_python_versions(self) -> Optional[list[str]]:
        """Return the list of supported python versions"""
        return self._supported_python_versions_and_executables[0]

    @cached_property
    def supported_python_executables(self) -> Optional[list[Path]]:
        """Return the list of supported python executables"""
        return self._supported_python_versions_and_executables[1]

    def get_python_executable(self, version: str) -> Path:
        """Return the Python executable path for the given version (e.g. '3.10').

        Raises:
            FileNotFoundError: If no Python is configured or the version is not found.
        """
        if not self.supported_python_versions or not self.supported_python_executables:
            raise FileNotFoundError("No Python versions/executables configured")
        try:
            idx = self.supported_python_versions.index(version)
            return self.supported_python_executables[idx]
        except ValueError:
            raise FileNotFoundError(
                f"Python version '{version}' not found. Available: {', '.join(self.supported_python_versions)}"
            )

    @cached_property
    def is_inside_docker(self) -> bool:
        """Check if the system is running inside a Docker container."""
        if os.path.exists("/.dockerenv"):
            return True
        try:
            with open("/proc/1/cgroup") as f:
                cgroup = f.read()
                if "docker" in cgroup or "containerd" in cgroup:
                    return True
        except (FileNotFoundError, PermissionError):
            pass
        return False

    @cached_property
    def oshrun_exec(self) -> Optional[Path]:
        """Get the path to the oshrun executable."""
        result = shutil.which("oshrun")
        return Path(result) if result else None

    @cached_property
    def oshrun_version(self) -> Optional[tuple[int, ...]]:
        """Get the parsed version of oshrun as a tuple (major, minor)"""
        if not self.oshrun_exec:
            return None
        try:
            result = subprocess.run(
                [str(self.oshrun_exec), "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return None
            version_str = result.stdout.strip() or result.stderr.strip()
            match = re.search(r"(\d+)\.(\d+)", version_str)
            if not match:
                return None
            return (int(match.group(1)), int(match.group(2)))
        except (subprocess.SubprocessError, OSError):
            return None

    @cached_property
    def rocprofiler_sdk_version(self) -> Optional[tuple[int, int, int]]:
        """Return rocprofiler-sdk (major, minor, patch) from ``version.h`` under ROCm.

        ROCm root resolution: configured ``rocm_path`` when set, else the
        ``ROCM_PATH`` environment variable if set, else ``/opt/rocm``. Parses
        ``ROCPROFILER_SDK_VERSION_STRING`` from
        ``<root>/include/rocprofiler-sdk/version.h``.

        Returns:
            ``(major, minor, patch)`` or ``None`` if the file is missing or unparsable.
        """
        if self.rocm_path is not None:
            root = self.rocm_path
        else:
            env_root = os.environ.get("ROCM_PATH")
            root = Path(env_root) if env_root else Path("/opt/rocm")
        return _rocprofiler_sdk_version_from_version_h(
            root / "include" / "rocprofiler-sdk" / "version.h"
        )

    @cached_property
    def julia_exec(self) -> Optional[Path]:
        """Get the path to the Julia executable."""
        path = shutil.which("julia")
        return Path(path) if path else None

    @cached_property
    def mpiexec_exec(self) -> Optional[Path]:
        """Find MPI launcher executable."""
        for candidate in ["mpiexec", "mpirun"]:
            path = shutil.which(candidate)
            if path:
                return Path(path)
        return None

    def target_support_mpi(self, target_path: Path) -> bool:
        """Check if the target supports MPI by checking if the target is linked to MPI."""
        if not target_path.exists():
            return False
        ldd_exec = shutil.which("ldd")
        if not ldd_exec:
            return False
        try:
            result = subprocess.run(
                [ldd_exec, str(target_path)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return False
            return "mpi" in result.stdout.lower()
        except (subprocess.SubprocessError, OSError):
            return False


_ROCPROFILER_SDK_VERSION_H_RE = re.compile(
    r'^#define\s+ROCPROFILER_SDK_VERSION_STRING\s+"(\d+)\.(\d+)\.(\d+)"',
    re.MULTILINE,
)


def _rocprofiler_sdk_version_from_version_h(path: Path) -> Optional[tuple[int, int, int]]:
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    m = _ROCPROFILER_SDK_VERSION_H_RE.search(text)
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)))


def _get_python_version(executable: Path) -> Optional[str]:
    """Get major.minor Python version from an executable."""
    try:
        result = subprocess.run(
            [str(executable), "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            output = result.stdout.strip() or result.stderr.strip()
            match = re.match(r"Python (\d+\.\d+)", output)
            if match:
                return match.group(1)
    except (subprocess.SubprocessError, OSError):
        pass
    return None


def _find_python_in_dirs(version: str, root_dirs: list[Path]) -> Optional[Path]:
    """Search for a specific Python version across multiple root directories."""
    for root_dir in root_dirs:
        for name in [f"python{version}", "python3", "python"]:
            candidate = root_dir / "bin" / name
            if not candidate.exists():
                candidate = root_dir / name
            if candidate.exists() and candidate.is_file():
                detected = _get_python_version(candidate)
                if detected and detected == version:
                    return candidate
    return None


def _get_supported_python_versions_and_executables(
    rocprofsys_site_packages: Optional[Path],
    python_versions_hint: Optional[list[str]],
    python_root_dirs_hint: Optional[list[Path]],
) -> tuple[Optional[list[str]], Optional[list[Path]]]:
    """Return the list of supported python versions and executables

    A supported python version is one that has a corresponding libpyrocprofsys.<IMPL>-<VERSION>-<ARCH>-<OS>-<ABI>.so.

    When both python_versions_hint and python_root_dirs_hint are provided,
    each version is searched for across ALL root directories (not 1:1 paired).
    Falls back to PATH lookup if not found in any root directory.
    """
    if not rocprofsys_site_packages:
        return None, None

    found_versions: list[str] = []
    found_executables: list[Path] = []

    if python_versions_hint:
        for version in python_versions_hint:
            found: Optional[Path] = None
            if python_root_dirs_hint:
                found = _find_python_in_dirs(version, python_root_dirs_hint)
            if found is None:
                exe = shutil.which(f"python{version}")
                if exe:
                    exe_path = Path(exe)
                    detected = _get_python_version(exe_path)
                    if detected and detected.startswith(version):
                        found = exe_path
            if found is not None:
                detected = _get_python_version(found)
                if detected:
                    found_versions.append(detected)
                    found_executables.append(found)
    else:
        import sys

        current_exe = Path(sys.executable)
        version = _get_python_version(current_exe)
        if version:
            found_versions.append(version)
            found_executables.append(current_exe)
        else:
            exe = shutil.which("python3")
            if exe:
                exe_path = Path(exe)
                version = _get_python_version(exe_path)
                if version:
                    found_versions.append(version)
                    found_executables.append(exe_path)

    if len(found_versions) != len(found_executables):
        raise RuntimeError(
            f"found_versions ({len(found_versions)}) and found_executables "
            f"({len(found_executables)}) length mismatch"
        )
    if not found_versions:
        return None, None

    # Filter out based on the rocprofsys site packages
    supported_versions: list[str] = []
    supported_executables: list[Path] = []
    rocprofsys_pkg = rocprofsys_site_packages / "rocprofsys"
    for version, executable in zip(found_versions, found_executables):
        version_tag = version.replace(".", "")
        if any(rocprofsys_pkg.glob(f"libpyrocprofsys.*-{version_tag}-*.so")):
            supported_versions.append(version)
            supported_executables.append(executable)
    return supported_versions or None, supported_executables or None
