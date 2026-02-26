# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

from __future__ import annotations

from dataclasses import dataclass
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

    rocm_path: Optional[Path] = None
    mpiexec: Optional[Path] = None
    rocprofsys_tests_dir: Optional[Path] = None
    rocprofsys_examples_dir: Optional[Path] = None
    rocprofsys_avail: Optional[Path] = None

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
            mpiexec=config.mpiexec,
            rocprofsys_tests_dir=config.rocprofsys_tests_dir,
            rocprofsys_examples_dir=config.rocprofsys_examples_dir,
            rocprofsys_avail=config.rocprofsys_avail,
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
        if self.mpiexec is None:
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
                [self.mpiexec, "-n", "2", mpi_send_recv],
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
