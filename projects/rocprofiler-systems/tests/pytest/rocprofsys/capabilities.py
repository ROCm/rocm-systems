# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import functools
import os
import shutil
import subprocess
import re

from .cache import persistent_cache, persistent_cached_property


def _get_amdsmi_version_output(rocm_path: Optional[Path] = None) -> Optional[str]:
    """Return combined stdout/stderr of ``amd-smi version`` if available, else None."""
    amdsmi_exe = None
    if rocm_path is not None:
        candidate = Path(rocm_path) / "bin" / "amd-smi"
        if candidate.exists():
            amdsmi_exe = str(candidate)
    if not amdsmi_exe:
        amdsmi_exe = shutil.which("amd-smi")
    if not amdsmi_exe:
        rocm_env = os.environ.get("ROCM_PATH", "/opt/rocm")
        candidate = Path(rocm_env) / "bin" / "amd-smi"
        if candidate.exists():
            amdsmi_exe = str(candidate)
    if not amdsmi_exe:
        return None
    try:
        result = subprocess.run(
            [amdsmi_exe, "version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return (result.stdout or "") + " " + (result.stderr or "")
    except (subprocess.SubprocessError, OSError):
        return None


def get_amdsmi_version(rocm_path: Optional[Path] = None) -> Optional[tuple[int, int]]:
    """Return (major, minor) of amd-smi if available, else None."""
    raw = _get_amdsmi_version_output(rocm_path)
    if raw is None:
        return None
    # e.g. Output from `amd-smi version` = "AMDSMI Tool: 26.4.0+... | AMDSMI Library version: 26.4.0 | ROCm version: 7.13.0 ..."
    m = re.search(r"AMDSMI\s+(?:Tool|Library version):\s*(\d+)\.(\d+)", raw)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    return None


def get_amdgpu_version(
    rocm_path: Optional[Path] = None,
) -> Optional[tuple[int, int, int]]:
    """Return (major, minor, patch) of the amdgpu driver if available, else None."""
    raw = _get_amdsmi_version_output(rocm_path)
    if raw is None:
        return None
    # amdgpu version is "major.minor[.patch[.build]]"; e.g. "6.19.14.31400000"
    # or "6.19.4". Compare on (major, minor, patch) only — the build suffix is
    # ignored, and an absent patch is treated as .0.
    m = re.search(r"amdgpu version:\s*(\d+)\.(\d+)(?:\.(\d+))?", raw)
    if m:
        patch = int(m.group(3)) if m.group(3) is not None else 0
        return (int(m.group(1)), int(m.group(2)), patch)
    return None


# capchk builds its capability table behind #if defined(CAP_*), so a capability
# newer than the kernel headers it was compiled against is reported as unknown
# rather than absent.
#
# The list here is the set of capabilities capchk may legitimately not know
# about; any other unknown name is a typo in the caller.
_OPTIONAL_CAPABILITIES = frozenset({"CAP_PERFMON"})  # Introduced in kernel 5.8


def find_roctx_site_packages(
    rocm_path: Optional[Path], python_version: str
) -> Optional[Path]:
    """Return the ROCm-provided roctx site-packages directory for a Python version.

    rocprofiler-sdk builds and installs its ``roctx`` Python bindings
    (``libpyroctx.<abi>.so``) per interpreter into a versioned directory under
    the ROCm install tree, e.g. ``<rocm_path>/lib/python3.11/site-packages``.
    Unlike rocprofsys's own bindings, these are not consolidated into a single
    ABI-agnostic directory, so the correct versioned directory must be
    resolved per Python version.

    Args:
        rocm_path: Path to the ROCm installation, or None.
        python_version: Python version string, e.g. "3.11".

    Returns:
        Path to the site-packages directory containing the ``roctx`` package
        for the given version, or None if not found.
    """
    if not rocm_path:
        return None
    candidates = (
        rocm_path / lib_name / f"python{python_version}" / "site-packages" / "roctx"
        for lib_name in ("lib", "lib64")
    )
    # The package directory alone isn't sufficient: some ROCm packaging variants
    # ship the __init__.py without the compiled extension it imports
    # (libpyroctx.<abi>.so), which would still raise on import.
    return next(
        (
            roctx_dir.parent
            for roctx_dir in candidates
            if (roctx_dir / "__init__.py").is_file()
            and any(roctx_dir.glob("libpyroctx.*"))
        ),
        None,
    )


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

    @persistent_cached_property
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

    @persistent_cached_property
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

    @persistent_cached_property
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

    @persistent_cached_property
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

    @persistent_cached_property
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

    @persistent_cached_property
    def num_procs(self) -> int:
        """Number of processors available to this process.

        Uses sched_getaffinity so Slurm/cgroup/taskset limits match CMake
        ProcessorCount and runtime thread-pool sizing.
        """
        try:
            affinity = os.sched_getaffinity(0)
            count = len(affinity)
        except (AttributeError, NotImplementedError, OSError):
            count = os.cpu_count() or 0

        return count if count > 0 else 2

    @persistent_cached_property
    def ptrace_scope(self) -> Optional[int]:
        """Get the value of the yama ``ptrace_scope`` kernel parameter.

        Returns ``None`` when the sysctl does not exist, which means the yama
        LSM is not active and therefore places no restriction on ptrace.
        """
        scope_path = Path("/proc/sys/kernel/yama/ptrace_scope")
        if not scope_path.exists():
            return None
        try:
            return int(scope_path.read_text().strip())
        except (OSError, ValueError):
            return 3

    @persistent_cached_property
    def perf_event_paranoid(self) -> int:
        """Get the value of the perf_event_paranoid kernel parameter."""
        if not Path("/proc/sys/kernel/perf_event_paranoid").exists():
            return 4
        try:
            return int(Path("/proc/sys/kernel/perf_event_paranoid").read_text().strip())
        except (OSError, ValueError):
            return 4

    # Results for the relevant caps are cached in functions below
    def _has_capability(self, name: str, capability_set: str = "effective") -> bool:
        """Return True if this process holds *name* in *capability_set*.

        Exit codes of rocprof-sys-capchk, which is where its answer lives:

          - 0: the capability is held
          - 1: the capability is absent
          - 2: the capability name is unknown
          - 3: the capability set name is unknown

        2 and 3 mean the caller asked for something that does not exist, so they
        raise rather than report the capability as absent and silently skip the
        tests that need it. The exception is a capability newer than the kernel
        headers capchk was built against, which is a real system difference
        rather than a mistake; see ``_OPTIONAL_CAPABILITIES``.

        Stdout is a human-readable sentence and must never be parsed
        Changes here should be reflected in tests/rocprof-sys-capchk.cpp
        """
        capchk = self.rocprofsys_tests_dir / "rocprof-sys-capchk"
        if not capchk.exists():
            return False
        try:
            result = subprocess.run(
                [capchk, name, capability_set],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (subprocess.SubprocessError, OSError):
            return False

        if result.returncode == 3:
            raise ValueError(
                f"rocprof-sys-capchk does not support capability set "
                f"{capability_set!r}"
            )
        if result.returncode == 2 and name.upper() not in _OPTIONAL_CAPABILITIES:
            raise ValueError(f"rocprof-sys-capchk does not support capability {name!r}")
        return result.returncode == 0

    @persistent_cached_property
    def cap_sys_admin(self) -> bool:
        """Whether CAP_SYS_ADMIN is in the effective capability set."""
        return self._has_capability("CAP_SYS_ADMIN")

    @persistent_cached_property
    def cap_perfmon(self) -> bool:
        """Whether CAP_PERFMON is in the effective capability set."""
        return self._has_capability("CAP_PERFMON")

    @persistent_cached_property
    def cap_sys_ptrace(self) -> bool:
        """Whether CAP_SYS_PTRACE is in the effective capability set."""
        return self._has_capability("CAP_SYS_PTRACE")

    @persistent_cached_property
    def perf_events_usable(self) -> bool:
        """Whether perf_event_open-based features can actually be used.

        This gates anything that opens Linux perf events, including PAPI
        hardware/software counters and overflow sampling. It mirrors the
        runtime gate in ``source/lib/core/config.cpp``
        """
        return self.perf_event_paranoid <= 2 or self.cap_perfmon or self.cap_sys_admin

    @persistent_cached_property
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

    @persistent_cached_property
    def max_threads(self) -> int:
        """Compile-time limit on the total number of threads profiled per process.

        The limit is cumulative over the lifetime of the process rather than a
        cap on concurrency: every thread that is created consumes a slot, and
        that slot is not reused when the thread exits.

        This is the ``ROCPROFSYS_MAX_THREADS`` CMake constant baked into the
        binaries at build time. It is queried from
        ``rocprof-sys-avail --max-threads``

        Returns 0 when the value cannot be determined, so callers degrade
        gracefully instead of aborting the whole configuration step.
        """
        try:
            result = subprocess.run(
                [str(self.rocprofsys_avail), "--max-threads"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return 0
        except (subprocess.SubprocessError, OSError):
            return 0
        # Must stay in sync with the exact line printed by rocprof-sys-avail's
        # `--max-threads` action (source/bin/rocprof-sys-avail/avail.cpp).
        match = re.search(
            r"total number of threads \(ROCPROFSYS_MAX_THREADS\):\s*(\d+)", result.stdout
        )
        return int(match.group(1)) if match else 0

    # ---------------------------------------------------------------------------
    # Do NOT make this a persistent_cached_property: the result depends on the
    # per-process --python-versions / --python-root-dirs hints, so it must not be
    # shared across processes
    @functools.cached_property
    def _supported_python_versions_and_executables(
        self,
    ) -> tuple[Optional[list[str]], Optional[list[Path]]]:
        """Return the list of supported python versions and executables"""
        return _get_supported_python_versions_and_executables(
            self.rocprofsys_site_packages,
            self._python_versions_hint,
            self._python_root_dirs_hint,
        )

    @property
    def supported_python_versions(self) -> Optional[list[str]]:
        """Return the list of supported python versions"""
        return self._supported_python_versions_and_executables[0]

    @property
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

    def roctx_site_packages(self, python_version: str) -> Optional[Path]:
        """ROCm's roctx site-packages directory for ``python_version``, if usable.

        See :func:`find_roctx_site_packages`.
        """
        return find_roctx_site_packages(self.rocm_path, python_version)

    def python_lib_dir(self, python_version: str) -> Optional[Path]:
        """Return the interpreter's own ``lib`` directory, or None if absent.

        On some rocprofiler-sdk builds the compiled ``libpyroctx.<abi>.so``
        extension resolves ``libpython<version>.so`` through the loader search
        path rather than through an rpath/``$ORIGIN`` entry, so this directory
        has to be on ``LD_LIBRARY_PATH`` for the import to succeed even though
        every file is present on disk.
        """
        try:
            python_executable = self.get_python_executable(python_version)
        except FileNotFoundError:
            return None
        # Typical layout: <env_root>/bin/python3.X -> <env_root>/lib
        lib_dir = python_executable.parent.parent / "lib"
        return lib_dir if lib_dir.is_dir() else None

    @persistent_cache("cap.roctx_available_for", method=True)
    def roctx_available_for(self, python_version: str) -> bool:
        """Return whether ROCm's roctx Python bindings are installed AND
        importable for this version.

        File presence alone isn't sufficient, so this invokes the target
        interpreter with the same roctx site-packages and interpreter lib
        directory that ``PythonRunner`` adds for the real test run. Only the
        base environment differs - the probe extends this process's
        environment, the runner extends its layered one - and the runner's
        search paths are a superset, so a negative here is conservative.
        """
        site_packages = self.roctx_site_packages(python_version)
        if site_packages is None:
            return False
        try:
            python_executable = self.get_python_executable(python_version)
        except FileNotFoundError:
            return False

        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            p for p in (env.get("PYTHONPATH", ""), str(site_packages)) if p
        )
        lib_dir = self.python_lib_dir(python_version)
        if lib_dir is not None:
            env["LD_LIBRARY_PATH"] = os.pathsep.join(
                p for p in (env.get("LD_LIBRARY_PATH", ""), str(lib_dir)) if p
            )
        try:
            result = subprocess.run(
                [str(python_executable), "-c", "import roctx"],
                capture_output=True,
                text=True,
                timeout=10,
                env=env,
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, OSError):
            return False

    # ---------------------------------------------------------------------------

    @persistent_cached_property
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

    @persistent_cached_property
    def oshrun_exec(self) -> Optional[Path]:
        """Get the path to the oshrun executable."""
        result = shutil.which("oshrun")
        return Path(result) if result else None

    @persistent_cached_property
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

    @persistent_cached_property
    def oshrun_strips_double_dash(self) -> bool:
        """Return True if this oshrun strips the first '--' from application argv.

        Probes the live binary by running:
            oshrun -n 1 probe.sh -- SENTINEL
        and checking whether the script receives 'SENTINEL' (stripped) or '--'
        (preserved).  Falls back to False when oshrun is absent or the probe
        fails.
        """
        if not self.oshrun_exec:
            return False
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".sh", delete=False
        ) as probe_file:
            probe_file.write("#!/bin/sh\nprintf '%s\\n' \"$1\"\n")
            probe_path = probe_file.name
        try:
            os.chmod(probe_path, 0o700)
            for extra in ([], ["--allow-run-as-root"]):
                cmd = (
                    [str(self.oshrun_exec)]
                    + extra
                    + ["-n", "1", probe_path, "--", "SENTINEL"]
                )
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    return result.stdout.strip() == "SENTINEL"
            return False
        except (subprocess.SubprocessError, OSError):
            return False
        finally:
            os.unlink(probe_path)

    @persistent_cached_property
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

    @persistent_cached_property
    def julia_exec(self) -> Optional[Path]:
        """Get the path to the Julia executable."""
        path = shutil.which("julia")
        return Path(path) if path else None

    @persistent_cached_property
    def mpiexec_exec(self) -> Optional[Path]:
        """Find MPI launcher executable."""
        for candidate in ["mpiexec", "mpirun"]:
            path = shutil.which(candidate)
            if path:
                return Path(path)
        return None

    @persistent_cache("cap.target_support_mpi", method=True)
    def target_support_mpi(self, target_path: Path) -> bool:
        """Check if the target supports MPI by checking if the target is linked to MPI.

        Cached per ``target_path`` (``method=True`` keeps ``self`` out of the
        key); a binary's MPI linkage is constant within a CTest session.
        """
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

    @persistent_cached_property
    def amdsmi_version(self) -> Optional[tuple[int, int]]:
        """Get (major, minor) version of amd-smi, or None if not available."""
        return get_amdsmi_version(self.rocm_path)

    @persistent_cached_property
    def amdgpu_version(self) -> Optional[tuple[int, int, int]]:
        """Get (major, minor, patch) of the amdgpu driver, or None if not available."""
        return get_amdgpu_version(self.rocm_path)


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
