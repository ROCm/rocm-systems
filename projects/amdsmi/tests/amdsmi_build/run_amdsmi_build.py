#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
run_amdsmi_build.py - Local runner for the AMDSMI build and install workflow.

This script mirrors the build and install steps from
.github/workflows/amdsmi-build.yml so that you can reproduce builds and
installs directly on a local machine
(or inside a Docker container) without GitHub Actions.  Every command is logged to
--log-dir, failures surface immediately with the path to the log file, and test
results are written to --test-results-dir with well-known filenames for CI upload.

Prerequisites
=============
* Python 3.8+
* CMake >= 3.16
* A C/C++ toolchain (gcc/g++ or clang)
* Root / sudo privileges (package install touches /opt/rocm)
* For Debian/Ubuntu: apt
* For RHEL/AlmaLinux/Fedora: dnf
* For SLES: zypper

Quick Start (Local)
===================
Clone the repo, cd into it, and run the script.  It will auto-detect the project
root, package manager, and CPU count.

    sudo python3 projects/amdsmi/tests/amdsmi_build/run_amdsmi_build.py

Per-OS Examples
===============

Ubuntu 20.04 / 22.04 / 24.04 (Debian-based, apt)
--------------------------------------------------
    sudo python3 run_amdsmi_build.py \
        --package-manager apt \
        --os-label Ubuntu22

Debian 10
------------------------------------
    sudo python3 run_amdsmi_build.py \
        --package-manager apt \
        --debian10-sources \
        --os-label Debian10

RHEL 8 / 9 (dnf)
-----------------
    sudo python3 run_amdsmi_build.py \
        --package-manager dnf \
        --package-format rpm \
        --os-label RHEL9

RHEL 10 / AlmaLinux 8 (dnf - needs QA_RPATHS)
----------------------------------------------
    sudo python3 run_amdsmi_build.py \
        --package-manager dnf \
        --package-format rpm \
        --qa-rpaths \
        --os-label RHEL10

SLES (zypper)
-------------
    sudo python3 run_amdsmi_build.py \
        --package-manager zypper \
        --package-format rpm \
        --os-label SLES

AzureLinux 3 (dnf - skip setuptools upgrade, install more_itertools)
--------------------------------------------------------------------
    sudo python3 run_amdsmi_build.py \
        --package-manager dnf \
        --package-format rpm \
        --skip-setuptools-upgrade \
        --install-more-itertools \
        --os-label AzureLinux3

Skipping Stages
===============
Use --skip-build or --skip-install to skip
individual stages.  For example, to re-run only the tests against an existing
build directory:

    sudo python3 run_amdsmi_build.py --skip-build

Output
======
* Logs:         --log-dir   (default: logs/amdsmi/)
* Test results: --test-results-dir (default: /tmp/test-results-<os-label>/)

Run with --help for the full list of options.
"""


# ---------------------------------------------------------------------------
# Bootstrap: ensure Python 3.7+ (required for dataclasses & type annotations).
# On SLES the default python3 may be 3.6; this block auto-installs a newer
# interpreter via zypper and re-execs.  The code below intentionally avoids
# any 3.7+ syntax so it can run on Python 3.6.
# ---------------------------------------------------------------------------
def _bootstrap_python():
    import os
    import shutil
    import subprocess
    import sys

    if sys.version_info >= (3, 7):
        return

    pkg_mgr = None
    install_cmd = None
    pkg_candidates = None
    if shutil.which("zypper"):
        pkg_mgr = "zypper"
        install_cmd = ["zypper", "--non-interactive", "install"]
        pkg_candidates = [
            ["python311", "python311-pip"],
            ["python310", "python310-pip"],
            ["python39", "python39-pip"],
            ["python38", "python38-pip"],
        ]
    elif shutil.which("dnf"):
        pkg_mgr = "dnf"
        install_cmd = ["dnf", "install", "-y"]
        pkg_candidates = [
            ["python3.11", "python3.11-pip"],
            ["python3.9", "python3.9-pip"],
            ["python39", "python39-pip"],
            ["python3", "python3-pip"],
        ]
    elif shutil.which("apt"):
        pkg_mgr = "apt"
        install_cmd = ["apt-get", "install", "-y", "--no-install-recommends"]
        pkg_candidates = [
            ["python3.11", "python3-pip"],
            ["python3.10", "python3-pip"],
            ["python3.9", "python3-pip"],
            ["python3", "python3-pip"],
        ]
    else:
        sys.exit(
            "ERROR: Python 3.7+ is required (have %d.%d). "
            "Install a newer interpreter or use a newer base image."
            % (sys.version_info[0], sys.version_info[1])
        )
    print(
        "Python %d.%d is too old - upgrading via %s..."
        % (sys.version_info[0], sys.version_info[1], pkg_mgr)
    )
    if pkg_mgr == "apt":
        try:
            # Keep stdout quiet but let stderr surface install diagnostics.
            subprocess.check_call(["apt-get", "update"], stdout=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            pass
    for pkgs in pkg_candidates:
        try:
            # Surface stderr so an operator can see *why* every candidate
            # failed if we fall through the loop with no interpreter installed.
            subprocess.check_call(install_cmd + pkgs, stdout=subprocess.DEVNULL)
            break
        except subprocess.CalledProcessError:
            continue
    new_py = None
    for name in ["python3.11", "python3.10", "python3.9", "python3.8"]:
        p = shutil.which(name)
        if p:
            new_py = p
            break
    if new_py is None:
        sys.exit("ERROR: could not find a Python 3.7+ interpreter after install")
    try:
        subprocess.check_call(
            ["alternatives", "--set", "python3", new_py], stdout=subprocess.DEVNULL
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        link = "/usr/bin/python3"
        if os.path.exists(link) or os.path.islink(link):
            os.unlink(link)
        os.symlink(new_py, link)
    print("Re-executing under %s ..." % new_py)
    os.execvp(new_py, [new_py] + sys.argv)


_bootstrap_python()
# ---------------------------------------------------------------------------

import argparse
import datetime as _dt
import os
import shutil
import subprocess
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

DEFAULT_LOG_DIR = Path("logs") / "amdsmi"


class CommandError(RuntimeError):
    def __init__(self, name: str, cmd: List[str], code: int, log_path: Path):
        super().__init__(f"{name} failed with exit code {code}; see {log_path}")
        self.name = name
        self.cmd = cmd
        self.code = code
        self.log_path = log_path


def _timestamp() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d-%H%M%S")


def run_command(
    cmd: Iterable[str],
    *,
    name: str,
    cwd: Optional[Path] = None,
    env: Optional[dict] = None,
    retries: int = 1,
    log_dir: Path = DEFAULT_LOG_DIR,
    result_file: Optional[Path] = None,
) -> Path:
    """Run a command, streaming output to stdout and a log file.

    If *result_file* is given the full output is also written there (using a
    well-known filename that the CI workflow can display later).
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{_timestamp()}-{name}.log"
    cmd_list = [str(part) for part in cmd]
    attempt = 0

    while attempt < retries:
        attempt += 1
        prefix = f"[{name}] attempt {attempt}/{retries}"
        print(f"{prefix}: {' '.join(cmd_list)}")

        outputs = [log_path]
        if result_file is not None:
            result_file.parent.mkdir(parents=True, exist_ok=True)
            outputs.append(result_file)

        handles = [p.open("a", encoding="utf-8") for p in outputs]
        try:
            for fh in handles:
                fh.write(f"{prefix}\n")
            proc = subprocess.Popen(
                cmd_list,
                cwd=str(cwd) if cwd else None,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            # proc.stdout is guaranteed non-None because stdout=PIPE above.
            for line in proc.stdout:
                sys.stdout.write(line)
                for fh in handles:
                    fh.write(line)
            code = proc.wait()
        finally:
            for fh in handles:
                fh.close()

        if code == 0:
            return log_path

        if attempt >= retries:
            raise CommandError(name, cmd_list, code, log_path)

        print(f"{prefix} failed with exit code {code}; retrying...")

    return log_path


def tail_log(log_path: Path, max_lines: int = 40) -> str:
    """Return the last max_lines from a log file for quick context."""
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as fh:
            tail = deque(fh, maxlen=max_lines)
        return "".join(tail)
    except OSError as exc:
        return f"(could not read log {log_path}: {exc})"


def read_log(log_path: Path) -> str:
    """Return the full contents of a log file."""
    try:
        return log_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"(could not read log {log_path}: {exc})"


def report_and_raise(stage: str, err: CommandError) -> None:
    print(f"\nAMDSMI failed to {stage}: step '{err.name}' exited with {err.code}")
    print(f"Log: {err.log_path}")
    print("Log tail (last ~40 lines):")
    print("----------------------------------------")
    print(tail_log(err.log_path))
    print("----------------------------------------\n")
    raise err


def find_project_dir(explicit: Optional[Path]) -> Path:
    if explicit:
        return explicit.resolve()

    def _repo_root(start: Path) -> Path:
        for ancestor in [start] + list(start.parents):
            if (ancestor / ".git").exists():
                return ancestor
        return start.parents[1]

    repo_root = _repo_root(Path(__file__).resolve())
    primary = repo_root / "projects" / "amdsmi" / "CMakeLists.txt"
    if primary.exists():
        return primary.parent

    for candidate in repo_root.rglob("CMakeLists.txt"):
        return candidate.parent

    raise FileNotFoundError(f"Could not find CMakeLists.txt under {repo_root}")


def detect_package_manager(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    for candidate in ("apt", "dnf", "zypper"):
        if shutil.which(candidate):
            return candidate
    raise RuntimeError("Could not detect a package manager (apt, dnf, zypper)")


def detect_package_format(pkg_manager: str, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    if pkg_manager == "apt":
        return "deb"
    if pkg_manager in ("dnf", "zypper"):
        return "rpm"
    raise ValueError(f"Unsupported package manager {pkg_manager}")


def detect_os_profile(os_release: Optional[Path] = None) -> dict:
    """Parse /etc/os-release and return a dict of derived runner settings.

    Returned keys:
      os_label, package_manager, package_format,
      debian10_sources, qa_rpaths,
      skip_setuptools_upgrade, install_more_itertools
    Returns an empty dict if /etc/os-release is missing/unrecognized.

    *os_release* may be overridden in tests.
    """
    osr = os_release if os_release is not None else Path("/etc/os-release")
    if not osr.exists():
        return {}
    fields = {}
    for line in osr.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        fields[k.strip()] = v.strip().strip('"').strip("'")

    os_id = fields.get("ID", "").lower()
    id_like = fields.get("ID_LIKE", "").lower()
    version_id = fields.get("VERSION_ID", "")
    major = version_id.split(".", 1)[0] if version_id else ""

    profile = {
        "debian10_sources": False,
        "qa_rpaths": False,
        "skip_setuptools_upgrade": False,
        "install_more_itertools": False,
    }

    # Debian/Ubuntu family --------------------------------------------------
    if os_id == "ubuntu":
        profile["package_manager"] = "apt"
        profile["package_format"] = "deb"
        profile["os_label"] = f"Ubuntu{major}" if major else "Ubuntu"
    elif os_id == "debian":
        profile["package_manager"] = "apt"
        profile["package_format"] = "deb"
        profile["os_label"] = f"Debian{major}" if major else "Debian"
        if major == "10":
            profile["debian10_sources"] = True
    # RHEL family -----------------------------------------------------------
    elif os_id in ("rhel", "redhat", "redhatenterpriseserver"):
        profile["package_manager"] = "dnf"
        profile["package_format"] = "rpm"
        profile["os_label"] = f"RHEL{major}" if major else "RHEL"
        if major in ("10",):
            profile["qa_rpaths"] = True
    elif os_id == "almalinux":
        profile["package_manager"] = "dnf"
        profile["package_format"] = "rpm"
        profile["os_label"] = f"AlmaLinux{major}" if major else "AlmaLinux"
        if major == "8":
            profile["qa_rpaths"] = True
    elif os_id in ("rocky", "centos"):
        profile["package_manager"] = "dnf"
        profile["package_format"] = "rpm"
        label_prefix = os_id.capitalize()
        profile["os_label"] = f"{label_prefix}{major}" if major else label_prefix
    elif os_id in ("mariner", "azurelinux"):
        profile["package_manager"] = "dnf"
        profile["package_format"] = "rpm"
        prefix = "AzureLinux" if os_id == "azurelinux" else "Mariner"
        profile["os_label"] = f"{prefix}{major}" if major else prefix
        profile["skip_setuptools_upgrade"] = True
        profile["install_more_itertools"] = True
    # SUSE family -----------------------------------------------------------
    elif os_id in ("sles", "sled", "opensuse-leap", "opensuse-tumbleweed") or "suse" in id_like:
        profile["package_manager"] = "zypper"
        profile["package_format"] = "rpm"
        profile["os_label"] = "SLES" if os_id.startswith("sles") else os_id

    return profile


def package_glob(package_format: str) -> str:
    if package_format == "deb":
        return "amd-smi-lib*99999-local_amd64.deb"
    return "amd-smi-lib-*99999-local*.rpm"


# ---------------------------------------------------------------------------
# Pre-build helpers
# ---------------------------------------------------------------------------


def install_build_prereqs(cfg: "RunnerConfig") -> None:
    """Ensure cmake + a basic build toolchain are installed.

    Several ROCm base images (notably rhel-9.x-bld, rhel-10.x-bld) do not
    ship cmake on PATH. This step installs cmake/gcc/g++/make on demand so
    the cmake-configure step doesn't die with FileNotFoundError.
    """
    if (
        shutil.which("cmake")
        and shutil.which("make")
        and (shutil.which("gcc") or shutil.which("cc"))
    ):
        return

    print("Installing build prerequisites (cmake/gcc/make)...")
    if cfg.package_manager == "apt":
        run_command(
            ["apt-get", "update"],
            name="apt-update-prereqs",
            retries=cfg.retries,
            log_dir=cfg.log_dir,
        )
        run_command(
            [
                "apt-get",
                "install",
                "-y",
                "--no-install-recommends",
                "cmake",
                "build-essential",
                "git",
            ],
            name="apt-install-prereqs",
            retries=cfg.retries,
            log_dir=cfg.log_dir,
        )
    elif cfg.package_manager == "dnf":
        # gcc-c++ + make are provided via "Development Tools" group; install the
        # individual packages explicitly so this works on minimal images too.
        run_command(
            [
                "dnf",
                "install",
                "-y",
                "--setopt=skip_if_unavailable=True",
                "cmake",
                "gcc",
                "gcc-c++",
                "make",
                "git",
            ],
            name="dnf-install-prereqs",
            retries=cfg.retries,
            log_dir=cfg.log_dir,
        )
    elif cfg.package_manager == "zypper":
        run_command(
            ["zypper", "--non-interactive", "install", "cmake", "gcc", "gcc-c++", "make", "git"],
            name="zypper-install-prereqs",
            retries=cfg.retries,
            log_dir=cfg.log_dir,
        )


def install_netlink_deps(cfg: "RunnerConfig") -> None:
    """Install libnl-3 / libmnl development headers used by amdsmi's CMake.

    CMakeLists.txt requires libnl-3.0 and libmnl via pkg_check_modules. Most
    base container images do not ship these, so install them here. No-op if
    pkg-config can already locate both modules.
    """
    pkg_config = shutil.which("pkg-config")
    if pkg_config:
        try:
            subprocess.check_call(
                [pkg_config, "--exists", "libnl-3.0", "libmnl"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return
        except subprocess.CalledProcessError:
            pass

    print("Installing netlink development headers (libnl-3, libmnl)...")
    if cfg.package_manager == "apt":
        if cfg.refresh_apt:
            run_command(
                ["apt-get", "update"],
                name="apt-update-netlink",
                retries=cfg.retries,
                log_dir=cfg.log_dir,
            )
        run_command(
            [
                "apt-get",
                "install",
                "-y",
                "--no-install-recommends",
                "libnl-3-dev",
                "libnl-genl-3-dev",
                "libmnl-dev",
            ],
            name="apt-install-netlink",
            retries=cfg.retries,
            log_dir=cfg.log_dir,
        )
    elif cfg.package_manager == "dnf":
        # AzureLinux 3 ships tdnf; prefer it when present, fall back to dnf.
        is_tdnf = bool(shutil.which("tdnf"))
        installer = "tdnf" if is_tdnf else "dnf"
        cmd = [installer, "install", "-y", "--setopt=skip_if_unavailable=True"]
        # AzureLinux 3 CI containers ship a tdnfrepogpgcheck plugin that
        # rejects all repo metadata when the Azure Linux GPG keys are
        # missing/expired, which disables every repo before install can run.
        # --noplugins skips that check; --nogpgcheck covers package-level
        # signing if that also trips.
        if is_tdnf:
            cmd += ["--noplugins", "--nogpgcheck"]
        cmd += ["libnl3-devel", "libmnl-devel"]
        run_command(
            cmd, name=f"{installer}-install-netlink", retries=cfg.retries, log_dir=cfg.log_dir
        )
    elif cfg.package_manager == "zypper":
        run_command(
            ["zypper", "--non-interactive", "install", "libnl3-devel", "libmnl-devel"],
            name="zypper-install-netlink",
            retries=cfg.retries,
            log_dir=cfg.log_dir,
        )


def clean_stale_artifacts(log_dir: Path) -> None:
    """Remove SWIG-based .so baked into Docker images.

    The package now uses ctypes; the stale extension references symbols removed
    from libamd_smi.so and causes 'undefined symbol' errors on import.
    """
    stale = Path("/opt/rocm/share/amd_smi/amdsmi/libamd_smi_python.so")
    if stale.exists():
        stale.unlink()
        print(f"Removed stale artifact: {stale}")
    # Refresh linker cache
    if shutil.which("ldconfig"):
        try:
            run_command(["ldconfig"], name="ldconfig", log_dir=log_dir)
        except CommandError:
            print("Warning: ldconfig failed (non-fatal)")


def install_more_itertools(log_dir: Path, retries: int) -> None:
    """AzureLinux3 needs more_itertools installed separately."""
    run_command(
        ["python3", "-m", "pip", "install", "more_itertools"],
        name="pip-more-itertools",
        retries=retries,
        log_dir=log_dir,
    )


def upgrade_setuptools(log_dir: Path, retries: int) -> None:
    run_command(
        ["python3", "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        name="pip-upgrade",
        retries=retries,
        log_dir=log_dir,
    )


def repair_cmake(log_dir: Path) -> None:
    """Re-install the cmake pip package if the wrapper script is broken.

    On SLES the system cmake is a pip-installed Python wrapper.  Upgrading
    pip/setuptools can break it (``ModuleNotFoundError: No module named
    'cmake'``).  This function detects the breakage and force-reinstalls
    the cmake package so the wrapper works again.
    """
    try:
        subprocess.run(
            ["cmake", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("cmake is broken or missing — reinstalling via pip...")
        run_command(
            ["python3", "-m", "pip", "install", "--force-reinstall", "cmake"],
            name="pip-repair-cmake",
            retries=1,
            log_dir=log_dir,
        )


def update_debian10_sources(log_dir: Path, retries: int) -> None:
    # buster-backports provides newer linux-libc-dev (kernel >= 5.10) which
    # supplies <linux/time_types.h> that ualoe_lib requires. Without it the
    # build dies on Debian 10's 4.19-based headers.
    content = (
        "deb http://archive.debian.org/debian buster main\n"
        "deb http://archive.debian.org/debian-security buster/updates main\n"
        "deb http://archive.debian.org/debian buster-backports main\n"
    )
    sources_list = Path("/etc/apt/sources.list")
    print("Updating sources.list for Debian10 (archived repos + backports)")
    sources_list.write_text(content, encoding="utf-8")
    Path("/etc/apt/apt.conf.d/99-disable-check-valid-until").write_text(
        'Acquire::Check-Valid-Until "false";\n', encoding="utf-8"
    )
    run_command(["apt", "update"], name="apt-update", retries=retries, log_dir=log_dir)


def install_debian10_kernel_headers(log_dir: Path, retries: int) -> None:
    """Pull linux-libc-dev from buster-backports for newer UAPI headers.

    Required so that ualoe_lib (which unconditionally includes
    <linux/time_types.h>, added to UAPI in kernel 5.1) builds on Debian 10.
    """
    print("Installing linux-libc-dev from buster-backports for newer UAPI headers")
    run_command(
        [
            "apt-get",
            "install",
            "-y",
            "--no-install-recommends",
            "-t",
            "buster-backports",
            "linux-libc-dev",
        ],
        name="apt-install-linux-libc-dev-backports",
        retries=retries,
        log_dir=log_dir,
    )


def _mark_safe_git_dir(path: Path) -> None:
    """Register *path* as a safe git directory so cmake's `git rev-parse`
    succeeds inside containers with bind-mounted repos. Without this, the
    package version string ends up as e.g. ``26.4.0+unknown``.
    """
    git_bin = shutil.which("git")
    if not git_bin or not path.exists():
        return
    subprocess.run(
        [git_bin, "config", "--global", "--add", "safe.directory", str(path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def build_amdsmi(cfg: "RunnerConfig") -> None:
    if cfg.build_dir.exists():
        shutil.rmtree(cfg.build_dir)
    cfg.build_dir.mkdir(parents=True, exist_ok=True)

    # Avoid cmake producing "+unknown" version hashes when the repo is
    # bind-mounted into a container with a different uid.
    _mark_safe_git_dir(cfg.project_dir)
    esmi_repo = cfg.project_dir / "esmi_ib_library"
    if esmi_repo.exists():
        _mark_safe_git_dir(esmi_repo)

    env = os.environ.copy()
    env.setdefault("DEBIAN_FRONTEND", "noninteractive")
    env.setdefault("DEBCONF_NONINTERACTIVE_SEEN", "true")

    # RHEL10 / AlmaLinux8 need QA_RPATHS to ignore empty RPATHs
    if cfg.qa_rpaths:
        env["QA_RPATHS"] = str(0x0010 | 0x0002)
        print(f"QA_RPATHS set to {env['QA_RPATHS']}")

    cmake_args = ["cmake", str(cfg.project_dir), "-DBUILD_TESTS=ON", "-DENABLE_ESMI_LIB=ON"]
    run_command(
        cmake_args,
        name="cmake-configure",
        cwd=cfg.build_dir,
        retries=cfg.retries,
        env=env,
        log_dir=cfg.log_dir,
    )

    run_command(
        ["make", "-j", str(cfg.jobs)],
        name="make",
        cwd=cfg.build_dir,
        retries=cfg.retries,
        env=env,
        log_dir=cfg.log_dir,
    )

    wheel_dir = cfg.build_dir / "py-interface" / "python_package"
    # Clean stale wheel artifacts before building
    for pattern in ("*.whl", "*.egg-info"):
        for stale_path in wheel_dir.glob(pattern):
            if stale_path.is_dir():
                shutil.rmtree(stale_path)
            else:
                stale_path.unlink()
    for subdir_name in ("dist", "build"):
        subdir = wheel_dir / subdir_name
        if subdir.exists():
            shutil.rmtree(subdir)

    wheel_cmd = [
        "python3",
        "-m",
        "pip",
        "wheel",
        "--no-deps",
        "--no-build-isolation",
        "-w",
        ".",
        ".",
    ]
    run_command(
        wheel_cmd,
        name="python-wheel",
        cwd=wheel_dir,
        retries=cfg.retries,
        env=env,
        log_dir=cfg.log_dir,
    )

    run_command(
        ["make", "package"],
        name="make-package",
        cwd=cfg.build_dir,
        retries=cfg.retries,
        env=env,
        log_dir=cfg.log_dir,
    )


def locate_package(build_dir: Path, package_format: str) -> Path:
    matches = sorted(build_dir.glob(package_glob(package_format)))
    if not matches:
        raise FileNotFoundError(f"No {package_format} artifact found in {build_dir}")

    def _is_tests_package(p: Path) -> bool:
        return "tests" in p.name

    # Prefer the main package over the tests package if both exist.
    main_pkgs = [p for p in matches if not _is_tests_package(p)]
    if main_pkgs:
        return max(main_pkgs, key=lambda p: p.stat().st_mtime)
    return max(matches, key=lambda p: p.stat().st_mtime)


def install_package(cfg: "RunnerConfig", package_path: Path) -> None:
    print(f"Installing package {package_path}")
    if cfg.package_manager == "apt":
        if cfg.refresh_apt:
            run_command(
                ["apt", "update"], name="apt-update", retries=cfg.retries, log_dir=cfg.log_dir
            )
        # Install main package; if a tests package exists, install it together to satisfy deps.
        extra_pkg = package_path.parent / package_path.name.replace("lib_", "lib-tests_", 1)
        pkg_list = [str(package_path)]
        if extra_pkg.exists():
            pkg_list.append(str(extra_pkg))
        run_command(
            ["apt", "install", "--reinstall", "-y", *pkg_list],
            name="apt-install",
            retries=cfg.retries,
            log_dir=cfg.log_dir,
        )
    elif cfg.package_manager == "dnf":
        run_command(
            [
                "dnf",
                "install",
                "python3-setuptools",
                "python3-wheel",
                "-y",
                "--setopt=skip_if_unavailable=True",
            ],
            name="dnf-prep",
            retries=cfg.retries,
            log_dir=cfg.log_dir,
        )
        # Install main package; if a tests package exists, install it too.
        tests_pkg = package_path.parent / package_path.name.replace(
            "amd-smi-lib-", "amd-smi-lib-tests-", 1
        )
        rpm_list = [str(package_path)]
        if tests_pkg.exists() and tests_pkg != package_path:
            rpm_list.append(str(tests_pkg))
        run_command(
            ["dnf", "install", "-y", "--skip-broken", "--disablerepo=*", *rpm_list],
            name="dnf-install",
            retries=cfg.retries,
            log_dir=cfg.log_dir,
        )
    elif cfg.package_manager == "zypper":
        # Install main package; if a tests package exists, install it too.
        tests_pkg = package_path.parent / package_path.name.replace(
            "amd-smi-lib-", "amd-smi-lib-tests-", 1
        )
        rpm_list = [str(package_path)]
        if tests_pkg.exists() and tests_pkg != package_path:
            rpm_list.append(str(tests_pkg))
        # --no-gpg-checks is safe here because rpm_list contains local file
        # paths we just built ourselves, not repo-resolved package names.
        run_command(
            ["zypper", "--no-refresh", "--no-gpg-checks", "install", "-y", *rpm_list],
            name="zypper-install",
            retries=cfg.retries,
            log_dir=cfg.log_dir,
        )
    else:
        raise ValueError(f"Unsupported package manager {cfg.package_manager}")

    rocm_binary = Path("/opt/rocm/bin/amd-smi")
    symlink_path = Path("/usr/local/bin/amd-smi")
    if rocm_binary.exists():
        symlink_path.parent.mkdir(parents=True, exist_ok=True)
        if symlink_path.exists() or symlink_path.is_symlink():
            symlink_path.unlink()
        symlink_path.symlink_to(rocm_binary)
        print(f"Linked {symlink_path} -> {rocm_binary}")

    # Verify installation: CLI version + Python import/init/shutdown under the
    # interpreter the module was actually installed for. The system package
    # installs amdsmi/ into a specific interpreter's site-packages, so verify
    # against THAT interpreter, not a bare /usr/bin/python3: this harness itself
    # repoints /usr/bin/python3 via `alternatives` when it has to bootstrap a
    # newer python on an old base image (e.g. el8's 3.6), so /usr/bin/python3 at
    # verify time may be a different minor than the one the module landed under.
    # Derive the interpreter from the installed CLI's shebang so the check
    # mirrors what a real user's `amd-smi` invocation resolves to. Fall back to
    # /usr/bin/python3 (then plain python3) if the shebang can't be read.
    verify_python = "/usr/bin/python3" if Path("/usr/bin/python3").exists() else "python3"
    try:
        cli_target = rocm_binary.resolve() if rocm_binary.exists() else None
        if cli_target and cli_target.exists():
            first_line = cli_target.read_text(errors="replace").splitlines()[:1]
            if first_line and first_line[0].startswith("#!"):
                tokens = first_line[0][2:].strip().split()
                interp = tokens[0] if tokens else ""
                # "#!/usr/bin/env python3" names the interpreter in the second
                # token; the first is env itself.
                if os.path.basename(interp) == "env" and len(tokens) > 1:
                    interp = tokens[1]
                resolved = interp if Path(interp).exists() else shutil.which(interp)
                if resolved:
                    verify_python = resolved
    except (OSError, IndexError):
        pass
    print(f"Verifying import under interpreter: {verify_python}")
    system_python = verify_python
    import_smoke = (
        "import amdsmi; "
        "print('amdsmi from:', amdsmi.__file__); "
        "amdsmi.amdsmi_init(); "
        "amdsmi.amdsmi_shut_down(); "
        "print('init/shutdown ok')"
    )
    verify_commands = [[str(rocm_binary), "version"], [system_python, "-c", import_smoke]]
    for idx, verify_cmd in enumerate(verify_commands, start=1):
        try:
            run_command(verify_cmd, name=f"verify-{idx}", retries=1, log_dir=cfg.log_dir)
        except CommandError as exc:
            print(f"Verification command failed: {exc}")
            raise


# ---------------------------------------------------------------------------
# Wheel verification (matches the "Verify Wheel in Site-Packages" CI step)
# ---------------------------------------------------------------------------


def verify_wheel_site_packages(cfg: "RunnerConfig") -> None:
    """Install the built wheel via pip and verify it lands in site-packages."""
    wheel_dir = cfg.build_dir / "py-interface" / "python_package"
    wheels = sorted(wheel_dir.glob("*.whl"))
    if not wheels:
        print(f"WARNING: No .whl found in {wheel_dir}; skipping wheel verification")
        return

    wheel = wheels[0]
    print(f"Found wheel: {wheel}")

    run_command(
        ["python3", "-m", "pip", "install", "--force-reinstall", str(wheel)],
        name="pip-install-wheel",
        retries=1,
        log_dir=cfg.log_dir,
    )

    run_command(
        ["python3", "-m", "pip", "show", "amdsmi"],
        name="pip-show-amdsmi",
        retries=1,
        log_dir=cfg.log_dir,
    )

    # Prove the pip install takes priority over any coexisting system copy for
    # plain Python scripting: a user who `pip install`s amdsmi wants that
    # version. `pip show` reports where pip put the wheel; assert `import
    # amdsmi` resolves there. Clear PYTHONPATH first: the CI container and a
    # ROCm install both point PYTHONPATH at /opt/rocm/share/amd_smi, which
    # unconditionally precedes site-packages, so leaving it set would test the
    # environment's path config rather than the package. A bare interpreter
    # (no PYTHONPATH) is the real scripting scenario where pip must win.
    # This check is hardware-independent and runs before the GPU smoke test
    # below, so it executes even on GPU-less runners (manylinux, containers)
    # where amdsmi_init() would otherwise fail first and mask a wrong-path
    # install.
    priority_check = (
        "import os, subprocess, amdsmi\n"
        "out = subprocess.check_output(['python3', '-m', 'pip', 'show', 'amdsmi'], text=True)\n"
        "loc = next((l.split(':', 1)[1].strip() for l in out.splitlines() "
        "if l.startswith('Location:')), '')\n"
        "loc = os.path.realpath(loc)\n"
        "p = os.path.realpath(amdsmi.__file__)\n"
        "print('pip install location: ' + loc)\n"
        "print('amdsmi imported from : ' + p)\n"
        "assert loc and p.startswith(loc), "
        "'a system copy shadowed the pip install: import resolved to ' + p\n"
        "assert '/opt/rocm/' not in p, "
        "'the /opt/rocm system copy shadowed the pip install: ' + p\n"
        "print('PASS: pip install takes priority for scripting')\n"
    )
    priority_env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    run_command(
        ["python3", "-c", priority_check],
        name="wheel-priority-check",
        cwd=Path("/tmp"),
        env=priority_env,
        retries=1,
        log_dir=cfg.log_dir,
    )

    # GPU-dependent smoke test: initialize the library and enumerate devices.
    # Skip cleanly when no GPU/driver is present so packaging-only runners are
    # not failed by the absence of hardware.
    smoke_test = (
        "import amdsmi\n"
        "try:\n"
        "    amdsmi.amdsmi_init()\n"
        "except amdsmi.AmdSmiException as e:\n"
        "    print('SKIP: amdsmi_init() failed (no GPU/driver?): %s' % e)\n"
        "    raise SystemExit(0)\n"
        "print('PASS: amdsmi_init() OK')\n"
        "devs = amdsmi.amdsmi_get_processor_handles()\n"
        "print('PASS: Found %d device(s)' % len(devs))\n"
        "amdsmi.amdsmi_shut_down()\n"
        "print('PASS: amdsmi_shut_down() OK')\n"
        "print('=== Wheel GPU smoke test passed ===')\n"
    )
    run_command(
        ["python3", "-c", smoke_test],
        name="wheel-smoke-test",
        cwd=Path("/tmp"),
        retries=1,
        log_dir=cfg.log_dir,
    )


def verify_soname_distinct(cfg: "RunnerConfig") -> None:
    """Assert the system and wheel libraries keep distinct SONAMEs.

    Runs the standalone SONAME conflict check against the build tree. Only
    meaningful when the wheel library was built (BUILD_PYTHON_WHEEL=ON), so a
    build tree without libamd_smi_python.so is skipped rather than failed.
    """
    wheel_libs = list(cfg.build_dir.glob("**/libamd_smi_python.so.*"))
    if not wheel_libs:
        print("Skipping SONAME conflict check: no wheel library in the build tree")
        return

    test_script = cfg.project_dir / "tests" / "run_amdsmi_pkg_conflict_test.py"
    run_command(
        ["python3", str(test_script), "--build-root", str(cfg.build_dir)],
        name="pkg-conflict-soname",
        retries=1,
        log_dir=cfg.log_dir,
    )


def verify_dual_copy(cfg: "RunnerConfig") -> None:
    """Assert the system package's two module copies are byte-identical.

    The DEB/RPM installs amdsmi into both site-packages and share/amd_smi. Only
    runs when the share/amd_smi copy exists (system package installed), skipping
    on a wheel-only install where there is a single copy.
    """
    rocm_path = os.environ.get("ROCM_PATH") or os.environ.get("ROCM_HOME") or "/opt/rocm"
    share_copy = Path(rocm_path) / "share" / "amd_smi" / "amdsmi"
    if not share_copy.is_dir():
        print(f"Skipping dual-copy check: {share_copy} not present")
        return

    test_script = cfg.project_dir / "tests" / "run_amdsmi_dual_copy_test.py"
    run_command(
        ["python3", str(test_script)], name="dual-copy-guard", retries=1, log_dir=cfg.log_dir
    )


def verify_cpack_paths(cfg: "RunnerConfig", artifact: Path) -> None:
    """Assert the built package ships the amdsmi module on an import path."""
    test_script = cfg.project_dir / "tests" / "run_amdsmi_cpack_path_test.py"
    run_command(
        ["python3", str(test_script), str(artifact)],
        name="cpack-path-check",
        retries=1,
        log_dir=cfg.log_dir,
    )


def verify_python_versions(cfg: "RunnerConfig") -> None:
    """Run the loader contract tests under multiple Python interpreters."""
    test_script = cfg.project_dir / "tests" / "run_amdsmi_python_versions_test.py"
    cmd = ["python3", str(test_script)]
    if cfg.python_interpreters:
        cmd += ["--interpreters"] + cfg.python_interpreters.split(",")
    run_command(cmd, name="python-versions", retries=1, log_dir=cfg.log_dir)


def verify_upgrade_downgrade(cfg: "RunnerConfig", new_artifact: Path) -> None:
    """Exercise the deb/rpm upgrade and downgrade path from a prior package."""
    if not cfg.upgrade_from:
        print("Skipping upgrade/downgrade check: no --upgrade-from package given")
        return
    test_script = cfg.project_dir / "tests" / "run_amdsmi_upgrade_downgrade_test.py"
    run_command(
        [
            "python3",
            str(test_script),
            "--old-package",
            str(cfg.upgrade_from),
            "--new-package",
            str(new_artifact),
            "--package-manager",
            cfg.package_manager,
        ],
        name="upgrade-downgrade",
        retries=1,
        log_dir=cfg.log_dir,
    )


@dataclass
class RunnerConfig:
    project_dir: Path
    build_dir: Path
    package_manager: str
    package_format: str
    log_dir: Path
    test_results_dir: Path
    retries: int
    build_type: str
    jobs: int
    os_label: str
    refresh_apt: bool
    debian10_sources: bool
    skip_setuptools_upgrade: bool
    install_more_itertools: bool
    qa_rpaths: bool
    skip_build: bool
    skip_install: bool
    upgrade_from: "Optional[Path]"
    python_interpreters: "Optional[str]"


def parse_args() -> RunnerConfig:
    parser = argparse.ArgumentParser(description="Run AMDSMI build/install locally.")
    parser.add_argument("--project-dir", type=Path, help="Path to the AMDSMI project root")
    parser.add_argument("--build-dir", type=Path, help="Build directory (default: <project>/build)")
    parser.add_argument(
        "--log-dir", type=Path, default=DEFAULT_LOG_DIR, help="Directory for command logs"
    )
    parser.add_argument(
        "--test-results-dir",
        type=Path,
        help="Directory for well-known test result files (default: /tmp/test-results-<os-label>)",
    )
    parser.add_argument(
        "--package-manager", choices=["apt", "dnf", "zypper"], help="Package manager to use"
    )
    parser.add_argument("--package-format", choices=["deb", "rpm"], help="Force package format")
    parser.add_argument("--retries", type=int, default=3, help="Retries for build/install steps")
    parser.add_argument("--build-type", default="Release", help="CMake build type")
    parser.add_argument(
        "--jobs", type=int, default=os.cpu_count() or 4, help="Parallel jobs for make"
    )
    parser.add_argument("--os-label", default="local", help="Label used in log/result dir names")
    parser.add_argument(
        "--no-apt-update", action="store_true", help="Do not run apt update before install"
    )
    parser.add_argument(
        "--debian10-sources", action="store_true", help="Rewrite apt sources for Debian10 archive"
    )
    parser.add_argument(
        "--skip-setuptools-upgrade",
        action="store_true",
        help="Skip pip/setuptools/wheel upgrade (e.g. AzureLinux3)",
    )
    parser.add_argument(
        "--install-more-itertools",
        action="store_true",
        help="Install more_itertools (e.g. AzureLinux3)",
    )
    parser.add_argument(
        "--qa-rpaths", action="store_true", help="Set QA_RPATHS for RPM builds (RHEL10, AlmaLinux8)"
    )
    parser.add_argument(
        "--skip-build", action="store_true", help="Skip build step (use existing build dir)"
    )
    parser.add_argument("--skip-install", action="store_true", help="Skip package installation")
    parser.add_argument(
        "--upgrade-from",
        type=Path,
        default=None,
        help="Prior-version .deb/.rpm to test the upgrade/downgrade path against",
    )
    parser.add_argument(
        "--python-interpreters",
        default=None,
        help="Comma-separated interpreters for the Python-version loader tests "
        "(default: discovered python3.6..3.14)",
    )
    parser.add_argument(
        "--no-autodetect",
        action="store_true",
        help="Disable auto-detection of OS profile from /etc/os-release",
    )
    args = parser.parse_args()

    profile = {} if args.no_autodetect else detect_os_profile()
    if profile:
        print(f"Auto-detected OS profile: {profile}")

    project_dir = find_project_dir(args.project_dir)
    build_dir = args.build_dir or project_dir / "build"
    package_manager = detect_package_manager(args.package_manager or profile.get("package_manager"))
    package_format = detect_package_format(
        package_manager, args.package_format or profile.get("package_format")
    )
    # os-label CLI default is "local"; only treat that as user-supplied if
    # autodetect produced nothing, otherwise prefer the autodetected label.
    if args.os_label != "local":
        os_label = args.os_label
    else:
        os_label = profile.get("os_label", args.os_label)
    test_results_dir = args.test_results_dir or Path(f"/tmp/test-results-{os_label}")

    return RunnerConfig(
        project_dir=project_dir,
        build_dir=build_dir,
        package_manager=package_manager,
        package_format=package_format,
        log_dir=args.log_dir,
        test_results_dir=test_results_dir,
        retries=max(1, args.retries),
        build_type=args.build_type,
        jobs=max(1, args.jobs),
        os_label=os_label,
        refresh_apt=not args.no_apt_update,
        debian10_sources=args.debian10_sources or profile.get("debian10_sources", False),
        skip_setuptools_upgrade=args.skip_setuptools_upgrade
        or profile.get("skip_setuptools_upgrade", False),
        install_more_itertools=args.install_more_itertools
        or profile.get("install_more_itertools", False),
        qa_rpaths=args.qa_rpaths or profile.get("qa_rpaths", False),
        skip_build=args.skip_build,
        skip_install=args.skip_install,
        upgrade_from=args.upgrade_from,
        python_interpreters=args.python_interpreters,
    )


def _write_result(results_dir: Path, filename: str, message: str) -> None:
    """Write a status message to a well-known result file."""
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / filename).write_text(message + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CI summary rendering (replaces the duplicated bash block in amdsmi-build.yml)
# ---------------------------------------------------------------------------


def _fenced(text: str) -> str:
    """Wrap *text* in a markdown fenced code block."""
    return "```\n" + text + "\n```"


def summarize_results(results_dir: Path, os_label: str, summary_file: Optional[Path]) -> int:
    """Render a CI summary for *results_dir*.

    Returns the number of failures detected (0 == clean). When *summary_file*
    is given the rendered markdown is appended to it (used with
    ``$GITHUB_STEP_SUMMARY``).
    """
    failures: List[str] = []
    details: List[str] = []

    if not results_dir.exists():
        print(f"summarize: results dir {results_dir} does not exist")

    # 1. build/install/verify stage results
    for result_file in sorted(results_dir.glob("*_result.txt")):
        content = result_file.read_text(encoding="utf-8", errors="replace")
        if "FAILED" in content:
            stage = result_file.stem.replace("_result", "").replace("_", " ")
            failures.append(stage)
            details.append(f"#### {stage}\n\n" + _fenced(content))
            print(f"FAILED: {stage}")

    # 2. amd-smi command test logs -- logged but non-fatal. The workflow
    # appends "Error code: <rc>" (space + digit, same line) only on a real
    # non-zero exit. The CLI's debug logging also prints an unsupported-feature
    # status code, but wrapped onto the next line ("Error code:\n\t2 | ..."),
    # so restrict the match to a same-line digit to skip that benign noise.
    import re as _re

    err_code_re = _re.compile(r"Error code:[ \t]*\d")
    cmd_fails: List[str] = []
    for log in sorted(results_dir.glob("amd-smi_*.log")):
        log_text = log.read_text(encoding="utf-8", errors="replace")
        if "Traceback" in log_text or "AmdSmiException" in log_text or err_code_re.search(log_text):
            cmd_fails.append(log.stem.replace("amd-smi_", ""))
    if cmd_fails:
        joined = " ".join(cmd_fails)
        details.append(f"#### Command Tests (non-fatal)\n\nFailed: `{joined}`")

    # 3. AMDSMI gtest output
    gtest_log = results_dir / "amdsmi_tests.log"
    if gtest_log.exists():
        text = gtest_log.read_text(encoding="utf-8", errors="replace")
        gtest_fails = text.count("[  FAILED  ]")
        if gtest_fails > 0:
            failures.append(f"AMDSMI Tests ({gtest_fails})")
            details.append(f"#### AMDSMI Tests \u2014 {gtest_fails} failure(s)\n\n" + _fenced(text))

    # 4. Python test outputs
    fail_re = _re.compile(r"^(FAIL|ERROR):", _re.MULTILINE)
    for test_file in (
        "python_test_output.txt",
        "perf_test_output.txt",
        "abi_compat_output.txt",
    ):
        full = results_dir / test_file
        if not full.exists():
            continue
        text = full.read_text(encoding="utf-8", errors="replace")
        py_fails = len(fail_re.findall(text))
        if py_fails > 0:
            name = test_file.replace("_output.txt", "").replace("_", " ")
            failures.append(f"{name} ({py_fails})")
            details.append(f"#### {name} \u2014 {py_fails} failure(s)\n\n" + _fenced(text))

    # 5. example test results (segfault detection)
    crash_re = _re.compile(r"segfault|SIGSEGV|abort", _re.IGNORECASE)
    for ex_log in ("amd_smi_drm_ex.log", "amd_smi_nodrm_ex.log"):
        full = results_dir / ex_log
        if not full.exists():
            continue
        text = full.read_text(encoding="utf-8", errors="replace")
        if crash_re.search(text):
            name = ex_log.replace(".log", "")
            failures.append(f"Example {name}")
            details.append(f"#### Example {name}\n\n" + _fenced(text))

    # Render. Hard failures (build/install/verify/gtest/python/examples) drive
    # the exit status; command-test failures are surfaced only as a warning so
    # transient amd-smi CLI flakes do not fail the whole job.
    cmd_summary = ""
    if cmd_fails:
        cmd_summary = f"\n\n:warning: Command tests (non-fatal): `{' '.join(cmd_fails)}`\n"

    if failures:
        header = [
            f"## :x: CI Failed \u2014 {os_label}",
            "",
            f"**{len(failures)} failure(s) detected:**",
            "",
        ]
        header += [f"- :red_circle: {f}" for f in failures]
        body = "\n".join(header) + cmd_summary + "\n\n" + "\n\n".join(details) + "\n"
    else:
        body = (
            f"## :white_check_mark: CI Passed \u2014 {os_label}\n\n"
            "All stages and tests passed successfully.\n"
        )
        if cmd_fails:
            body += cmd_summary + "\n" + "\n\n".join(details) + "\n"

    if summary_file is not None:
        with summary_file.open("a", encoding="utf-8") as fh:
            fh.write(body)

    # Always print to stdout so the step log is self-contained.
    print("=" * 42)
    print(f"CI SUMMARY for {os_label}")
    print("=" * 42)
    print(body)

    if failures:
        joined = ", ".join(failures)
        print(f"::error::{len(failures)} failure(s) for {os_label}: {joined}")
    elif cmd_fails:
        print(f"::warning::Command tests failed (non-fatal) for {os_label}: {' '.join(cmd_fails)}")
        print(f"All hard stages PASSED for {os_label}")
    else:
        print(f"All stages and tests PASSED for {os_label}")
    return len(failures)


def _summarize_cli(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="run_amdsmi_build.py summarize",
        description="Render CI step summary from a test-results directory.",
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--os-label", default="local")
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=None,
        help="Append markdown to this file (e.g. $GITHUB_STEP_SUMMARY).",
    )
    args = parser.parse_args(argv)
    return summarize_results(args.results_dir, args.os_label, args.summary_file)


def main() -> None:
    # `summarize` subcommand short-circuits before parse_args/sudo logic.
    if len(sys.argv) > 1 and sys.argv[1] == "summarize":
        sys.exit(0 if _summarize_cli(sys.argv[2:]) == 0 else 1)
    cfg = parse_args()

    print(f"Using project directory: {cfg.project_dir}")
    print(f"Logs will be saved under: {cfg.log_dir}")
    print(f"Test results will be saved under: {cfg.test_results_dir}")
    print(f"Package manager: {cfg.package_manager} (format {cfg.package_format})")
    print(f"OS label: {cfg.os_label}")

    # 1. Debian10 archived repos
    if cfg.debian10_sources and cfg.package_manager != "apt":
        print("Warning: --debian10-sources ignored because package manager is not apt")
    if cfg.debian10_sources:
        update_debian10_sources(cfg.log_dir, cfg.retries)
        install_debian10_kernel_headers(cfg.log_dir, cfg.retries)

    # 2. Install more_itertools if requested (e.g. AzureLinux3)
    if cfg.install_more_itertools:
        install_more_itertools(cfg.log_dir, cfg.retries)

    # 3. Upgrade setuptools (skip on AzureLinux3)
    if not cfg.skip_setuptools_upgrade:
        upgrade_setuptools(cfg.log_dir, cfg.retries)

    # 3b. Repair cmake if the pip wrapper broke during setuptools upgrade (SLES)
    repair_cmake(cfg.log_dir)

    # 4. Clean stale ROCm Python artifacts from Docker image
    clean_stale_artifacts(cfg.log_dir)

    # 4b. Ensure cmake + toolchain exist (some base images ship without cmake)
    install_build_prereqs(cfg)

    # 4c. Install libnl-3 / libmnl headers required by amdsmi's CMake config.
    install_netlink_deps(cfg)

    # 5. Build
    if not cfg.skip_build:
        try:
            build_amdsmi(cfg)
        except CommandError as exc:
            _write_result(
                cfg.test_results_dir,
                "build_result.txt",
                f"BUILD FAILED: {exc.name} exited {exc.code}\n\n"
                f"Log ({exc.log_path}):\n{read_log(exc.log_path)}",
            )
            report_and_raise("BUILD", exc)

    artifact = locate_package(cfg.build_dir, cfg.package_format)
    print(f"Build artifact: {artifact}")
    _write_result(cfg.test_results_dir, "build_result.txt", f"BUILD PASSED\nArtifact: {artifact}")

    # 5b. Package ships the module on an import path (no install needed)
    try:
        verify_cpack_paths(cfg, artifact)
    except CommandError as exc:
        _write_result(
            cfg.test_results_dir,
            "cpack_path_result.txt",
            f"CPACK PATH CHECK FAILED: {exc.name} exited {exc.code}\n\n"
            f"Log ({exc.log_path}):\n{read_log(exc.log_path)}",
        )
        report_and_raise("CPACK PATH CHECK", exc)
    _write_result(cfg.test_results_dir, "cpack_path_result.txt", "CPACK PATH CHECK PASSED")

    # 5c. Loader behaves identically across supported Python versions
    try:
        verify_python_versions(cfg)
    except CommandError as exc:
        _write_result(
            cfg.test_results_dir,
            "python_versions_result.txt",
            f"PYTHON VERSIONS FAILED: {exc.name} exited {exc.code}\n\n"
            f"Log ({exc.log_path}):\n{read_log(exc.log_path)}",
        )
        report_and_raise("PYTHON VERSIONS", exc)
    _write_result(cfg.test_results_dir, "python_versions_result.txt", "PYTHON VERSIONS PASSED")

    # 5d. Upgrade/downgrade from a prior package (only when one is provided).
    # Deliberately ahead of the install/wheel stages: `pip install` puts the
    # wheel under /usr/local, which precedes the deb/rpm site-packages path for
    # `import amdsmi`, so running this afterwards would verify the wheel on
    # every transition instead of the package actually being upgraded.
    if not cfg.skip_install and cfg.upgrade_from:
        try:
            verify_upgrade_downgrade(cfg, artifact)
        except CommandError as exc:
            _write_result(
                cfg.test_results_dir,
                "upgrade_downgrade_result.txt",
                f"UPGRADE/DOWNGRADE FAILED: {exc.name} exited {exc.code}\n\n"
                f"Log ({exc.log_path}):\n{read_log(exc.log_path)}",
            )
            report_and_raise("UPGRADE/DOWNGRADE", exc)
        _write_result(
            cfg.test_results_dir, "upgrade_downgrade_result.txt", "UPGRADE/DOWNGRADE PASSED"
        )

    # 6. Install
    if not cfg.skip_install:
        try:
            install_package(cfg, artifact)
        except CommandError as exc:
            _write_result(
                cfg.test_results_dir,
                "install_result.txt",
                f"INSTALL FAILED: {exc.name} exited {exc.code}\n\n"
                f"Log ({exc.log_path}):\n{read_log(exc.log_path)}",
            )
            report_and_raise("INSTALL", exc)
        _write_result(
            cfg.test_results_dir, "install_result.txt", f"INSTALL PASSED\nPackage: {artifact}"
        )

    # 7. Verify wheel in site-packages
    if not cfg.skip_install:
        try:
            verify_wheel_site_packages(cfg)
        except CommandError as exc:
            _write_result(
                cfg.test_results_dir,
                "verify_wheel_result.txt",
                f"VERIFY WHEEL FAILED: {exc.name} exited {exc.code}\n\n"
                f"Log ({exc.log_path}):\n{read_log(exc.log_path)}",
            )
            report_and_raise("VERIFY WHEEL", exc)
        _write_result(cfg.test_results_dir, "verify_wheel_result.txt", "VERIFY WHEEL PASSED")

    # 8. SONAME distinctness (system vs wheel library)
    if not cfg.skip_install:
        try:
            verify_soname_distinct(cfg)
        except CommandError as exc:
            _write_result(
                cfg.test_results_dir,
                "pkg_conflict_result.txt",
                f"SONAME CHECK FAILED: {exc.name} exited {exc.code}\n\n"
                f"Log ({exc.log_path}):\n{read_log(exc.log_path)}",
            )
            report_and_raise("SONAME CHECK", exc)
        _write_result(cfg.test_results_dir, "pkg_conflict_result.txt", "SONAME CHECK PASSED")

    # 9. Dual-copy drift guard (system package installs the module twice)
    if not cfg.skip_install:
        try:
            verify_dual_copy(cfg)
        except CommandError as exc:
            _write_result(
                cfg.test_results_dir,
                "dual_copy_result.txt",
                f"DUAL COPY CHECK FAILED: {exc.name} exited {exc.code}\n\n"
                f"Log ({exc.log_path}):\n{read_log(exc.log_path)}",
            )
            report_and_raise("DUAL COPY CHECK", exc)
        _write_result(cfg.test_results_dir, "dual_copy_result.txt", "DUAL COPY CHECK PASSED")

    print("AMDSMI workflow complete")


if __name__ == "__main__":
    main()
