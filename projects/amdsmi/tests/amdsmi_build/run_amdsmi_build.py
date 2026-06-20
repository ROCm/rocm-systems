#!/usr/bin/env python3
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
            subprocess.check_call(
                ["apt-get", "update"],
                stdout=subprocess.DEVNULL,
            )
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
            ["alternatives", "--set", "python3", new_py],
            stdout=subprocess.DEVNULL,
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

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
    if shutil.which("cmake") and shutil.which("make") and (
        shutil.which("gcc") or shutil.which("cc")
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
            [
                "zypper",
                "--non-interactive",
                "install",
                "cmake",
                "gcc",
                "gcc-c++",
                "make",
                "git",
            ],
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
            cmd,
            name=f"{installer}-install-netlink",
            retries=cfg.retries,
            log_dir=cfg.log_dir,
        )
    elif cfg.package_manager == "zypper":
        run_command(
            [
                "zypper",
                "--non-interactive",
                "install",
                "libnl3-devel",
                "libmnl-devel",
            ],
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

    cmake_args = [
        "cmake",
        str(cfg.project_dir),
        "-DBUILD_TESTS=ON",
        "-DENABLE_ESMI_LIB=ON",
    ]
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

    # Verify installation: CLI version + Python import/init/shutdown under
    # the SYSTEM python. The system package installs amdsmi/ to the path
    # /usr/bin/python3 searches (see py-interface/CMakeLists.txt). The
    # test must use /usr/bin/python3 explicitly -- some build containers
    # (notably ubuntu-24.04-bld) put a venv ahead of /usr/bin on PATH,
    # and that venv has its own sys.path that does NOT include the
    # system dist-packages. Falls back to plain `python3` if /usr/bin/python3
    # is absent.
    system_python = "/usr/bin/python3" if Path("/usr/bin/python3").exists() else "python3"
    import_smoke = (
        "import amdsmi; "
        "print('amdsmi from:', amdsmi.__file__); "
        "amdsmi.amdsmi_init(); "
        "amdsmi.amdsmi_shut_down(); "
        "print('init/shutdown ok')"
    )
    verify_commands = [
        [str(rocm_binary), "version"],
        [system_python, "-c", import_smoke],
    ]
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

    smoke_test = (
        "import amdsmi\n"
        "print('PASS: import amdsmi OK')\n"
        "amdsmi.amdsmi_init()\n"
        "print('PASS: amdsmi_init() OK')\n"
        "devs = amdsmi.amdsmi_get_processor_handles()\n"
        "print('PASS: Found %d device(s)' % len(devs))\n"
        "amdsmi.amdsmi_shut_down()\n"
        "print('PASS: amdsmi_shut_down() OK')\n"
        "print('=== Wheel verification passed ===')\n"
    )
    run_command(
        ["python3", "-c", smoke_test],
        name="wheel-smoke-test",
        cwd=Path("/tmp"),
        retries=1,
        log_dir=cfg.log_dir,
    )

    run_command(
        ["python3", "-m", "pip", "show", "amdsmi"],
        name="pip-show-amdsmi",
        retries=1,
        log_dir=cfg.log_dir,
    )

    # Check install location -- the wheel must land under site-packages
    # or dist-packages; the system DEB/RPM also installs to dist-packages,
    # so a coexisting system module is fine (whichever sys.path entry wins
    # is whichever is searched first by the active python).
    run_command(
        [
            "python3",
            "-c",
            (
                "import amdsmi; p = amdsmi.__file__; "
                "print('amdsmi imported from: ' + p); "
                "ok = 'site-packages' in p or 'dist-packages' in p or '/opt/rocm/' in p; "
                "assert ok, 'Unexpected install location: ' + p; "
                "print('PASS: Wheel correctly installed')"
            ),
        ],
        name="wheel-location-check",
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
    package_manager = detect_package_manager(
        args.package_manager or profile.get("package_manager")
    )
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
        skip_setuptools_upgrade=args.skip_setuptools_upgrade or profile.get("skip_setuptools_upgrade", False),
        install_more_itertools=args.install_more_itertools or profile.get("install_more_itertools", False),
        qa_rpaths=args.qa_rpaths or profile.get("qa_rpaths", False),
        skip_build=args.skip_build,
        skip_install=args.skip_install,
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

    # 2. amd-smi command test logs -- logged but non-fatal
    cmd_fails: List[str] = []
    for log in sorted(results_dir.glob("amd-smi_*.log")):
        log_text = log.read_text(encoding="utf-8", errors="replace")
        if any(token in log_text for token in ("Traceback", "AmdSmiException", "Error code:")):
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
            details.append(
                f"#### AMDSMI Tests \u2014 {gtest_fails} failure(s)\n\n" + _fenced(text)
            )

    # 4. Python test outputs
    import re as _re

    fail_re = _re.compile(r"^(FAIL|ERROR):", _re.MULTILINE)
    for test_file in ("integration_test_output.txt", "unit_test_output.txt", "perf_test_output.txt"):
        full = results_dir / test_file
        if not full.exists():
            continue
        text = full.read_text(encoding="utf-8", errors="replace")
        py_fails = len(fail_re.findall(text))
        if py_fails > 0:
            name = test_file.replace("_output.txt", "").replace("_", " ")
            failures.append(f"{name} ({py_fails})")
            details.append(
                f"#### {name} \u2014 {py_fails} failure(s)\n\n" + _fenced(text)
            )

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
        cmd_summary = (
            f"\n\n:warning: Command tests (non-fatal): `{' '.join(cmd_fails)}`\n"
        )

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


# ---------------------------------------------------------------------------
# Multi-Python sanity sweep (the "python-matrix" subcommand)
#
# Builds/installs once, then runs the *installed* Python test suite under a
# range of interpreters so CI produces per-version PASS/FAIL sanity data on a
# real GPU.  The amdsmi package is a pure-ctypes wrapper around libamd_smi.so
# (py-interface/setup.py.in: python_requires>=3.6, no ext_modules) and the
# tests import it from $AMDSMI_PATH (tests/python_unittest/common.py), so one
# install is exercised by every interpreter -- no per-version venvs or wheel
# installs are required.
#
# Interpreters are resolved per requested minor version by (1) a system
# pythonX.Y already on PATH -- which covers distro pythons uv cannot supply,
# e.g. 3.6 on AlmaLinux8 -- then (2) a uv-managed download.  Versions that
# neither source can provide are recorded as SKIP rather than failing.
# ---------------------------------------------------------------------------

DEFAULT_PY_MATRIX_VERSIONS = "3.6 3.7 3.8 3.9 3.10 3.11 3.12 3.13 3.14"
DEFAULT_AMDSMI_PATH = Path("/opt/rocm/share/amd_smi")
DEFAULT_UV_VERSION = "0.11.23"
PY_MATRIX_TEST_FILES = ("integration_test.py", "unit_tests.py", "cli_unit_test.py")


@dataclass
class PyVersionResult:
    requested: str
    status: str = "SKIP"  # "PASS" | "FAIL" | "SKIP"
    interpreter: str = ""
    resolved: str = ""
    note: str = ""
    failed_tests: List[str] = field(default_factory=list)


def parse_python_versions(spec: str) -> List[str]:
    """Parse a whitespace/comma separated version spec into a deduped list.

    Accepts ``X.Y`` or ``X.Y.Z`` tokens; raises ValueError on anything else.
    """
    import re

    out: List[str] = []
    for tok in re.split(r"[,\s]+", spec.strip()):
        if not tok:
            continue
        if not re.fullmatch(r"\d+\.\d+(?:\.\d+)?", tok):
            raise ValueError(f"Invalid Python version token: {tok!r}")
        if tok not in out:
            out.append(tok)
    return out


def minor_of(version: str) -> str:
    """Return the ``X.Y`` minor prefix of a version string."""
    import re

    m = re.match(r"^\s*(\d+\.\d+)", version)
    if not m:
        raise ValueError(f"Cannot parse minor version from {version!r}")
    return m.group(1)


def _py_status_icon(status: str) -> str:
    return {
        "PASS": ":white_check_mark:",
        "FAIL": ":x:",
        "SKIP": ":fast_forward:",
    }.get(status, ":grey_question:")


def render_python_matrix_summary(os_label: str, results: List[PyVersionResult]) -> str:
    """Render a markdown table of per-interpreter results."""
    n_pass = sum(1 for r in results if r.status == "PASS")
    n_fail = sum(1 for r in results if r.status == "FAIL")
    n_skip = sum(1 for r in results if r.status == "SKIP")

    lines = [
        f"## Python Version Matrix \u2014 {os_label}",
        "",
        "| Requested | Status | Interpreter | Detail |",
        "| --- | --- | --- | --- |",
    ]
    for r in results:
        interp = f"{r.resolved} ({r.interpreter})" if r.resolved else "\u2014"
        detail = ("failed: " + ", ".join(r.failed_tests)) if r.failed_tests else r.note
        lines.append(
            f"| {r.requested} | {_py_status_icon(r.status)} {r.status} | {interp} | {detail} |"
        )
    lines += [
        "",
        f"**{n_pass} passed, {n_fail} failed, {n_skip} skipped** "
        f"(of {len(results)} requested)",
        "",
    ]
    return "\n".join(lines)


def _query_interpreter_version(python_bin: str) -> Optional[str]:
    """Return ``X.Y.Z`` reported by *python_bin*, or None if it cannot run."""
    try:
        proc = subprocess.run(
            [python_bin, "-c", "import sys;print('%d.%d.%d' % sys.version_info[:3])"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def ensure_uv(log_dir: Path, uv_version: str, install_dir: Path) -> Optional[Path]:
    """Return a path to a uv binary, fetching a pinned release if not on PATH."""
    on_path = shutil.which("uv")
    if on_path:
        print(f"python-matrix: using uv on PATH: {on_path}")
        return Path(on_path)

    import platform
    import tarfile
    import urllib.request

    machine = platform.machine().lower()
    arch = {
        "x86_64": "x86_64",
        "amd64": "x86_64",
        "aarch64": "aarch64",
        "arm64": "aarch64",
    }.get(machine)
    if arch is None:
        print(f"python-matrix: unsupported arch {machine!r}; cannot fetch uv")
        return None

    triple = f"uv-{arch}-unknown-linux-gnu"
    url = (
        f"https://github.com/astral-sh/uv/releases/download/"
        f"{uv_version}/{triple}.tar.gz"
    )
    install_dir.mkdir(parents=True, exist_ok=True)
    tarball = install_dir / f"{triple}.tar.gz"
    uv_dest = install_dir / "uv"
    print(f"python-matrix: downloading uv {uv_version} from {url}")
    try:
        with urllib.request.urlopen(url, timeout=180) as resp, tarball.open("wb") as fh:
            shutil.copyfileobj(resp, fh)
        # Extract only the uv binary (by basename) to avoid tar path traversal.
        with tarfile.open(tarball) as tf:
            member = next(
                (
                    m
                    for m in tf.getmembers()
                    if m.isfile() and os.path.basename(m.name) == "uv"
                ),
                None,
            )
            if member is None:
                print("python-matrix: uv binary not present in archive")
                return None
            src = tf.extractfile(member)
            if src is None:
                return None
            with src, uv_dest.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    except Exception as exc:  # best-effort: network/extract failure -> skip uv
        print(f"python-matrix: failed to fetch uv: {exc}")
        return None

    uv_dest.chmod(0o755)
    print(f"python-matrix: uv ready at {uv_dest}")
    return uv_dest


def resolve_interpreter(
    version: str, uv_bin: Optional[Path], log_dir: Path
) -> Tuple[Optional[str], str, str]:
    """Resolve *version* to an interpreter path.

    Returns ``(path_or_None, resolved_version, note)``.
    """
    minor = minor_of(version)

    # 1. System interpreter on PATH (pythonX.Y) -- covers distro pythons that
    #    uv cannot provide (e.g. 3.6) and skips an unnecessary download.
    cand = shutil.which(f"python{minor}")
    if cand:
        got = _query_interpreter_version(cand)
        if got and minor_of(got) == minor:
            return cand, got, "system"

    # 2. uv-managed download (python-build-standalone) -- covers 3.8-3.14.
    if uv_bin is not None:
        try:
            run_command(
                [str(uv_bin), "python", "install", version],
                name=f"uv-install-{version}",
                retries=2,
                log_dir=log_dir,
            )
        except CommandError:
            return None, "", f"uv could not install {version}"
        try:
            found = subprocess.run(
                [str(uv_bin), "python", "find", version],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=60,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return None, "", f"uv find failed for {version}: {exc}"
        out = found.stdout.strip()
        path = out.splitlines()[0].strip() if out else ""
        if found.returncode == 0 and path:
            got = _query_interpreter_version(path)
            if got and minor_of(got) == minor:
                return path, got, "uv"

    return None, "", "not available via system PATH or uv"


def _run_suite_for_interpreter(
    python_bin: str,
    tests_dir: Path,
    amdsmi_path: Path,
    results_dir: Path,
    version_tag: str,
    log_dir: Path,
) -> List[str]:
    """Run the installed Python test suite under *python_bin*.

    Returns the list of failing test files (empty == all passed/absent).
    """
    env = dict(os.environ)
    env["AMDSMI_PATH"] = str(amdsmi_path)
    # Prepend the install to PYTHONPATH so `import amdsmi` resolves to it
    # under every interpreter (TheRock tarball layout keeps it off sys.path).
    pp_parts = [str(amdsmi_path)]
    if env.get("PYTHONPATH"):
        pp_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pp_parts)

    # PATH shim so the `amd-smi` CLI subprocess (shebang: /usr/bin/env python3)
    # spawned by cli_unit_test.py also runs under *python_bin*.
    # Absolute so the shim resolves regardless of each test's working dir.
    shim_dir = Path(os.path.abspath(str(log_dir))) / f"py-shim-{version_tag}"
    shim_dir.mkdir(parents=True, exist_ok=True)
    shim = shim_dir / "python3"
    try:
        if shim.exists() or shim.is_symlink():
            shim.unlink()
        shim.symlink_to(python_bin)
        env["PATH"] = str(shim_dir) + os.pathsep + env.get("PATH", "")
    except OSError:
        pass  # shim is best-effort; tests still run under python_bin directly

    failed: List[str] = []
    for test_file in PY_MATRIX_TEST_FILES:
        script = tests_dir / test_file
        if not script.exists():
            continue
        stem = test_file[:-3] if test_file.endswith(".py") else test_file
        result_path = results_dir / f"python_{version_tag}_{stem}.log"
        try:
            run_command(
                [python_bin, str(script), "-v"],
                name=f"py{version_tag}-{stem}",
                cwd=tests_dir,
                env=env,
                retries=1,
                log_dir=log_dir,
                result_file=result_path,
            )
        except CommandError:
            failed.append(stem)
    return failed


def run_python_matrix(
    versions: List[str],
    tests_dir: Path,
    amdsmi_path: Path,
    results_dir: Path,
    os_label: str,
    uv_version: str,
    log_dir: Path,
    summary_file: Optional[Path],
) -> int:
    """Resolve interpreters, run the suite per version, emit a summary table.

    Returns the number of versions that FAILED (skips do not count).
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    rows: List[PyVersionResult] = []

    if not tests_dir.exists():
        print(f"python-matrix: tests dir {tests_dir} does not exist; nothing to run")
    else:
        uv_bin = ensure_uv(log_dir, uv_version, log_dir / "uv")
        if uv_bin is None:
            print("python-matrix: proceeding with system interpreters only (no uv)")
        for ver in versions:
            path, resolved, note = resolve_interpreter(ver, uv_bin, log_dir)
            if path is None:
                print(f"python-matrix: {ver} -> SKIP ({note})")
                rows.append(PyVersionResult(requested=ver, status="SKIP", note=note))
                continue
            print(f"python-matrix: {ver} -> {resolved} at {path} (via {note})")
            failed = _run_suite_for_interpreter(
                path, tests_dir, amdsmi_path, results_dir, ver, log_dir
            )
            rows.append(
                PyVersionResult(
                    requested=ver,
                    status="FAIL" if failed else "PASS",
                    interpreter=path,
                    resolved=resolved,
                    failed_tests=failed,
                    note=note,
                )
            )

    summary = render_python_matrix_summary(os_label, rows)
    (results_dir / "python_matrix_summary.md").write_text(summary, encoding="utf-8")
    if summary_file is not None:
        with summary_file.open("a", encoding="utf-8") as fh:
            fh.write(summary)

    print("=" * 42)
    print(summary)

    n_fail = sum(1 for r in rows if r.status == "FAIL")
    if n_fail:
        failed_versions = ", ".join(r.requested for r in rows if r.status == "FAIL")
        print(f"::error::Python matrix failures for {os_label}: {failed_versions}")
    else:
        print(f"python-matrix: all resolved interpreters PASSED for {os_label}")
    return n_fail


def _python_matrix_cli(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="run_amdsmi_build.py python-matrix",
        description="Run the installed Python test suite under several interpreters.",
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--os-label", default="local")
    parser.add_argument("--amdsmi-path", type=Path, default=DEFAULT_AMDSMI_PATH)
    parser.add_argument(
        "--tests-dir",
        type=Path,
        default=None,
        help="Default: <amdsmi-path>/tests/python_unittest",
    )
    parser.add_argument("--python-versions", default=DEFAULT_PY_MATRIX_VERSIONS)
    parser.add_argument("--uv-version", default=DEFAULT_UV_VERSION)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=None,
        help="Append the markdown matrix to this file (e.g. $GITHUB_STEP_SUMMARY).",
    )
    args = parser.parse_args(argv)

    versions = parse_python_versions(args.python_versions)
    tests_dir = args.tests_dir or (args.amdsmi_path / "tests" / "python_unittest")
    return run_python_matrix(
        versions=versions,
        tests_dir=tests_dir,
        amdsmi_path=args.amdsmi_path,
        results_dir=args.results_dir,
        os_label=args.os_label,
        uv_version=args.uv_version,
        log_dir=args.log_dir,
        summary_file=args.summary_file,
    )


def main() -> None:
    # `summarize` subcommand short-circuits before parse_args/sudo logic.
    if len(sys.argv) > 1 and sys.argv[1] == "summarize":
        sys.exit(0 if _summarize_cli(sys.argv[2:]) == 0 else 1)
    if len(sys.argv) > 1 and sys.argv[1] == "python-matrix":
        sys.exit(0 if _python_matrix_cli(sys.argv[2:]) == 0 else 1)
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

    print("AMDSMI workflow complete")


if __name__ == "__main__":
    main()
