#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Install or remove the amdsmi Python module from the packaged wheel.

Invoked by the DEB/RPM post-install and pre-remove scriptlets, running under the
target system interpreter. Prefers pip; falls back to a stdlib zip extraction so
python3-pip is not a hard dependency. Both paths install into the same detected
site-packages directory, so uninstall is deterministic. Detecting the
interpreter and its site-packages here (on the target host, at install time)
avoids the build-host vs target-host python mismatch a build-time baked path
suffers from.

Usage: amdsmi_install_wheel.py {install|uninstall}
"""

import argparse
import os
import re
import shutil
import site
import subprocess
import sys
import sysconfig
import zipfile
from pathlib import Path


def _find_wheel() -> "Path | None":
    matches = sorted(Path(__file__).resolve().parent.glob("wheels/amdsmi-*.whl"))
    return matches[-1] if matches else None


def _wheel_version(wheel: Path) -> "str | None":
    match = re.match(r"amdsmi-(.+)-py3-none-any\.whl$", wheel.name)
    return match.group(1) if match else None


def _target_sitelib() -> Path:
    # Prefer a site-packages directory guaranteed to be on the interpreter's
    # sys.path. On Debian/Ubuntu that is /usr/lib/python3/dist-packages; on
    # RHEL/SLES/Fedora it is the versioned .../site-packages under /usr/lib.
    try:
        candidates = site.getsitepackages()
    except AttributeError:
        candidates = []
    if "/usr/lib/python3/dist-packages" in candidates:
        return Path("/usr/lib/python3/dist-packages")
    for candidate in candidates:
        if candidate.endswith("/dist-packages"):
            return Path(candidate)
    # A noarch wheel belongs in purelib (/usr/lib/...), not platlib
    # (/usr/lib64/...); the trailing slash keeps /usr/lib64 from matching.
    for candidate in candidates:
        if candidate.endswith("/site-packages") and candidate.startswith("/usr/lib/"):
            return Path(candidate)
    for candidate in candidates:
        if candidate.endswith("/site-packages"):
            return Path(candidate)
    if candidates:
        return Path(candidates[0])
    return Path(sysconfig.get_paths()["purelib"])


def _pip_env() -> dict:
    env = dict(os.environ)
    # PEP 668: allow writing into an externally-managed system interpreter.
    env["PIP_BREAK_SYSTEM_PACKAGES"] = "1"
    # Silence the "running pip as root" advisory in scriptlet output.
    env["PIP_ROOT_USER_ACTION"] = "ignore"
    return env


def _pip_available() -> bool:
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def _extract(wheel: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    prior = dest / "amdsmi"
    if prior.is_dir():
        shutil.rmtree(prior, ignore_errors=True)
    dest_root = str(dest.resolve()) + os.sep
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            top = name.split("/", 1)[0]
            if not (name.startswith("amdsmi/") or top.endswith(".dist-info")):
                continue
            # Defense in depth: ZipFile.extract() already strips traversal, but
            # skip anything that would still resolve outside dest.
            if not (str((dest / name).resolve()) + os.sep).startswith(dest_root):
                continue
            archive.extract(name, dest)


def _remove_installed(sitelib: Path, version: "str | None") -> None:
    shutil.rmtree(sitelib / "amdsmi", ignore_errors=True)
    if version:
        shutil.rmtree(sitelib / "amdsmi-{}.dist-info".format(version), ignore_errors=True)


def install() -> int:
    wheel = _find_wheel()
    if not wheel:
        sys.stderr.write("[amdsmi] no packaged wheel found; skipping module install\n")
        return 0
    sitelib = _target_sitelib()
    version = _wheel_version(wheel)
    # Deterministic replace: clear any prior copy first so both the pip --target
    # path (which does not reliably overwrite an existing target) and the extract
    # fallback start from a clean state.
    _remove_installed(sitelib, version)
    if _pip_available():
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            "--no-cache-dir",
            "--force-reinstall",
            "--target",
            str(sitelib),
            str(wheel),
        ]
        if subprocess.run(cmd, env=_pip_env()).returncode == 0:
            return 0
        sys.stderr.write("[amdsmi] pip install failed; falling back to stdlib extract\n")
    _extract(wheel, sitelib)
    return 0


def uninstall() -> int:
    wheel = _find_wheel()
    sitelib = _target_sitelib()
    version = _wheel_version(wheel) if wheel else None
    # Only remove what this package installed: the amdsmi tree plus the
    # dist-info matching our shipped version. If a different version is present
    # (a user pip-installed their own amdsmi over ours), leave it untouched.
    our_distinfo = sitelib / "amdsmi-{}.dist-info".format(version) if version else None
    if our_distinfo is not None and our_distinfo.is_dir():
        _remove_installed(sitelib, version)
    elif version is None:
        # No shipped wheel to identify our version: remove only the module tree
        # we would have installed, not any dist-info we cannot attribute here.
        shutil.rmtree(sitelib / "amdsmi", ignore_errors=True)
    return 0


def main(argv: "list[str]") -> int:
    parser = argparse.ArgumentParser(description="Install or remove the amdsmi Python module.")
    parser.add_argument("action", choices=["install", "uninstall"])
    args = parser.parse_args(argv[1:])
    return install() if args.action == "install" else uninstall()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
