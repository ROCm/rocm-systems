#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Install or remove the amdsmi Python module from the packaged wheel.

Invoked by the DEB/RPM post-install and pre-remove scriptlets, running under the
target system interpreter. Prefers pip; falls back to a stdlib zip extraction so
no python3-pip dependency is required. Detecting the interpreter and its
site-packages at install time (on the target host) avoids the build-host vs
target-host python mismatch that a build-time baked path suffers from.

Usage: amdsmi_install_wheel.py {install|uninstall}
"""

import glob
import os
import shutil
import site
import subprocess
import sys
import sysconfig
import zipfile


def _find_wheel():
    here = os.path.dirname(os.path.abspath(__file__))
    matches = sorted(glob.glob(os.path.join(here, "wheels", "amdsmi-*.whl")))
    return matches[-1] if matches else None


def _target_sitelib():
    # Prefer a site-packages directory guaranteed to be on the interpreter's
    # sys.path. On Debian/Ubuntu that is /usr/lib/python3/dist-packages; on
    # RHEL/SLES/Fedora it is the versioned .../site-packages. Avoid
    # sysconfig purelib, which on Debian points at a path not on sys.path.
    try:
        candidates = site.getsitepackages()
    except Exception:
        candidates = []
    if "/usr/lib/python3/dist-packages" in candidates:
        return "/usr/lib/python3/dist-packages"
    for candidate in candidates:
        if candidate.endswith("/dist-packages"):
            return candidate
    for candidate in candidates:
        if candidate.endswith("/site-packages") and candidate.startswith("/usr/lib"):
            return candidate
    for candidate in candidates:
        if candidate.endswith("/site-packages"):
            return candidate
    if candidates:
        return candidates[0]
    return sysconfig.get_paths()["purelib"]


def _pip_env():
    env = dict(os.environ)
    # PEP 668: allow writing into an externally-managed system interpreter.
    env["PIP_BREAK_SYSTEM_PACKAGES"] = "1"
    # Silence the "running pip as root" advisory in scriptlet output.
    env["PIP_ROOT_USER_ACTION"] = "ignore"
    return env


def _pip_available():
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False


def _extract(wheel, dest):
    os.makedirs(dest, exist_ok=True)
    prior = os.path.join(dest, "amdsmi")
    if os.path.isdir(prior):
        shutil.rmtree(prior, ignore_errors=True)
    with zipfile.ZipFile(wheel) as zf:
        for name in zf.namelist():
            top = name.split("/", 1)[0]
            if name.startswith("amdsmi/") or top.endswith(".dist-info"):
                zf.extract(name, dest)


def install():
    wheel = _find_wheel()
    if not wheel:
        sys.stderr.write("[amdsmi] no packaged wheel found; skipping module install\n")
        return 0
    if _pip_available():
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            "--force-reinstall",
            wheel,
        ]
        result = subprocess.run(cmd, env=_pip_env())
        if result.returncode == 0:
            return 0
        sys.stderr.write("[amdsmi] pip install failed; falling back to stdlib extract\n")
    _extract(wheel, _target_sitelib())
    return 0


def uninstall():
    if _pip_available():
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", "amdsmi"],
            env=_pip_env(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    dest = _target_sitelib()
    shutil.rmtree(os.path.join(dest, "amdsmi"), ignore_errors=True)
    for distinfo in glob.glob(os.path.join(dest, "amdsmi-*.dist-info")):
        shutil.rmtree(distinfo, ignore_errors=True)
    return 0


def main(argv):
    if len(argv) != 2 or argv[1] not in ("install", "uninstall"):
        sys.stderr.write("usage: {} {{install|uninstall}}\n".format(argv[0]))
        return 2
    return install() if argv[1] == "install" else uninstall()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
