#!/usr/bin/env python3
#
# Copyright (C) Advanced Micro Devices. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
run_amdsmi_pkg_conflict_test.py
================================

Exercise the system-package + pip-wheel coexistence matrix in a clean
container so we catch SONAME collisions between ``/opt/rocm/lib64/libamd_smi.so``
(shipped by the .deb / .rpm) and ``<site>/amdsmi/libamd_smi_python.so``
(bundled by the wheel).

The scenarios we validate:

    1.  venv-isolation contract  -> with VIRTUAL_ENV set, dpkg/rpm install
                                    drops amdsmi.pth into the SYSTEM
                                    site-packages only, never into the venv.
    2.  alt-python tracking      -> if a second python3.X is symlinked as
                                    /usr/local/bin/python3 (winning in
                                    dpkg/rpm's stripped PATH), the
                                    postinst follows -- .pth lands in the
                                    NEW interpreter's purelib, not the
                                    previously-default one.
    3.  system pkg ALONE   -> import resolves to /opt/rocm/share/amd_smi
                              and loads /opt/rocm/lib64/libamd_smi.so
                              (also asserts prerm/preun deleted the .pth
                              when the package is removed)
    4.  wheel ALONE        -> import resolves to <site>/amdsmi/
                              and loads <site>/amdsmi/libamd_smi_python.so
    5.  system pkg, then wheel installed on top
                          -> sys.path order picks one; the OTHER .so must
                             never be dlopen'd as a side-effect (different
                             SONAMEs guarantee no symbol-table collision)
    6.  both installed + a forced dual-load via ``ctypes.CDLL(..., RTLD_GLOBAL)``
                          -> amdsmi_init / amdsmi_shut_down must complete
                             without segfault (the failure mode the
                             two-library design protects against)
    7.  uninstall ordering -> install system pkg + wheel; uninstall the
                              system pkg first; wheel must keep working
                              and must still load libamd_smi_python.so
                              (i.e. the wheel is self-contained and does
                              not depend on the system .so).

This script intentionally does NOT require GPU hardware -- ``amdsmi_init``
returning AmdSmiLibraryException because no GPU is present is fine; a
SIGSEGV / ImportError / SONAME match between the two libs is not.

Usage
-----
    python3 tests/run_amdsmi_pkg_conflict_test.py \\
        --build-dir /path/with/amd-smi-lib_*.deb \\
        --wheel-dir /path/with/amdsmi-*.whl \\
        --image ubuntu:22.04

The script is Python 3.6.8-safe so it can run inside RHEL 8 containers if
ever invoked from a downstream pipeline.
"""

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("pkg-conflict")


# ---------------------------------------------------------------------------
# Inline test driver -- runs INSIDE the container.
#
# Embedded as a string so the host script can be invoked anywhere without a
# helper file. Heredoc / docker cp would also work; a plain string keeps
# everything self-contained.
# ---------------------------------------------------------------------------
IN_CONTAINER_DRIVER = r"""
import json
import os
import subprocess
import sys
from pathlib import Path

PKG = os.environ["AMDSMI_PKG"]
WHEEL = os.environ["AMDSMI_WHEEL"]
FMT = os.environ["AMDSMI_FMT"]


def run(cmd, check=True):
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=check)


def install_system_pkg():
    if FMT == "deb":
        run(["dpkg", "--force-depends", "-i", PKG])
    else:
        run(["rpm", "-ivh", "--nodeps", PKG])


def remove_system_pkg():
    if FMT == "deb":
        # `dpkg -r` requires the package name, not the file
        name = subprocess.check_output(
            ["dpkg-deb", "-f", PKG, "Package"], universal_newlines=True
        ).strip()
        run(["dpkg", "--purge", "--force-depends", name], check=False)
    else:
        name = subprocess.check_output(
            ["rpm", "-qp", "--queryformat", "%{NAME}", PKG],
            universal_newlines=True,
        ).strip()
        run(["rpm", "-e", "--nodeps", name], check=False)


def _pip_supports_break_system(py):
    # Return ['--break-system-packages'] when supported (pip 23+), else [].
    out = subprocess.run(
        [py, "-m", "pip", "install", "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    if "--break-system-packages" in (out.stdout + out.stderr):
        return ["--break-system-packages"]
    return []


def install_wheel():
    flag = _pip_supports_break_system(sys.executable)
    run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", *flag, WHEEL],
        check=False,
    )


def uninstall_wheel():
    flag = _pip_supports_break_system(sys.executable)
    run(
        [sys.executable, "-m", "pip", "uninstall", "-y", *flag, "amdsmi"],
        check=False,
    )


def _reset_state():
    # Best-effort scrub of leftover install artefacts between scenarios.
    #
    # Each scenario must start from a clean slate so an earlier scenario's
    # bug cannot leak into the next one's assertions and either mask a real
    # regression (false PASS) or introduce a phantom one (false FAIL).
    # Removes any installed system pkg, any pip-installed amdsmi, any stray
    # amdsmi.pth in known site-packages dirs, and the isolation venv used
    # by scenario 1.
    remove_system_pkg()
    uninstall_wheel()
    # Scrub stray amdsmi.pth from any python on PATH.
    import glob as _g
    for py in sorted(set(_g.glob('/usr/bin/python3*') + _g.glob('/usr/local/bin/python3*'))):
        if not os.access(py, os.X_OK) or py.endswith(('-config', '.pyc')):
            continue
        try:
            site = subprocess.check_output(
                [py, '-c',
                 "import sysconfig; print(sysconfig.get_path('purelib'))"],
                universal_newlines=True, stderr=subprocess.DEVNULL,
            ).strip()
        except (subprocess.CalledProcessError, OSError):
            continue
        pth = Path(site) / 'amdsmi.pth'
        if pth.exists():
            try:
                pth.unlink()
            except OSError:
                pass
    # Scenario 1's isolation venv.
    _v = Path('/tmp/amdsmi-isolation-venv')
    if _v.exists():
        import shutil as _sh
        _sh.rmtree(_v, ignore_errors=True)


# F-31: scenarios append their numeric ID to this list on PASS so the
# driver can emit a final summary table -- much easier to scan in CI logs
# than scrolling for the per-scenario banners.
_PASSED_SCENARIOS = []


def _scenario_pass(num, title):
    _PASSED_SCENARIOS.append((num, title))
    print("STATUS:PASS scenario %d: %s" % (num, title), flush=True)


def probe_import():
    code = (
        "import json, sys, amdsmi\n"
        "from pathlib import Path\n"
        "from amdsmi import amdsmi_wrapper as w\n"
        "lib = getattr(w, '_loaded_lib_path', None)\n"
        "lib_str = str(Path(lib).resolve()) if lib else None\n"
        "mod_str = str(Path(amdsmi.__file__).resolve())\n"
        "print('AMDSMI_PROBE=' + json.dumps({"
        "'module_path': mod_str, 'loaded_lib': lib_str}))\n"
    )
    out = subprocess.check_output(
        [sys.executable, "-c", code], universal_newlines=True
    )
    for line in out.splitlines():
        if line.startswith("AMDSMI_PROBE="):
            return json.loads(line.split("=", 1)[1])
    raise RuntimeError("probe did not emit AMDSMI_PROBE marker:\n" + out)


def soname(path):
    if not path or not Path(path).exists():
        return None
    out = subprocess.check_output(
        ["objdump", "-p", path], universal_newlines=True, stderr=subprocess.DEVNULL
    )
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("SONAME"):
            return line.split(None, 1)[1]
    return None


def init_smoke(extra_dlopen=None):
    code_lines = [
        "import sys",
        "rc = 0",
    ]
    if extra_dlopen:
        code_lines += [
            "import ctypes",
            "ctypes.CDLL(%r, mode=ctypes.RTLD_GLOBAL)" % extra_dlopen,
        ]
    code_lines += [
        "import amdsmi",
        "try:",
        "    amdsmi.amdsmi_init()",
        "    handles = amdsmi.amdsmi_get_processor_handles()",
        "    print('init OK; handles=', len(handles))",
        "    amdsmi.amdsmi_shut_down()",
        "except amdsmi.AmdSmiLibraryException as exc:",
        "    print('init returned library exception (expected on non-GPU):', exc)",
        "except Exception as exc:",
        "    print('UNEXPECTED:', type(exc).__name__, exc)",
        "    rc = 1",
        "sys.exit(rc)",
    ]
    return subprocess.call([sys.executable, "-c", "\n".join(code_lines)])


# --------------------------------------------------------------------------
# Scenario 1: venv-isolation contract
# --------------------------------------------------------------------------
# The system package MUST target only the system Python, even when a venv
# is active in the calling shell. Verify that:
#   - dpkg/rpm install with VIRTUAL_ENV set drops `amdsmi.pth` ONLY into
#     the system site-packages, NOT into the venv site-packages.
#   - The interpreter logged by the postinst is the system one.
# A regression here would silently start polluting users' venvs again.
print("\nSTATUS:START scenario 1: venv-isolation contract", flush=True)
_reset_state()
import re as _re
import shutil as _sh
import venv as _venv

_VENV_DIR = Path("/tmp/amdsmi-isolation-venv")
if _VENV_DIR.exists():
    _sh.rmtree(_VENV_DIR)
_venv.create(str(_VENV_DIR), with_pip=False)
_VENV_PY = _VENV_DIR / "bin" / "python3"
_GET_SITE = (
    "import site, os\n"
    "for p in site.getsitepackages():\n"
    "    if os.path.isdir(p):\n"
    "        print(p); break\n"
)
venv_site = subprocess.check_output(
    [str(_VENV_PY), "-c", _GET_SITE], universal_newlines=True
).strip()
assert venv_site, "could not resolve venv site-packages"
print("  venv site-packages:", venv_site)

# Install with VIRTUAL_ENV set to mimic `sudo -E apt install` from inside
# a venv -- the strongest case where the postinst could leak into the
# venv if it inspected VIRTUAL_ENV.
_env = os.environ.copy()
_env["VIRTUAL_ENV"] = str(_VENV_DIR)
_env["PATH"] = str(_VENV_DIR / "bin") + os.pathsep + _env.get("PATH", "")
if FMT == "deb":
    subprocess.run(["dpkg", "--force-depends", "-i", PKG], env=_env, check=True)
else:
    subprocess.run(["rpm", "-ivh", "--nodeps", PKG], env=_env, check=True)

assert not (Path(venv_site) / "amdsmi.pth").exists(), (
    "REGRESSION: postinst polluted the venv site-packages at " + venv_site
)
sys_site = subprocess.check_output(
    ["python3", "-c", _GET_SITE], universal_newlines=True
).strip()
assert (Path(sys_site) / "amdsmi.pth").exists(), (
    "system python site-packages missing amdsmi.pth at " + sys_site
)
print("  \u2713 venv NOT polluted; system .pth present at", sys_site + "/amdsmi.pth")

# Postinst log must record the system interpreter, not the venv one.
_log_text = Path("/var/log/amd_smi_lib/postinst.log").read_text()
_interp_lines = _re.findall(r"INFO interpreter (\S+)", _log_text)
assert _interp_lines, "postinst did not log INFO interpreter line"
_last_interp = _interp_lines[-1]
assert str(_VENV_DIR) not in _last_interp, (
    "REGRESSION: postinst logged venv interpreter " + _last_interp
)
print("  \u2713 postinst logged system interpreter:", _last_interp)
_scenario_pass(1, "venv-isolation contract")

# Reset state so the next scenario starts from a fresh install.
remove_system_pkg()
_sh.rmtree(_VENV_DIR)

# --------------------------------------------------------------------------
# Scenario 2: alt-python tracking
# --------------------------------------------------------------------------
# When a user has multiple python3.X installed and changes which one
# `python3` resolves to (via update-alternatives, PATH ordering, or a
# symlink under /usr/local/bin which dpkg/rpm see first in their stripped
# PATH), the postinst must follow -- the .pth must land in the NEW
# python's site-packages, not the previously-default one. Otherwise
# `python3 -c "import amdsmi"` (using the new default) silently fails
# even though the package is "installed".
print("\nSTATUS:START scenario 2: alt-python tracking", flush=True)
_reset_state()
_default_py = subprocess.check_output(
    ["python3", "-c", "import sys; print(sys.executable)"], universal_newlines=True
).strip()
_default_purelib = subprocess.check_output(
    ["python3", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
    universal_newlines=True,
).strip()
# Find a second python3.X binary that exists on PATH and is NOT the
# current default. If we cannot find one (minimal docker images often
# ship only one), the contract is untestable here -- skip with a
# warning, do not fail.
import glob as _glob
_alt_py = None
for cand in sorted(set(_glob.glob("/usr/bin/python3.*") + _glob.glob("/usr/local/bin/python3.*"))):
    if not os.access(cand, os.X_OK) or cand.endswith(("-config", ".pyc")):
        continue
    try:
        cand_real = subprocess.check_output(
            [cand, "-c", "import sys; print(sys.executable)"],
            universal_newlines=True, stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError:
        continue
    if cand_real != _default_py and Path(cand_real).resolve() != Path(_default_py).resolve():
        _alt_py = cand
        break

if _alt_py is None:
    print("  SKIP: only one python3 found on this image; cannot test alt-python tracking")
    print("STATUS:SKIP scenario 2: alt-python tracking", flush=True)
else:
    _alt_purelib = subprocess.check_output(
        [_alt_py, "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
        universal_newlines=True,
    ).strip()
    assert _alt_purelib != _default_purelib, (
        "alt python " + _alt_py + " reports the same purelib as the default; "
        "test cannot distinguish the two"
    )
    print("  default python:", _default_py, "purelib:", _default_purelib)
    print("  alt     python:", _alt_py, "purelib:", _alt_purelib)

    # /usr/local/bin is first on dpkg's and rpm's stripped PATH, so
    # symlinking python3 there beats whatever was the system default.
    Path("/usr/local/bin").mkdir(parents=True, exist_ok=True)
    _alt_link = Path("/usr/local/bin/python3")
    if _alt_link.exists() or _alt_link.is_symlink():
        _alt_link.unlink()
    _alt_link.symlink_to(_alt_py)
    try:
        # Sanity: from a fresh shell with default PATH, python3 must
        # resolve to the alt now.
        _resolved = subprocess.check_output(
            ["env", "-i", "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
             "bash", "-c", "command -v python3"],
            universal_newlines=True,
        ).strip()
        assert _resolved == str(_alt_link), (
            "expected /usr/local/bin/python3 to win in stripped PATH, got " + _resolved
        )

        install_system_pkg()
        assert (Path(_alt_purelib) / "amdsmi.pth").exists(), (
            "REGRESSION: postinst did not follow alternative python3; "
            "expected " + _alt_purelib + "/amdsmi.pth"
        )
        # And the previous default must not have been written.
        if _default_purelib != _alt_purelib:
            _shadow = Path(_default_purelib) / "amdsmi.pth"
            assert not _shadow.exists(), (
                "REGRESSION: postinst wrote .pth to the previous default python "
                "site-packages " + str(_shadow)
            )
        # postinst.log must record the alt interpreter.
        _log_text = Path("/var/log/amd_smi_lib/postinst.log").read_text()
        _last_interp = _re.findall(r"INFO interpreter (\S+)", _log_text)[-1]
        assert _last_interp == str(_alt_link) or Path(_last_interp).resolve() == Path(_alt_py).resolve(), (
            "REGRESSION: postinst.log recorded " + _last_interp +
            " but expected the alternative python " + str(_alt_link)
        )
        print("  \u2713 postinst followed /usr/local/bin/python3 ->", _alt_py)
        print("  \u2713 .pth landed at", _alt_purelib + "/amdsmi.pth")

        # And prerm/preun must clean up the .pth in the alt python's
        # site-packages too.
        remove_system_pkg()
        assert not (Path(_alt_purelib) / "amdsmi.pth").exists(), (
            "REGRESSION: prerm/preun did not clean up amdsmi.pth in alt python "
            "site-packages " + _alt_purelib
        )
        print("  \u2713 prerm/preun cleaned up alt python .pth")
        _scenario_pass(2, "alt-python tracking")
    finally:
        if _alt_link.exists() or _alt_link.is_symlink():
            _alt_link.unlink()

# --------------------------------------------------------------------------
# Scenario 3: system pkg alone
# --------------------------------------------------------------------------
print("\nSTATUS:START scenario 3: system pkg ALONE", flush=True)
_reset_state()
install_system_pkg()
sys1 = probe_import()
print("  module:", sys1["module_path"])
print("  lib   :", sys1["loaded_lib"])
assert sys1["loaded_lib"] is not None and "/opt/rocm" in sys1["loaded_lib"], (
    "system-only must load /opt/rocm/lib64/libamd_smi.so; got: "
    + repr(sys1["loaded_lib"])
)
assert init_smoke() == 0, "system-only smoke failed"
_scenario_pass(3, "system pkg ALONE")

# --------------------------------------------------------------------------
# Scenario 4: wheel alone
# --------------------------------------------------------------------------
print("\nSTATUS:START scenario 4: wheel ALONE", flush=True)
remove_system_pkg()

# prerm/preun MUST delete the amdsmi.pth that the postinst dropped --
# otherwise `import amdsmi` after `apt remove` keeps loading from
# /opt/rocm even though the package is gone.
_sys_site_after_remove = subprocess.check_output(
    ["python3", "-c", _GET_SITE], universal_newlines=True
).strip()
_leaked = Path(_sys_site_after_remove) / "amdsmi.pth"
assert not _leaked.exists(), (
    "REGRESSION: dpkg --purge / rpm -e left amdsmi.pth behind at " + str(_leaked)
)
print("  \u2713 prerm/preun cleaned up amdsmi.pth")

install_wheel()
wh1 = probe_import()
print("  module:", wh1["module_path"])
print("  lib   :", wh1["loaded_lib"])
assert "/opt/rocm" not in (wh1["loaded_lib"] or ""), \
    "wheel-only must NOT load /opt/rocm/lib64/libamd_smi.so"
assert "libamd_smi_python.so" in (wh1["loaded_lib"] or ""), \
    "wheel-only must load the SONAME-renamed libamd_smi_python.so"
assert init_smoke() == 0, "wheel-only smoke failed"
_scenario_pass(4, "wheel ALONE")

# --------------------------------------------------------------------------
# Scenario 5: system pkg + wheel coexisting
# --------------------------------------------------------------------------
print("\nSTATUS:START scenario 5: system pkg + wheel coexisting", flush=True)
install_system_pkg()
co = probe_import()
print("  module:", co["module_path"])
print("  lib   :", co["loaded_lib"])
sys_lib = "/opt/rocm/lib64/libamd_smi.so"
wheel_lib = co["loaded_lib"]
sys_son = soname(sys_lib)
wheel_son = soname(wheel_lib)
print("  system SONAME :", sys_son)
print("  wheel SONAME  :", wheel_son)
assert sys_son and wheel_son, "could not read SONAMEs from both libraries"
assert sys_son != wheel_son, (
    "SONAME collision: %s vs %s -- the two-library design (BUILD_PYTHON_WHEEL)"
    " is broken." % (sys_son, wheel_son)
)
assert init_smoke() == 0, "coexisting smoke failed"
_scenario_pass(5, "system pkg + wheel coexisting")

# --------------------------------------------------------------------------
# Scenario 6: forced dual-load via RTLD_GLOBAL
# --------------------------------------------------------------------------
print("\nSTATUS:START scenario 6: forced dual-load (RTLD_GLOBAL)", flush=True)
rc = init_smoke(extra_dlopen=sys_lib)
assert rc == 0, (
    "amdsmi_init crashed when /opt/rocm/lib64/libamd_smi.so was force-loaded"
    " into the same process as the wheel's libamd_smi_python.so"
)
_scenario_pass(6, "forced dual-load (RTLD_GLOBAL)")

# --------------------------------------------------------------------------
# Scenario 7: uninstall ordering -- system pkg removed while wheel stays
# --------------------------------------------------------------------------
# Real user flow: a developer installs amd-smi-lib (.deb / .rpm) for the
# CLI, ALSO does `pip install amdsmi` for use in a venv-less Python
# script, then later runs `apt remove amd-smi-lib` to upgrade ROCm. The
# wheel install must SURVIVE that uninstall: import amdsmi must keep
# working and must keep loading the wheel's bundled libamd_smi_python.so
# (not the now-deleted /opt/rocm/lib64/libamd_smi.so). This is the
# uninstall-ordering contract the SONAME split was designed to support.
print("\nSTATUS:START scenario 7: uninstall ordering (system pkg first)", flush=True)
remove_system_pkg()
# After uninstall, the system .pth must be gone:
_sys_site_after = subprocess.check_output(
    ["python3", "-c", _GET_SITE], universal_newlines=True
).strip()
_leaked7 = Path(_sys_site_after) / "amdsmi.pth"
assert not _leaked7.exists(), (
    "REGRESSION: system pkg removal left amdsmi.pth at " + str(_leaked7)
)
# Wheel must still resolve and still load the wheel's .so:
wh7 = probe_import()
print("  module:", wh7["module_path"])
print("  lib   :", wh7["loaded_lib"])
assert wh7["loaded_lib"] is not None, (
    "REGRESSION: wheel cannot load its bundled .so after system pkg"
    " was uninstalled (loaded_lib is None); the wheel must be"
    " self-contained"
)
assert "libamd_smi_python.so" in wh7["loaded_lib"], (
    "REGRESSION: wheel loaded " + repr(wh7["loaded_lib"]) +
    " instead of its bundled libamd_smi_python.so after system pkg"
    " removal"
)
assert "/opt/rocm" not in wh7["loaded_lib"], (
    "REGRESSION: wheel kept resolving to /opt/rocm even though the"
    " system pkg was uninstalled: " + wh7["loaded_lib"]
)
assert init_smoke() == 0, "uninstall-order smoke failed"
print("  \u2713 wheel survived system pkg uninstall and stayed self-contained")
_scenario_pass(7, "uninstall ordering (system pkg first)")

# F-31: final summary table -- one line per scenario that PASSED. Failures
# above raise AssertionError before we get here, so reaching this block
# means every scenario either passed or skipped cleanly.
print("\n=========================== SUMMARY ===========================", flush=True)
for _num, _title in _PASSED_SCENARIOS:
    print("  PASS   scenario %d: %s" % (_num, _title), flush=True)
print("================================================================", flush=True)
print("ALL SCENARIOS PASSED", flush=True)
"""


def find_artifact(directory, pattern, label):
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise SystemExit("no %s matching %s in %s" % (label, pattern, directory))
    if len(matches) > 1:
        raise SystemExit(
            "ambiguous %s: pattern %s matched multiple files in %s -- refusing to"
            " guess which one to test:\n  %s"
            % (label, pattern, directory, "\n  ".join(str(m) for m in matches))
        )
    return matches[0]


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--build-dir", type=Path, required=True, help="dir with .deb or .rpm artifacts")
    p.add_argument("--wheel-dir", type=Path, required=True, help="dir with amdsmi-*.whl artifacts")
    p.add_argument(
        "--image",
        default=None,
        help=(
            "container image. If omitted, --os-label is used to resolve the image "
            "via `gh variable get <LABEL>_DOCKER_IMAGE` with a public fallback. "
            "Public fallbacks: ubuntu:20.04, ubuntu:22.04, debian:10, "
            "registry.suse.com/bci/bci-base:15.6, almalinux:8, "
            "registry.access.redhat.com/ubi{8,9,10}/ubi:latest, "
            "mcr.microsoft.com/azurelinux/base/core:3.0"
        ),
    )
    p.add_argument(
        "--os-label",
        default="Ubuntu22",
        choices=[
            "Ubuntu20",
            "Ubuntu22",
            "Ubuntu24",
            "Debian10",
            "SLES",
            "RHEL8",
            "RHEL9",
            "RHEL10",
            "AzureLinux3",
            "AlmaLinux8",
        ],
        help="CI matrix label (matches .github/workflows/amdsmi-build.yml). Used to "
        "resolve the image via vars.<LABEL>_DOCKER_IMAGE when --image is not given.",
    )
    return p.parse_args()


# Public fallbacks used when the GitHub repo variable can't be resolved.
_PUBLIC_FALLBACKS = {
    "Ubuntu20": "ubuntu:20.04",
    "Ubuntu22": "ubuntu:22.04",
    "Ubuntu24": "ubuntu:24.04",
    "Debian10": "debian:10",
    "SLES": "registry.suse.com/bci/bci-base:15.6",
    "RHEL8": "registry.access.redhat.com/ubi8/ubi:latest",
    "RHEL9": "registry.access.redhat.com/ubi9/ubi:latest",
    "RHEL10": "registry.access.redhat.com/ubi10/ubi:latest",
    "AzureLinux3": "mcr.microsoft.com/azurelinux/base/core:3.0",
    "AlmaLinux8": "almalinux:8",
}


def resolve_image(label: str) -> str:
    """Prefer the CI image from `gh variable get`, fall back to a public image."""
    if shutil.which("gh"):
        # Defense-in-depth: subprocess.run already enforces timeout=10, but if
        # the system `timeout(1)` is available we wrap with a slightly longer
        # shell-level limit so a wedged child cannot exceed even if Python's
        # alarm-based timeout were to misfire on this platform.
        gh_cmd = ["gh", "variable", "get", f"{label}_DOCKER_IMAGE"]
        if shutil.which("timeout"):
            gh_cmd = ["timeout", "15", *gh_cmd]
        try:
            out = subprocess.run(gh_cmd, capture_output=True, text=True, timeout=10)
            img = out.stdout.strip()
            if out.returncode == 0 and img:
                return img
        except (subprocess.SubprocessError, OSError):
            pass
    return _PUBLIC_FALLBACKS[label]


def main():
    args = parse_args()

    if not shutil.which("docker"):
        raise SystemExit("docker is required to run the conflict matrix")

    image = args.image or resolve_image(args.os_label)
    img_lc = image.lower()
    fmt = (
        "rpm"
        if any(
            tag in img_lc
            for tag in ("ubi", "rhel", "fedora", "suse", "sles", "azurelinux", "almalinux", "rocky")
        )
        else "deb"
    )
    # Match the main runtime package only; exclude `-tests` subpackage
    # and other suffixes by anchoring the version digit immediately
    # after `amd-smi-lib-` / `amd-smi-lib_`.
    pkg_pattern = "amd-smi-lib-[0-9]*.rpm" if fmt == "rpm" else "amd-smi-lib_[0-9]*.deb"
    pkg_path = find_artifact(args.build_dir, pkg_pattern, fmt + " package").resolve()
    wheel_path = find_artifact(args.wheel_dir, "amdsmi-*.whl", "wheel").resolve()

    log.info("image : %s", image)
    log.info("pkg   : %s (%s)", pkg_path, fmt)
    log.info("wheel : %s", wheel_path)

    # The driver script lives in a private writable tempdir; the artifact dirs
    # may be root-owned (built in a container) and not writable.
    import tempfile

    work = Path(tempfile.mkdtemp(prefix="amdsmi-conflict-"))
    driver_path = work / "pkg_conflict_driver.py"
    driver_path.write_text(IN_CONTAINER_DRIVER)

    # Path translation: artifacts live at /work and /wheel inside the container.
    pkg_in = "/pkg/" + pkg_path.name
    whl_in = "/wheel/" + wheel_path.name
    drv_in = "/driver/" + driver_path.name

    # Note on scenario 2 (alt-python tracking): each matrix image ships exactly
    # ONE pythonX.Y from default repos (Ubuntu24->3.12, Ubuntu22->3.10,
    # RHEL9->3.9, RHEL8->3.6, ...). Enabling PPA/CRB/EPEL just to install a
    # second interpreter would more than double container setup time for a
    # single-scenario coverage win. Scenario 2 therefore prints STATUS:SKIP on
    # images without an alternate python3.X already installed; that's the
    # expected outcome and not a test gap.
    setup = (
        "set -e; "
        "if [ %s = deb ]; then "
        "  export DEBIAN_FRONTEND=noninteractive; "
        "  apt-get update -qq >/dev/null && "
        # python3-venv is REQUIRED by scenario 1 (venv-isolation):
        # `import venv; venv.create(...)` calls ensurepip on Debian/Ubuntu
        # which only ships in the python3-venv package.
        "  apt-get install -y -qq python3 python3-pip python3-venv binutils >/dev/null; "
        "else "
        "  yum install -y -q python3 python3-pip binutils >/dev/null; "
        "fi"
    ) % fmt

    cmd = [
        "docker",
        "run",
        "--rm",
        "-e",
        "AMDSMI_PKG=" + pkg_in,
        "-e",
        "AMDSMI_WHEEL=" + whl_in,
        "-e",
        "AMDSMI_FMT=" + fmt,
        "-v",
        "%s:/pkg:ro" % pkg_path.parent,
        "-v",
        "%s:/wheel:ro" % wheel_path.parent,
        "-v",
        "%s:/driver:ro" % work,
        image,
        "bash",
        "-c",
        setup + " && python3 " + drv_in,
    ]

    log.info("launching: %s", " ".join(cmd))
    rc = subprocess.call(cmd)
    shutil.rmtree(work, ignore_errors=True)

    if rc != 0:
        log.error("conflict matrix FAILED (rc=%d)", rc)
        sys.exit(rc)
    log.info("conflict matrix PASSED")


if __name__ == "__main__":
    main()
