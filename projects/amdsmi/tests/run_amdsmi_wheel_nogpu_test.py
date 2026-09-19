#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
run_amdsmi_wheel_nogpu_test.py
==============================

Install a built amdsmi wheel on a host with no GPU and no ROCm, then assert the
runtime contract holds:

  * the wheel is self-contained -- it ships libamd_smi_python.so and, when
    repaired for manylinux, the vendored netlink libraries the rpm/deb expects
    the system to provide
  * import resolves that bundled library, never a system libamd_smi.so
  * amdsmi_init() / amdsmi_get_processor_handles() / amdsmi_shut_down() succeed
    and report zero processors instead of raising

The install itself is the other half of the test. A wheel tagged
manylinux_2_28 is only installable by pip 20.3+, and several distros in the
support matrix ship an older pip whose rejection message ("not a supported
wheel on this platform") points at the architecture rather than at pip. This
harness checks the pip version up front so that failure names its own cause.

    python3 tests/run_amdsmi_wheel_nogpu_test.py --wheel wheels/ --expect-no-gpu

Python 3.6-safe: the oldest supported distro (AlmaLinux 8 / RHEL 8) ships
CPython 3.6.8, so no walrus, no ``text=``, no PEP 604 unions.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

# PEP 600 (manylinux_<glibcmajor>_<glibcminor>) tags landed in pip 20.3.
MIN_PIP_FOR_PEP600 = (20, 3)

# Netlink libraries libamd_smi links unconditionally (src/ualoe_lib). A
# repaired wheel must vendor them; the rpm/deb instead expects them from the
# distro, which is why the wheel works on a bare container and the package
# does not.
VENDORED_NETLINK_PREFIXES = ("libnl-3", "libnl-genl-3", "libmnl")

# Asserted inside the freshly installed environment, where amdsmi is importable.
CONTRACT = r"""
import os, sys
import amdsmi
import amdsmi.amdsmi_wrapper as w

lib = w._loaded_lib_path
print("  version      :", amdsmi.__version__)
print("  loaded lib   :", lib)
print("  sys fallback :", w._AMDSMI_ALLOW_SYSTEM_FALLBACK)

if lib is None:
    raise SystemExit("FAIL: no library loaded; the wheel shipped no libamd_smi_python.so")
if os.path.basename(lib) != "libamd_smi_python.so":
    raise SystemExit("FAIL: loaded %s; expected the wheel-private libamd_smi_python.so" % lib)
# The bundled .so must sit next to the imported package. A different directory
# means the loader reached outside the wheel for a system library.
if os.path.dirname(os.path.realpath(lib)) != os.path.dirname(os.path.realpath(amdsmi.__file__)):
    raise SystemExit("FAIL: %s is not the .so bundled beside %s" % (lib, amdsmi.__file__))
if w._AMDSMI_ALLOW_SYSTEM_FALLBACK:
    raise SystemExit("FAIL: wheel shipped with the system-library fallback enabled")

amdsmi.amdsmi_init()
try:
    handles = amdsmi.amdsmi_get_processor_handles()
finally:
    amdsmi.amdsmi_shut_down()

if not isinstance(handles, list):
    raise SystemExit("FAIL: amdsmi_get_processor_handles() returned %s" % type(handles))
print("  handles      :", len(handles))
if os.environ.get("AMDSMI_EXPECT_NO_GPU") == "1" and handles:
    raise SystemExit("FAIL: expected 0 processors on a GPU-less host, got %d" % len(handles))
print("  PASS: no-GPU runtime contract holds")
"""


def fail(message):
    sys.exit("FAIL: " + message)


def run(cmd, env=None):
    merged = os.environ.copy()
    if env:
        merged.update(env)
    return subprocess.run(cmd, env=merged, universal_newlines=True)


def resolve_wheel(target):
    """Accept either a wheel file or a directory holding exactly one wheel."""
    if target.is_file():
        return target
    if not target.is_dir():
        fail("no such wheel or directory: {}".format(target))
    wheels = sorted(target.glob("*.whl"))
    if not wheels:
        fail("no *.whl under {}; run tools/build_wheel.py first".format(target))
    if len(wheels) > 1:
        fail("{} holds {} wheels; pass one explicitly with --wheel".format(target, len(wheels)))
    return wheels[0]


def pip_version(python):
    out = subprocess.check_output([python, "-m", "pip", "--version"], universal_newlines=True)
    match = re.search(r"pip\s+(\d+)\.(\d+)", out)
    if not match:
        fail("could not parse pip version from: {}".format(out.strip()))
    return (int(match.group(1)), int(match.group(2)))


def check_pip_supports_wheel(python, wheel):
    """Reject an unusable pip here, where the message can name the real cause."""
    if "manylinux_" not in wheel.name:
        return
    found = pip_version(python)
    if found < MIN_PIP_FOR_PEP600:
        fail(
            "pip {}.{} cannot install {}: PEP 600 manylinux tags need pip {}.{}+. "
            "Run '{} -m pip install --upgrade pip' first.".format(
                found[0], found[1], wheel.name, MIN_PIP_FOR_PEP600[0], MIN_PIP_FOR_PEP600[1], python
            )
        )
    print("  pip {}.{} supports PEP 600 tags".format(found[0], found[1]))


def check_wheel_is_self_contained(wheel):
    with zipfile.ZipFile(str(wheel)) as archive:
        names = archive.namelist()

    if not any(n.endswith("amdsmi/libamd_smi_python.so") for n in names):
        fail("{} ships no amdsmi/libamd_smi_python.so; it is not self-contained".format(wheel.name))
    print("  bundles libamd_smi_python.so")

    # Only a repaired (manylinux) wheel vendors its transitive deps; an
    # unrepaired linux_x86_64 wheel is expected to rely on the build host.
    if "manylinux_" not in wheel.name:
        return
    vendored = [n[len("amdsmi.libs/") :] for n in names if n.startswith("amdsmi.libs/")]
    vendored = [lib for lib in vendored if lib]
    missing = [
        prefix
        for prefix in VENDORED_NETLINK_PREFIXES
        if not any(lib.startswith(prefix) for lib in vendored)
    ]
    if missing:
        fail(
            "{} vendors {} but is missing {}; auditwheel repair did not bundle the "
            "netlink dependencies, so the wheel will not import on a bare container".format(
                wheel.name, sorted(vendored) or "nothing", ", ".join(missing)
            )
        )
    print("  vendors netlink libs: {}".format(", ".join(sorted(vendored))))


def install_wheel(python, wheel):
    """Install into *python* itself, mirroring what a user runs in a container.

    Deliberately not a venv: on CPython 3.6 hosts ``venv`` bootstraps its own
    bundled pip, which is older than the host's and too old for PEP 600 tags,
    so the venv would fail for a reason that has nothing to do with the wheel.
    """
    # --no-index/--no-deps keep this an offline check of the built artifact:
    # a network fetch of a same-named package from PyPI would invalidate it.
    cmd = [python, "-m", "pip", "install", "--no-index", "--no-deps", str(wheel)]
    if run(cmd).returncode == 0:
        return
    # PEP 668 marks distro interpreters externally managed (Ubuntu 24.04 and
    # newer). The override is safe here because the target is a throwaway
    # container, and it keeps the check on the wheel rather than on packaging
    # policy.
    print("  install refused, retrying with --break-system-packages")
    if run(cmd + ["--break-system-packages"]).returncode != 0:
        fail("could not install {}".format(wheel.name))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wheel",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "wheels",
        help="Wheel file, or a directory holding one (default: <project>/wheels).",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Interpreter to install into (default: the running one).",
    )
    parser.add_argument(
        "--expect-no-gpu",
        action="store_true",
        help="Also require zero processors, i.e. assert this host really has no GPU.",
    )
    args = parser.parse_args()

    wheel = resolve_wheel(args.wheel.resolve())
    print("wheel  : {}".format(wheel.name))
    print("python : {}".format(args.python))

    check_wheel_is_self_contained(wheel)
    check_pip_supports_wheel(args.python, wheel)

    install_wheel(args.python, wheel)
    with tempfile.TemporaryDirectory() as workdir:
        # Run from an empty directory so a stray ./amdsmi in the checkout
        # cannot shadow the installed package.
        result = subprocess.run(
            [args.python, "-c", CONTRACT],
            cwd=workdir,
            env=dict(os.environ, AMDSMI_EXPECT_NO_GPU="1" if args.expect_no_gpu else "0"),
            universal_newlines=True,
        )
        if result.returncode != 0:
            fail("no-GPU contract check exited {}".format(result.returncode))

    print("PASS: {} installs and works with no GPU present.".format(wheel.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
