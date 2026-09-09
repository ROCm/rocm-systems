#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
run_amdsmi_python_versions_test.py
===========================
The amdsmi loader and wrapper must behave identically across the range of
CPython versions AMD SMI supports (3.6.8 through the latest release). This
harness runs the version-independent loader contract tests under each
interpreter it is given, so a change that only breaks on, say, 3.6 or a newer
release is caught.

For each interpreter it:
  * confirms the wrapper imports and the loader resolves a path (or degrades to
    the _MissingLibrary sentinel without raising at import), and
  * runs the ABI-compat and dual-copy guard unit tests under that interpreter.

No GPU is required: the loader tests use a fake ctypes.CDLL.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PY_INTERFACE = REPO_ROOT / "py-interface"
TEST_DIR = REPO_ROOT / "tests" / "python"


# Highest CPython minor AMD SMI claims support for; keep in sync with the
# matrix in .github/workflows/amdsmi-python-versions.yml.
NEWEST_SUPPORTED_MINOR = 14


def _discover_default_interpreters() -> list:
    found = []
    for minor in range(6, NEWEST_SUPPORTED_MINOR + 1):
        exe = shutil.which(f"python3.{minor}")
        if exe:
            found.append(exe)
    if not found:
        found.append(sys.executable)
    return found


def _import_smoke(interp: str) -> None:
    # Import the wrapper and print the resolved library path. Import must not
    # raise even when no library is present (the _MissingLibrary sentinel keeps
    # it tolerant); a raised exception here is a real regression.
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "import amdsmi_wrapper as w\n"
        "print('  loaded:', getattr(w, '_loaded_lib_path', None))\n"
    ) % str(PY_INTERFACE)
    subprocess.run([interp, "-c", code], check=True)


def _run_unit_tests(interp: str) -> None:
    for test in (
        "test_abi_compat.py",
        "test_dual_copy_guard.py",
        "test_cpack_path_guard.py",
        "test_upgrade_downgrade_guard.py",
        "test_packaging_scriptlets.py",
    ):
        subprocess.run([interp, str(TEST_DIR / test)], check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--interpreters",
        nargs="*",
        default=None,
        help="Python interpreters to test (default: discovered python3.6..3.14).",
    )
    args = parser.parse_args()
    interpreters = args.interpreters or _discover_default_interpreters()

    failures = []
    for interp in interpreters:
        ver = subprocess.run(
            [interp, "-c", "import sys; print('%d.%d.%d' % sys.version_info[:3])"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        ).stdout.strip()
        print(f"=== {interp} (Python {ver}) ===")
        try:
            _import_smoke(interp)
            _run_unit_tests(interp)
            print(f"PASS: {interp}")
        except subprocess.CalledProcessError as exc:
            print(f"FAIL: {interp} exited {exc.returncode}")
            failures.append(f"{interp} (Python {ver})")

    if failures:
        sys.exit("python-version tests failed for: " + ", ".join(failures))
    print("PASS: all interpreters behaved identically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
