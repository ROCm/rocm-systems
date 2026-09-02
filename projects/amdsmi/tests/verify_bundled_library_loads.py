#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""verify_bundled_library_loads.py
====================================

Prove that an installed amdsmi wheel actually loaded its bundled shared
object, not merely that ``import amdsmi`` succeeded.

py-interface/amdsmi_wrapper.py imports tolerantly: when the ``.so`` cannot be
loaded (missing auditwheel dependency, wrong RPATH, otherwise unloadable) it
catches the ``OSError``, installs a ``_MissingLibrary`` sentinel, and leaves
``_loaded_lib_path`` unset so docs / lint imports still work without a runtime
library. That tolerance means a plain import smoke test can pass on a wheel
whose library never loaded.

Run this against the exact final (repaired) wheel after installing it into a
clean environment. It asserts the bundled ``libamd_smi_python.so`` loaded from
inside the installed package, that no sentinel is in place, and that a
no-GPU-safe exported symbol is callable.
"""

import ctypes
import os
import sys


def _fail(message: str) -> None:
    sys.exit(f"bundled-library smoke check failed: {message}")


def main() -> None:
    import amdsmi
    from amdsmi import amdsmi_wrapper as wrapper

    package_dir = os.path.dirname(os.path.realpath(amdsmi.__file__))

    # A tolerant import installs a _MissingLibrary sentinel (and leaves
    # _loaded_lib_path unset) when the shared object cannot be loaded, so a
    # bare "import amdsmi" does not prove the library is usable.
    library = wrapper._libraries.get("libamd_smi.so")
    if isinstance(library, wrapper._MissingLibrary):
        _fail("amdsmi imported but its shared library did not load (sentinel installed)")
    if not isinstance(library, ctypes.CDLL):
        _fail(f"unexpected library object type: {type(library).__name__}")

    loaded_path = wrapper._loaded_lib_path
    if not loaded_path:
        _fail("no shared-library path was recorded")
    resolved = os.path.realpath(loaded_path)
    if os.path.basename(resolved) != "libamd_smi_python.so":
        _fail(f"loaded {resolved}, expected the bundled libamd_smi_python.so")
    if not resolved.startswith(package_dir + os.sep):
        _fail(f"loaded {resolved} from outside the installed package {package_dir}")

    # A no-GPU-safe exported call proves the library is callable, not just
    # mapped: amdsmi_get_lib_version needs no amdsmi_init and touches no device.
    try:
        version = amdsmi.amdsmi_get_lib_version()
    except Exception as exc:
        _fail(f"amdsmi_get_lib_version raised {type(exc).__name__}: {exc}")
    for field in ("major", "minor", "release"):
        if field not in version:
            _fail(f"amdsmi_get_lib_version result missing '{field}': {version}")

    print(
        f"bundled library OK: {resolved} "
        f"(version {version['major']}.{version['minor']}.{version['release']})"
    )


if __name__ == "__main__":
    main()
