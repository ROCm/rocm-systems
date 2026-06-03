# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Root-level pytest configuration.

This file is intentionally minimal.  Its primary purpose is to inject the
system-installed PyYAML package into the ``vendored`` namespace **before**
``tests/conftest.py`` tries to load the ``src/rocprof-compute`` entry-point.

In a full build environment the CMake build step copies a pure-Python PyYAML
into ``src/vendored/pyyaml/lib/``.  When running tests from a sparse-checkout
or development tree without a prior CMake build that artefact is missing, so
we fall back to the system-installed ``yaml`` package which is API-compatible.
"""

import sys
import types


def _patch_vendored_yaml() -> None:
    """Inject system yaml into ``sys.modules['vendored']`` if not already done
    and if the built pyyaml artefact is absent."""
    if "vendored" in sys.modules:
        return  # already patched or already loaded correctly
    try:
        # Try the normal vendored import first (works in installed environment)
        import vendored  # noqa: F401
    except ImportError:
        # Fall back to system yaml (development / sparse-checkout environment)
        try:
            import yaml as _yaml
        except ImportError:
            # If system yaml is also missing there is nothing we can do;
            # individual tests will fail with a meaningful error.
            return
        vendor_mod = types.ModuleType("vendored")
        vendor_mod.yaml = _yaml
        sys.modules["vendored"] = vendor_mod


_patch_vendored_yaml()
