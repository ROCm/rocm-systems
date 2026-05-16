# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import sys
from pathlib import Path


class ProfileModeImportGuard:
    """
    Import guard using sys.meta_path to enforce stdlib-only imports in profile mode.

    Python Version Compatibility:
        - Python 3.10+: Full enforcement (uses sys.stdlib_module_names)
        - Python 3.8-3.9: No-op mode (enforcement disabled, warning issued)

    Usage:
        with ProfileModeImportGuard():
            # Python 3.10+: Import checking active, non-stdlib imports raise ImportError
            # Python 3.8-3.9: No-op, all imports allowed (with warning)

    Context Manager Protocol:
        __enter__: Registers guard with Python's import system (sys.meta_path)
        __exit__: Unregisters guard after code execution completes
    """

    # Project modules that are allowed (non-stdlib)
    ALLOWED_PROJECT_MODULES = frozenset([
        "rocprof_compute",
        "rocprof_compute_profile",
        "rocprof_compute_analyze",
        "rocprof_compute_soc",
        "rocprof_compute_tui",
        "utils",
        "vendored",
        "roofline",
        "config",
        "argparser",  # src/argparser.py, not stdlib argparse
        "rocprof_compute_base",
    ])

    # ROCm system libraries (not pip packages)
    ALLOWED_ROCM_MODULES = frozenset([
        "amdsmi",  # AMD System Management Interface
        "hip",  # HIP runtime Python bindings
        "rocprofv3",  # rocprofv3 python modules such as avail
        "rocprofv3_avail_module",  # Alternative avail module for backward compatibility
    ])

    def __enter__(self):
        """
        Register import guard with Python's import system.

        Called automatically when entering 'with' block.
        Adds this object to sys.meta_path so Python calls our find_spec()
        for every import during the with block.
        """
        if sys.version_info >= (3, 10):
            sys.meta_path.insert(0, self)
        else:
            print(
                "\n" + "=" * 70 + "\n"
                "WARNING: ProfileModeImportGuard requires Python 3.10+\n"
                "(sys.stdlib_module_names unavailable).\n"
                "Import enforcement DISABLED for this test run.\n" + "=" * 70 + "\n",
                file=sys.stderr,
            )
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        """
        Unregister import guard from Python's import system.

        Called automatically when exiting 'with' block.
        Removes this object from sys.meta_path, disabling import checking.
        """
        if sys.version_info >= (3, 10) and self in sys.meta_path:
            sys.meta_path.remove(self)

    def find_spec(self, fullname, path, target=None):
        """
        PEP 451 import hook - called automatically by Python during imports.

        Python's import system calls this method for every import statement.
        We check if the module is allowed, and raise ImportError if not.
        """
        top_level = fullname.split(".")[0]

        # Check stdlib
        if top_level in sys.stdlib_module_names:
            return None

        # Check ROCm modules
        if top_level in self.ALLOWED_ROCM_MODULES:
            return None

        # Check project modules (validate origin to prevent third-party modules
        # with same name, e.g., "utils" from site-packages)
        if top_level in self.ALLOWED_PROJECT_MODULES:
            if self._is_from_project(top_level):
                return None

        # Forbidden module
        raise ImportError(
            f"\n{'=' * 70}\n"
            "PROFILE MODE DEPENDENCY VIOLATION\n"
            f"{'=' * 70}\n"
            f"Forbidden package: {top_level}\n\n"
            "Profile mode must use ONLY Python stdlib + ROCm libraries.\n"
            "Fix: Move import to analyze mode or use stdlib alternative.\n"
            "See CONTRIBUTING.md 'Profile Mode Dependency Policy'\n"
            f"{'=' * 70}\n"
        )

    def _is_from_project(self, module_name):
        """Check if module exists in project directory, not site-packages."""
        project_root = Path(__file__).parent.parent
        for base in [project_root / "src", project_root]:
            # Check for: module.py, module/__init__.py, or module/ (namespace pkg)
            candidates = [
                base / f"{module_name}.py",
                base / module_name / "__init__.py",
                base / module_name,  # namespace package (dir without __init__.py)
            ]
            for p in candidates:
                if p.is_file() or (p.is_dir() and p.exists()):
                    return True
        return False
