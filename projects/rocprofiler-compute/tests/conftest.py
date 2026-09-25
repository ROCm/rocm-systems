# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import builtins
import os
import random
import sys
from pathlib import Path


class ProfileModeImportGuard:
    """
    Import guard enforcing stdlib-only imports in profile mode.

    Full enforcement on Python 3.10+ (uses sys.stdlib_module_names); no-op with
    a warning on 3.8-3.9.
    """

    _real_import = None

    # Project modules that are allowed (non-stdlib)
    ALLOWED_PROJECT_MODULES = frozenset([
        "rocprof_compute",
        "rocprof_compute_profile",
        "rocprof_compute_analyze",
        "rocprof_compute_soc",
        "rocprof_compute_tui",
        "pc_sampling",
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
        "rocprofv3_avail_module",  # Alternative avail module for
        # backward compatibility
    ])

    def __enter__(self):
        """Install both import hooks (Python 3.10+ only)."""
        if sys.version_info >= (3, 10):
            sys.meta_path.insert(0, self)
            self._real_import = builtins.__import__
            builtins.__import__ = self._guarded_import
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
        """Restore builtins.__import__ and remove the meta_path finder."""
        if sys.version_info >= (3, 10):
            if self._real_import is not None:
                builtins.__import__ = self._real_import
                self._real_import = None
            if self in sys.meta_path:
                sys.meta_path.remove(self)

    def _guarded_import(self, name, *args, **kwargs):
        """Hook 1: builtins.__import__ wrapper; catches cached imports."""
        # Relative imports (level > 0) resolve within their already-checked
        # parent; hook 2 catches the resolved submodule, so only check absolute.
        level = args[3] if len(args) > 3 else kwargs.get("level", 0)
        if level == 0:
            self._raise_if_forbidden(name)
        return self._real_import(name, *args, **kwargs)

    def find_spec(self, fullname, path, target=None):
        """Hook 2: meta_path finder; catches dynamic uncached imports."""
        self._raise_if_forbidden(fullname)
        return None

    def _raise_if_forbidden(self, fullname):
        """Raise ImportError if the top-level package is not allowed."""
        top_level = fullname.split(".")[0]

        # Check stdlib
        if top_level in sys.stdlib_module_names:
            return

        # Check ROCm modules
        if top_level in self.ALLOWED_ROCM_MODULES:
            return

        # Check project modules (validate origin to prevent third-party modules
        # with same name, e.g., "utils" from site-packages)
        if top_level in self.ALLOWED_PROJECT_MODULES:
            if self._is_from_project(top_level):
                return

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


def pytest_addoption(parser):
    parser.addoption(
        "--rocprofiler-sdk-tool-path",
        type=str,
        default=str(
            Path(os.getenv("ROCM_PATH", "/opt/rocm"))
            / "lib/rocprofiler-sdk/librocprofiler-sdk-tool.so"
        ),
        help="Path to the rocprofiler-sdk tool",
    )

    parser.addoption(
        "--coverage-seed",
        type=int,
        default=random.randrange(2**32),
        help="RNG seed for test_torch_trace_coverage sampling.",
    )
    parser.addoption(
        "--coverage-n",
        type=int,
        default=20,
        help=(
            "Sample budget for test_torch_trace_coverage (default 20). "
            "Structural operators are always included. Use 100 for a nightly run."
        ),
    )
