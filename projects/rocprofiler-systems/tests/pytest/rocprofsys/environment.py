# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Layered test environment management for rocprofiler-systems tests.

The environment handed to a runner is composed of three layers, ordered from
lowest to highest precedence:

- ``base``: framework defaults for the test type (the ``*_environment``
  presets), NOT inherited verbatim from the shell.
- ``test``: settings from the test itself plus framework-injected values,
  including ``LD_LIBRARY_PATH`` and (for sanitizer builds) ``LD_PRELOAD``.
- ``user``: the inherited shell environment (``os.environ``). The user layer
  wins, so anything exported in the shell overrides the base and test layers.

Separately, a handful of variables are read directly by config discovery
(``RocprofsysConfig``) rather than flowing through these layers:
``ROCPROFSYS_INSTALL_DIR``, ``ROCPROFSYS_BUILD_DIR``, ``ROCM_PATH``, and
``ROCPROFSYS_PYTHON_HINTS``. They can be overridden by the user via the shell,
but they are tightly coupled with config discovery (they locate the install,
build, ROCm, and Python trees).

The following environment variables are excluded from the user layer overrides:
 - ``LD_LIBRARY_PATH``: Shell's value is folded in via ``config.get_library_path()``
 - ``LD_PRELOAD``: Shell's value is folded in via the runner for sanitizer builds
 - ``ROCPROFSYS_OUTPUT_PATH``: Testing and validation depends on this path being
     set by the framework.
 - ``ROCPROFSYS_CONFIG_FILE``: Testing and validation depends on this config file being
     set by the framework.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import RocprofsysConfig

# Environment layers ordered from lowest to highest precedence
ENV_LAYER_ORDER = ("base", "test", "user")


class TestEnvKind(str, Enum):
    """Base-environment preset selected for a test run."""

    NONE = "none"
    BASE = "base"
    BINARY = "binary"
    PYTHON = "python"
    CAUSAL = "causal"


# Fixed selection of fundamental shell variables surfaced in the test-session
# header. This is intentionally small
FUNDAMENTAL_SYSTEM_ENV_KEYS = (
    "PATH",
    "HOME",
    "USER",
    "SHELL",
    "TERM",
    "LANG",
)


@dataclass
class TestEnvironment:
    """Class that encapsulates the three layers of environments used for a single test run"""

    base: dict[str, str] = field(default_factory=dict)
    test: dict[str, str] = field(default_factory=dict)
    user: dict[str, str] = field(default_factory=dict)

    def set_base_environment(
        self,
        config: "RocprofsysConfig",
        test_type: TestEnvKind = TestEnvKind.BASE,
    ) -> None:
        """Set the base layer to the preset for ``test_type`` (default: BASE)."""
        try:
            kind = TestEnvKind(test_type)
        except ValueError as exc:
            valid = ", ".join(k.value for k in TestEnvKind)
            raise ValueError(
                f"Invalid test type: {test_type!r}. Expected one of: {valid}"
            ) from exc

        if kind is TestEnvKind.NONE:
            self.base = {}
        elif kind is TestEnvKind.BASE:
            self.base = base_environment()
        elif kind is TestEnvKind.BINARY:
            self.base = base_binary_environment()
        elif kind is TestEnvKind.PYTHON:
            self.base = base_python_environment(config)
        elif kind is TestEnvKind.CAUSAL:
            self.base = base_causal_environment()

    def set_test_environment(self, test_env: dict[str, str]) -> None:
        self.test.update(test_env)

    def merge(self) -> tuple[dict[str, str], dict[str, str]]:
        """Merge the layers into ``(env, origin)``.

        Layers are applied in precedence order (base -> test -> user), so each
        variable's value comes from, and its origin is, the highest-precedence
        layer that set it.
        """
        merged: dict[str, str] = {}
        origin: dict[str, str] = {}
        for layer in ENV_LAYER_ORDER:
            for key, value in getattr(self, layer).items():
                merged[key] = value
                origin[key] = layer
        return merged, origin

    def get_merged_environment(self, config: "RocprofsysConfig") -> dict[str, str]:
        """Return the effective merged environment (highest-precedence layer wins).

        ROCm-derived defaults (:func:`derived_defaults`) are applied at the
        lowest precedence, so any base/test/user value overrides them.
        """
        merged = self.merge()[0]
        for key, value in derived_defaults(config).items():
            merged.setdefault(key, value)
        return merged

    def format_layers(self) -> list[str]:
        """Format the environment grouped by ``[base]``, ``[test]``, ``[user]``.

        Each variable is shown only once, under the highest-precedence layer
        that sets it, so an entry overridden by a higher layer is not repeated
        lower down. The ``[user]`` layer is limited to variables that explicitly
        override a ``[base]`` or ``[test]`` setting; other purely shell-inherited
        values are omitted to keep the per-test output concise. Layers with no
        entries to show are omitted entirely.
        """
        sections = {
            "base": [k for k in self.base if k not in self.test and k not in self.user],
            "test": [k for k in self.test if k not in self.user],
            "user": [k for k in self.user if k in self.base or k in self.test],
        }
        lines: list[str] = []
        for layer in ENV_LAYER_ORDER:
            keys = sorted(sections[layer])
            if not keys:
                continue
            lines.append(f"[{layer}]")
            for key in keys:
                lines.append(self._format_entry(layer, key))
        return lines

    def _format_entry(self, layer: str, key: str) -> str:
        """Format one ``  key=value`` line for ``format_layers``.

        For user entries that also exist in ``base``/``test``, append a note
        showing the framework value the shell value replaced, e.g.
        ``ROCPROFSYS_TRACE=OFF  (overrides test=ON)``.
        """
        value = getattr(self, layer)[key]
        if layer == "user":
            if key in self.test:
                return f"  {key}={value}  (overrides test={self.test[key]})"
            if key in self.base:
                return f"  {key}={value}  (overrides base={self.base[key]})"
        return f"  {key}={value}"

    def set_user_environment(self) -> None:
        """Capture the invoking shell environment as the user layer.

        Certain environment variables are excluded (see
        the module docstring).
        """
        owned = (
            "LD_LIBRARY_PATH",
            "LD_PRELOAD",
            "ROCPROFSYS_OUTPUT_PATH",
            "ROCPROFSYS_CONFIG_FILE",
        )
        self.user.update({k: v for k, v in os.environ.items() if k not in owned})


def derived_defaults(config: "RocprofsysConfig") -> dict[str, str]:
    """ROCm-derived defaults, applied at the lowest precedence by
    :meth:`TestEnvironment.get_merged_environment` (any layer overrides them).
    Each entry is only produced when its derived target exists on disk.
    """
    defaults: dict[str, str] = {}

    # LIBVA_DRIVERS_PATH: fall back to ROCm's bundled VA drivers.
    if config.rocm_path:
        sysdeps = (config.rocm_path / "lib" / "rocm_sysdeps" / "lib").resolve()
        if sysdeps.is_dir():
            defaults["LIBVA_DRIVERS_PATH"] = str(sysdeps)

    return defaults


# Settings shared by every ``base_*`` environment preset.
COMMON_BASE_DEFAULT_VARS = {
    "ROCPROFSYS_CI": "ON",
    "ROCPROFSYS_CI_TIMEOUT": "300",
    "ROCPROFSYS_USE_PID": "OFF",
    "ROCPROFSYS_CONFIG_FILE": "",
}

# OpenMP thread-pinning shared by presets that exercise OpenMP workloads.
OMP_DEFAULT_VARS = {
    "OMP_PROC_BIND": "spread",
    "OMP_PLACES": "threads",
    "OMP_NUM_THREADS": "2",
}


def base_environment() -> dict[str, str]:
    """Framework default environment for instrumented test execution."""
    return {
        **COMMON_BASE_DEFAULT_VARS,
        **OMP_DEFAULT_VARS,
        "ROCPROFSYS_DEFAULT_MIN_INSTRUCTIONS": "64",
        "ROCPROFSYS_TRACE": "ON",
        "ROCPROFSYS_PROFILE": "ON",
        "ROCPROFSYS_USE_SAMPLING": "ON",
        "ROCPROFSYS_USE_PROCESS_SAMPLING": "ON",
        "ROCPROFSYS_TIME_OUTPUT": "OFF",
        "ROCPROFSYS_FILE_OUTPUT": "ON",
        "ROCPROFSYS_LOG_LEVEL": "info",
        "ROCPROFSYS_SAMPLING_FREQ": "300",
        "ROCPROFSYS_SAMPLING_DELAY": "0.05",
        "ROCPROFSYS_SAMPLING_GPUS": "all",
    }


def base_binary_environment() -> dict[str, str]:
    """Framework default environment for rocprof-sys binary test execution."""
    return {
        **COMMON_BASE_DEFAULT_VARS,
        "ROCPROFSYS_TRACE": "ON",
        "ROCPROFSYS_PROFILE": "ON",
        "ROCPROFSYS_USE_SAMPLING": "ON",
        "ROCPROFSYS_TIME_OUTPUT": "OFF",
        "ROCPROFSYS_LOG_LEVEL": "info",
    }


def base_python_environment(config: "RocprofsysConfig") -> dict[str, str]:
    """Framework default environment for Python test execution."""
    return {
        **COMMON_BASE_DEFAULT_VARS,
        "ROCPROFSYS_TRACE": "ON",
        "ROCPROFSYS_PROFILE": "ON",
        "ROCPROFSYS_USE_SAMPLING": "OFF",
        "ROCPROFSYS_USE_PROCESS_SAMPLING": "ON",
        "ROCPROFSYS_TIME_OUTPUT": "OFF",
        "ROCPROFSYS_TREE_OUTPUT": "OFF",
        "ROCPROFSYS_TIMEMORY_COMPONENTS": "wall_clock,trip_count",
        "PYTHONPATH": (
            str(config.rocprofsys_site_packages)
            if config.rocprofsys_site_packages
            else ""
        ),
    }


def base_causal_environment() -> dict[str, str]:
    """Framework default environment for causal profiling test execution."""
    return {
        **COMMON_BASE_DEFAULT_VARS,
        "ROCPROFSYS_THREAD_POOL_SIZE": "0",
        "ROCPROFSYS_VERBOSE": "1",
        "ROCPROFSYS_LOG_LEVEL": "info",
        "ROCPROFSYS_DL_VERBOSE": "0",
        "ROCPROFSYS_DEBUG_SETTINGS": "0",
    }


def flat_environment() -> dict[str, str]:
    """Environment for flat-profile tests."""
    return {
        **OMP_DEFAULT_VARS,
        "ROCPROFSYS_TRACE": "ON",
        "ROCPROFSYS_PROFILE": "ON",
        "ROCPROFSYS_TIME_OUTPUT": "OFF",
        "ROCPROFSYS_COUT_OUTPUT": "ON",
        "ROCPROFSYS_FLAT_PROFILE": "ON",
        "ROCPROFSYS_TIMELINE_PROFILE": "OFF",
        "ROCPROFSYS_COLLAPSE_PROCESSES": "ON",
        "ROCPROFSYS_COLLAPSE_THREADS": "ON",
        "ROCPROFSYS_SAMPLING_FREQ": "50",
        "ROCPROFSYS_TIMEMORY_COMPONENTS": "wall_clock,trip_count",
    }


def lock_environment() -> dict[str, str]:
    """Environment for thread-lock tracing tests."""
    return {
        "ROCPROFSYS_USE_SAMPLING": "ON",
        "ROCPROFSYS_USE_PROCESS_SAMPLING": "OFF",
        "ROCPROFSYS_SAMPLING_FREQ": "750",
        "ROCPROFSYS_COLLAPSE_THREADS": "ON",
        "ROCPROFSYS_TRACE_THREAD_LOCKS": "ON",
        "ROCPROFSYS_TRACE_THREAD_SPIN_LOCKS": "ON",
        "ROCPROFSYS_TRACE_THREAD_RW_LOCKS": "ON",
        "ROCPROFSYS_COUT_OUTPUT": "ON",
        "ROCPROFSYS_TIME_OUTPUT": "OFF",
        "ROCPROFSYS_TIMELINE_PROFILE": "OFF",
        "ROCPROFSYS_LOG_LEVEL": "info",
    }


def perfetto_environment() -> dict[str, str]:
    """Environment for perfetto-only tests."""
    return {
        **OMP_DEFAULT_VARS,
        "ROCPROFSYS_TRACE": "ON",
        "ROCPROFSYS_PROFILE": "OFF",
        "ROCPROFSYS_USE_SAMPLING": "ON",
        "ROCPROFSYS_USE_PROCESS_SAMPLING": "ON",
        "ROCPROFSYS_TIME_OUTPUT": "OFF",
        "ROCPROFSYS_PERFETTO_BACKEND": "inprocess",
        "ROCPROFSYS_PERFETTO_FILL_POLICY": "ring_buffer",
    }


def timemory_environment() -> dict[str, str]:
    """Environment for timemory-only tests."""
    return {
        **OMP_DEFAULT_VARS,
        "ROCPROFSYS_TRACE": "OFF",
        "ROCPROFSYS_PROFILE": "ON",
        "ROCPROFSYS_USE_SAMPLING": "ON",
        "ROCPROFSYS_USE_PROCESS_SAMPLING": "ON",
        "ROCPROFSYS_TIME_OUTPUT": "OFF",
        "ROCPROFSYS_TIMEMORY_COMPONENTS": "wall_clock,trip_count,peak_rss",
    }


def fundamental_system_environment() -> dict[str, str]:
    """Return the curated selection of shell env vars shown in the session header.

    Keys are returned in :data:`FUNDAMENTAL_SYSTEM_ENV_KEYS` order; missing
    variables map to an empty string.
    """
    return {key: os.environ.get(key, "") for key in FUNDAMENTAL_SYSTEM_ENV_KEYS}
