# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Layered test environment management for rocprofiler-systems tests.

The environment handed to a runner is composed of three layers, ordered from
lowest to highest precedence:

- ``base``: framework defaults for the test type (the ``*_environment``
  presets). These are NOT inherited verbatim from the shell.
- ``test``: settings supplied by the test itself (and per-test markers),
  plus framework-injected per-test values such as the CI timeout/monochrome
  flags and ``LD_LIBRARY_PATH`` (a test may set its own, otherwise it defaults
  to the computed ``config.library_path``: rocprofsys libs + shell
  ``LD_LIBRARY_PATH`` + appended ROCm LLVM libs).
- ``user``: the environment inherited from the invoking shell (the full
  ``os.environ``, minus ``LD_LIBRARY_PATH`` which is owned by the test layer).
  The user layer wins, so anything exported in the shell overrides the base and
  test layers.

Separately, a handful of variables are read directly by config discovery
(``RocprofsysConfig``) rather than flowing through these layers:
``ROCPROFSYS_INSTALL_DIR``, ``ROCPROFSYS_BUILD_DIR``, ``ROCM_PATH``, and
``ROCPROFSYS_PYTHON_HINTS``. They can be overridden by the user via the shell,
but they are tightly coupled with config discovery (they locate the install,
build, ROCm, and Python trees).
"""

from __future__ import annotations
from dataclasses import dataclass, field
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import RocprofsysConfig

# Environment layers ordered from lowest to highest precedence
ENV_LAYER_ORDER = ("base", "test", "user")

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
        self, config: "RocprofsysConfig", test_type: str = "base"
    ) -> None:
        """Get the base environment for a given test type (default: "base")

        Accepted test types:
         - "base" (default)
         - "binary"
         - "python"
         - "causal"
         - "none"
        """

        if test_type == "none":
            self.base = {}
        elif test_type == "base":
            self.base = base_environment()
        elif test_type == "binary":
            self.base = base_binary_environment()
        elif test_type == "python":
            self.base = base_python_environment(config)
        elif test_type == "causal":
            self.base = base_causal_environment()
        else:
            raise ValueError(f"Invalid test type: {test_type}")

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

    def get_merged_environment(self) -> dict[str, str]:
        """Return the effective merged environment (highest-precedence layer wins)."""
        return self.merge()[0]

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

        LD_LIBRARY_PATH is intentionally left out: the shell's value is already
        folded into config.library_path, and LD_LIBRARY_PATH is owned by the
        test layer (a default is injected there, and certain tests, e.g. Julia,
        append their own paths).
        """
        self.user.update({k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"})


def base_environment() -> dict[str, str]:
    """Framework default environment for instrumented test execution."""
    return {
        "ROCPROFSYS_DEFAULT_MIN_INSTRUCTIONS": "64",
        "ROCPROFSYS_CI": "ON",
        "ROCPROFSYS_CI_TIMEOUT": "300",
        "ROCPROFSYS_CONFIG_FILE": "",
        "ROCPROFSYS_TRACE": "ON",
        "ROCPROFSYS_PROFILE": "ON",
        "ROCPROFSYS_USE_SAMPLING": "ON",
        "ROCPROFSYS_USE_PROCESS_SAMPLING": "ON",
        "ROCPROFSYS_TIME_OUTPUT": "OFF",
        "ROCPROFSYS_FILE_OUTPUT": "ON",
        "ROCPROFSYS_USE_PID": "OFF",
        "ROCPROFSYS_LOG_LEVEL": "info",
        "ROCPROFSYS_SAMPLING_FREQ": "300",
        "ROCPROFSYS_SAMPLING_DELAY": "0.05",
        "ROCPROFSYS_SAMPLING_GPUS": "all",
        "OMP_PROC_BIND": "spread",
        "OMP_PLACES": "threads",
        "OMP_NUM_THREADS": "2",
    }


def base_binary_environment() -> dict[str, str]:
    """Framework default environment for rocprof-sys binary test execution."""
    return {
        "ROCPROFSYS_CI": "ON",
        "ROCPROFSYS_CI_TIMEOUT": "300",
        "ROCPROFSYS_TRACE": "ON",
        "ROCPROFSYS_PROFILE": "ON",
        "ROCPROFSYS_USE_SAMPLING": "ON",
        "ROCPROFSYS_TIME_OUTPUT": "OFF",
        "ROCPROFSYS_USE_PID": "OFF",
        "ROCPROFSYS_LOG_LEVEL": "info",
        "ROCPROFSYS_CONFIG_FILE": "",
    }


def base_python_environment(config: "RocprofsysConfig") -> dict[str, str]:
    """Framework default environment for Python test execution."""
    return {
        "ROCPROFSYS_CI": "ON",
        "ROCPROFSYS_CI_TIMEOUT": "300",
        "ROCPROFSYS_TRACE": "ON",
        "ROCPROFSYS_PROFILE": "ON",
        "ROCPROFSYS_USE_SAMPLING": "OFF",
        "ROCPROFSYS_USE_PROCESS_SAMPLING": "ON",
        "ROCPROFSYS_TIME_OUTPUT": "OFF",
        "ROCPROFSYS_TREE_OUTPUT": "OFF",
        "ROCPROFSYS_USE_PID": "OFF",
        "ROCPROFSYS_TIMEMORY_COMPONENTS": "wall_clock,trip_count",
        "PYTHONPATH": (
            str(config.rocprofsys_site_packages)
            if config.rocprofsys_site_packages
            else ""
        ),
        "ROCPROFSYS_CONFIG_FILE": "",
    }


def base_causal_environment() -> dict[str, str]:
    """Framework default environment for causal profiling test execution."""
    return {
        "ROCPROFSYS_CI": "ON",
        "ROCPROFSYS_CI_TIMEOUT": "300",
        "ROCPROFSYS_USE_PID": "OFF",
        "ROCPROFSYS_THREAD_POOL_SIZE": "0",
        "ROCPROFSYS_VERBOSE": "1",
        "ROCPROFSYS_LOG_LEVEL": "info",
        "ROCPROFSYS_DL_VERBOSE": "0",
        "ROCPROFSYS_DEBUG_SETTINGS": "0",
        "ROCPROFSYS_CONFIG_FILE": "",
    }


def fundamental_system_environment() -> dict[str, str]:
    """Return the curated selection of shell env vars shown in the session header.

    Keys are returned in :data:`FUNDAMENTAL_SYSTEM_ENV_KEYS` order; missing
    variables map to an empty string.
    """
    return {key: os.environ.get(key, "") for key in FUNDAMENTAL_SYSTEM_ENV_KEYS}
