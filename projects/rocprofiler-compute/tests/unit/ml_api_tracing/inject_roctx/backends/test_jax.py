# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for JAX backend discovery and install."""

import importlib
import sys
import types
from typing import List

from ml_api_tracing.inject_roctx.backends import jax as jax_backend


class FindSpecWithout:
    """Behave like find_spec but report one name as missing."""

    def __init__(self, absent_name, real_find_spec):
        self.absent_name = absent_name
        self.real_find_spec = real_find_spec

    def __call__(self, name, *args, **kwargs):
        if name == self.absent_name:
            return None
        return self.real_find_spec(name, *args, **kwargs)


def reset_jax_core_state():
    jax_backend._CoreState = jax_backend._JaxState()


def test_jax_backend_skips_when_jax_missing(monkeypatch):
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        FindSpecWithout("jax", real_find_spec),
    )

    warnings: List[tuple] = []
    monkeypatch.setattr(jax_backend, "console_warning", lambda *a: warnings.append(a))

    jax_backend.JaxBackend().install()
    assert any(
        "JAX is not installed" in str(args[1]) for args in warnings if len(args) > 1
    )


def test_jax_backend_warns_when_import_fails(monkeypatch):
    warnings: List[tuple] = []
    monkeypatch.setattr(jax_backend, "console_warning", lambda *a: warnings.append(a))
    monkeypatch.setattr(
        jax_backend.importlib.util,
        "find_spec",
        lambda name, package=None: types.SimpleNamespace() if name == "jax" else None,
    )

    def fail_jax_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "jax" or name.startswith("jax."):
            raise ImportError("broken jax")
        return importlib.__import__(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fail_jax_import)
    jax_backend.JaxBackend().install()
    assert any(
        "Could not import jax" in str(args[1]) for args in warnings if len(args) > 1
    )


def test_jax_backend_skips_when_python_tier_unavailable(monkeypatch):
    monkeypatch.setattr(jax_backend, "_resolve_jax", lambda: True)
    monkeypatch.setattr(jax_backend.marker_ranges, "ensure_python_tier", lambda: False)

    patched: List[bool] = []
    monkeypatch.setattr(
        jax_backend, "patch_jax_entry_points", lambda: patched.append(True)
    )
    warnings: List[tuple] = []
    monkeypatch.setattr(jax_backend, "console_warning", lambda *a: warnings.append(a))

    jax_backend.JaxBackend().install()
    assert patched == []
    assert any("ROCTX bindings not found" in str(a[1]) for a in warnings if len(a) > 1)


def test_jax_backend_registers_framework_root(monkeypatch):
    reset_jax_core_state()
    monkeypatch.setattr(jax_backend, "_resolve_jax", lambda: True)
    monkeypatch.setattr(jax_backend.marker_ranges, "ensure_python_tier", lambda: True)
    monkeypatch.setattr(jax_backend, "patch_jax_entry_points", lambda: None)

    fake_jax = types.ModuleType("jax")
    fake_jax.__file__ = "/opt/fake/jax/__init__.py"
    monkeypatch.setitem(sys.modules, "jax", fake_jax)

    roots: List[str] = []
    monkeypatch.setattr(
        jax_backend.marker_ranges, "add_framework_root", lambda p: roots.append(p)
    )

    jax_backend.JaxBackend().install()
    assert roots == ["/opt/fake/jax"]
