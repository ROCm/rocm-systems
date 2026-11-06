# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for the JAX inject_roctx backend."""

import sys
from types import ModuleType, SimpleNamespace

from utils.inject_roctx import core
from utils.inject_roctx._backends import jax as jax_backend


def fake_compiled_type():
    class FakeCompiled:
        def __init__(self, name, body=None):
            self.__name__ = name
            self.body = body

        def __call__(self, *args, **kwargs):
            if self.body is not None:
                return self.body(*args, **kwargs)
            return None

    return FakeCompiled


def fake_primitive_type():
    class FakePrimitive:
        def __init__(self, name):
            self.name = name

        def bind_with_trace(self, *args, **kwargs):
            return None

    return FakePrimitive


def install_fake_jax(monkeypatch, jax_file="/tmp/fake_jax/__init__.py"):
    compiled_type = fake_compiled_type()
    primitive_type = fake_primitive_type()

    def fake_jit(fun, *args, **kwargs):
        name = getattr(fun, "__name__", "<unknown>")
        return compiled_type(name, body=fun)

    def fake_pmap(fun, *args, **kwargs):
        name = getattr(fun, "__name__", "<unknown>")

        def mapped(*mapped_args, **mapped_kwargs):
            return fun(*mapped_args, **mapped_kwargs)

        mapped.__name__ = name
        return mapped

    jax_mod = ModuleType("jax")
    jax_mod.__file__ = jax_file
    jax_mod.jit = fake_jit
    jax_mod.pmap = fake_pmap
    extend = ModuleType("jax.extend")
    extend_core = ModuleType("jax.extend.core")
    extend_core.Primitive = primitive_type
    monkeypatch.setitem(sys.modules, "jax", jax_mod)
    monkeypatch.setitem(sys.modules, "jax.extend", extend)
    monkeypatch.setitem(sys.modules, "jax.extend.core", extend_core)
    monkeypatch.setattr(
        jax_backend.importlib.util,
        "find_spec",
        lambda name, package=None: SimpleNamespace() if name == "jax" else None,
    )
    return jax_mod, compiled_type, primitive_type


def reset_jax_backend_state():
    jax_backend._STATE = jax_backend._JaxState()
    jax_backend._jax_thread_local.depth = 0


def install_jax_backend(
    monkeypatch, python_tier=True, jax_file="/tmp/fake_jax/__init__.py"
):
    reset_jax_backend_state()
    jax_mod, compiled_type, primitive_type = install_fake_jax(
        monkeypatch, jax_file=jax_file
    )
    monkeypatch.setattr(core, "ensure_python_tier", lambda: python_tier)
    monkeypatch.setattr(core, "add_framework_root", lambda path: None)
    jax_backend.JaxBackend().install()
    return jax_mod, compiled_type, primitive_type


def record_scopes(monkeypatch):
    pushes = []

    def push(marker, location, backend="", args="n/a"):
        pushes.append({"marker": marker, "backend": backend, "location": location})

    monkeypatch.setattr(core, "_push_scope", push)
    monkeypatch.setattr(core, "_pop_scope", lambda: None)
    return pushes


def test_jax_backend_skips_when_jax_missing(monkeypatch):
    warnings = []

    def capture_warning(prefix, message=""):
        warnings.append((prefix, message))

    monkeypatch.setattr(
        jax_backend.importlib.util, "find_spec", lambda name, package=None: None
    )
    monkeypatch.setattr(jax_backend, "console_warning", capture_warning)
    jax_backend.JaxBackend().install()
    assert warnings
    assert warnings[0][0] == "ml api trace"
    assert "JAX is not installed" in warnings[0][1]


def test_jax_backend_skips_when_python_tier_unavailable(monkeypatch):
    warnings = []
    wraps = []

    def capture_warning(prefix, message=""):
        warnings.append((prefix, message))

    reset_jax_backend_state()
    install_fake_jax(monkeypatch)
    monkeypatch.setattr(core, "ensure_python_tier", lambda: False)
    monkeypatch.setattr(jax_backend, "console_warning", capture_warning)
    monkeypatch.setattr(
        jax_backend, "patch_jax_entry_points", lambda: wraps.append(True)
    )
    jax_backend.JaxBackend().install()
    assert wraps == []
    assert warnings
    assert warnings[0][0] == "ml api trace"
    assert "ROCTX bindings not found" in warnings[0][1]


def test_jax_jit_call_emits_jit_marker(monkeypatch):
    jax_mod, compiled_type, _primitive_type = install_jax_backend(monkeypatch)
    pushes = record_scopes(monkeypatch)

    def train_step(value):
        return value

    compiled = jax_mod.jit(train_step)
    assert isinstance(compiled, compiled_type)
    assert compiled(1) == 1
    assert [item["marker"] for item in pushes] == ["jax.jit.train_step"]
    assert pushes[0]["backend"] == "jax"


def test_jax_pmap_call_emits_pmap_marker(monkeypatch):
    jax_mod, _compiled_type, _primitive_type = install_jax_backend(monkeypatch)
    pushes = record_scopes(monkeypatch)

    def pmap_square(value):
        return value

    mapped = jax_mod.pmap(pmap_square)
    assert mapped(1) == 1
    assert [item["marker"] for item in pushes] == ["jax.pmap.pmap_square"]
    assert pushes[0]["backend"] == "jax"
