# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for JAX backend discovery and install."""

import importlib
import sys
import types
from typing import List

import pytest

from ml_api_tracing.inject_roctx import marker_ranges
from ml_api_tracing.inject_roctx.backends import jax as jax_backend

FAKE_JAX_PACKAGE_INIT = "/opt/fake/jax/__init__.py"


class FindSpecWithout:
    """Behave like find_spec but report one name as missing."""

    def __init__(self, absent_name, real_find_spec):
        self.absent_name = absent_name
        self.real_find_spec = real_find_spec

    def __call__(self, name, *args, **kwargs):
        if name == self.absent_name:
            return None
        return self.real_find_spec(name, *args, **kwargs)


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


def install_fake_jax(monkeypatch, jax_package_init=FAKE_JAX_PACKAGE_INIT):
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
        mapped.lower = lambda *a, **k: "lowered"
        return mapped

    def fake_shard_map(fun=None, /, **kwargs):
        if fun is None:
            return lambda user_fun: fake_shard_map(user_fun, **kwargs)
        name = getattr(fun, "__name__", "<unknown>")

        def mapped(*mapped_args, **mapped_kwargs):
            return fun(*mapped_args, **mapped_kwargs)

        mapped.__name__ = name
        return mapped

    jax_mod = types.ModuleType("jax")
    jax_mod.__file__ = jax_package_init
    jax_mod.jit = fake_jit
    jax_mod.pmap = fake_pmap
    jax_mod.shard_map = fake_shard_map
    extend = types.ModuleType("jax.extend")
    extend_core = types.ModuleType("jax.extend.core")
    extend_core.Primitive = primitive_type
    monkeypatch.setitem(sys.modules, "jax", jax_mod)
    monkeypatch.setitem(sys.modules, "jax.extend", extend)
    monkeypatch.setitem(sys.modules, "jax.extend.core", extend_core)
    monkeypatch.setattr(
        jax_backend.importlib.util,
        "find_spec",
        lambda name, package=None: types.SimpleNamespace() if name == "jax" else None,
    )
    return jax_mod, compiled_type, primitive_type


def reset_jax_core_state():
    jax_backend._CoreState = jax_backend._JaxState()
    jax_backend._jax_thread_local.depth = 0


def install_jax_backend(monkeypatch, python_tier=True):
    reset_jax_core_state()
    jax_mod, compiled_type, primitive_type = install_fake_jax(monkeypatch)
    monkeypatch.setattr(marker_ranges, "ensure_python_tier", lambda: python_tier)
    monkeypatch.setattr(marker_ranges, "add_framework_root", lambda path: None)
    jax_backend.JaxBackend().install()
    return jax_mod, compiled_type, primitive_type


def record_scopes(monkeypatch):
    pushes = []

    def push(marker, location, backend="", args="n/a"):
        pushes.append({"marker": marker, "backend": backend, "location": location})

    monkeypatch.setattr(marker_ranges, "push_scope", push)
    monkeypatch.setattr(marker_ranges, "pop_scope", lambda: None)
    return pushes


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


def test_compiled_function_name_falls_back_to_wrapped_then_unknown():
    wrapped = types.SimpleNamespace(
        __name__=None, __wrapped__=types.SimpleNamespace(__name__="inner")
    )
    assert jax_backend._compiled_function_name(wrapped) == "inner"
    assert jax_backend._compiled_function_name(types.SimpleNamespace()) == "<unknown>"


def test_jax_jit_wrap_is_idempotent(monkeypatch):
    jax_mod, compiled_type, _primitive_type = install_jax_backend(monkeypatch)
    wrapped_once = compiled_type.__call__
    jax_backend.patch_jax_entry_points()
    assert compiled_type.__call__ is wrapped_once

    pushes = record_scopes(monkeypatch)

    def train_step(value):
        return value

    assert jax_mod.jit(train_step)(1) == 1
    assert [item["marker"] for item in pushes] == ["jax.jit.train_step"]


def test_jax_inner_jit_skips_second_marker(monkeypatch):
    _jax_mod, compiled_type, _primitive_type = install_jax_backend(monkeypatch)
    pushes = record_scopes(monkeypatch)
    inner = compiled_type("matmul_relu")
    outer = compiled_type("train_step", body=lambda *args, **kwargs: inner())
    outer()
    assert [item["marker"] for item in pushes] == ["jax.jit.train_step"]


def test_jax_range_restores_depth_after_exception(monkeypatch):
    _jax_mod, compiled_type, _primitive_type = install_jax_backend(monkeypatch)
    pushes = record_scopes(monkeypatch)

    def boom(*args, **kwargs):
        raise ValueError("jit failed")

    try:
        compiled_type("boom", body=boom)()
    except ValueError:
        pass
    pushes.clear()
    assert compiled_type("ok", body=lambda *args, **kwargs: 1)() == 1
    assert [item["marker"] for item in pushes] == ["jax.jit.ok"]


@pytest.mark.parametrize("primitive_name", ["dot_general", "reduce_sum"])
def test_jax_primitive_emits_named_marker(monkeypatch, primitive_name):
    _jax_mod, _compiled_type, primitive_type = install_jax_backend(monkeypatch)
    pushes = record_scopes(monkeypatch)
    primitive_type(primitive_name).bind_with_trace(None, (), (), {})
    assert [item["marker"] for item in pushes] == [f"jax.{primitive_name}"]
    assert pushes[0]["backend"] == "jax"


def test_jax_primitive_skips_when_range_open(monkeypatch):
    _jax_mod, compiled_type, primitive_type = install_jax_backend(monkeypatch)
    pushes = record_scopes(monkeypatch)
    primitive = primitive_type("dot_general")
    compiled = compiled_type(
        "train_step",
        body=lambda *args, **kwargs: primitive.bind_with_trace(None, (), (), {}),
    )
    compiled()
    assert [item["marker"] for item in pushes] == ["jax.jit.train_step"]


def test_jax_pmap_call_emits_pmap_marker(monkeypatch):
    jax_mod, _compiled_type, _primitive_type = install_jax_backend(monkeypatch)
    pushes = record_scopes(monkeypatch)

    def pmap_square(value):
        return value

    mapped = jax_mod.pmap(pmap_square)
    assert mapped(1) == 1
    assert [item["marker"] for item in pushes] == ["jax.pmap.pmap_square"]
    assert pushes[0]["backend"] == "jax"


def test_jax_pmap_method_binds_self(monkeypatch):
    jax_mod, _compiled_type, _primitive_type = install_jax_backend(monkeypatch)
    pushes = record_scopes(monkeypatch)

    class Worker:
        @jax_mod.pmap
        def step(self, value):
            return value

    assert Worker().step(1) == 1
    assert [item["marker"] for item in pushes] == ["jax.pmap.step"]
    assert pushes[0]["backend"] == "jax"


def test_jax_pmap_forwards_lower(monkeypatch):
    jax_mod, _compiled_type, _primitive_type = install_jax_backend(monkeypatch)

    def pmap_square(value):
        return value

    assert jax_mod.pmap(pmap_square).lower() == "lowered"


def test_jax_pmap_wrap_is_idempotent(monkeypatch):
    jax_mod, _compiled_type, _primitive_type = install_jax_backend(monkeypatch)
    wrapped_once = jax_mod.pmap
    jax_backend.patch_jax_entry_points()
    assert jax_mod.pmap is wrapped_once

    pushes = record_scopes(monkeypatch)

    def pmap_square(value):
        return value

    assert jax_mod.pmap(pmap_square)(1) == 1
    assert [item["marker"] for item in pushes] == ["jax.pmap.pmap_square"]


def test_jax_shard_map_call_emits_shard_map_marker(monkeypatch):
    jax_mod, _compiled_type, _primitive_type = install_jax_backend(monkeypatch)
    pushes = record_scopes(monkeypatch)

    def shard_map_square(value):
        return value

    assert jax_mod.shard_map(shard_map_square)(1) == 1
    assert [item["marker"] for item in pushes] == ["jax.shard_map.shard_map_square"]
    assert pushes[0]["backend"] == "jax"


def test_jax_shard_map_decorator_factory_emits_shard_map_marker(monkeypatch):
    jax_mod, _compiled_type, _primitive_type = install_jax_backend(monkeypatch)
    pushes = record_scopes(monkeypatch)

    def shard_map_cube(value):
        return value

    mapped = jax_mod.shard_map(mesh=None)(shard_map_cube)
    assert mapped(1) == 1
    assert [item["marker"] for item in pushes] == ["jax.shard_map.shard_map_cube"]
    assert pushes[0]["backend"] == "jax"
