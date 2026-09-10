# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for the ``utils.inject_roctx`` public surface:
``core.install_global_wraps``, ``registry.install_many``, ``TritonBackend``,
and ``core._push_scope`` / ``_pop_scope``."""

import functools
import importlib
import sys
import types

import common  # noqa: F401
import pytest

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def record_call(calls, names):
    """Append a snapshot of ``names`` to ``calls`` (an ``install_many`` spy)."""
    calls.append(list(names))


def find_spec_without_name(absent_name, real_find_spec, name, *args, **kwargs):
    """Behave like ``find_spec`` but report ``absent_name`` as missing."""
    if name == absent_name:
        return None
    return real_find_spec(name, *args, **kwargs)


def raise_import_skipped(*args, **kwargs):
    """Raise to assert that an import path is never reached."""
    raise AssertionError("roctx import should be skipped")


def make_backend(name, install_fn=None):
    backend = types.SimpleNamespace()
    backend.name = name
    backend.install = install_fn or (lambda: None)
    return backend


# ---------------------------------------------------------------------------
# install_global_wraps
# ---------------------------------------------------------------------------


@pytest.fixture
def captured_install(monkeypatch):
    """Replace ``registry.install_many`` with a recorder."""
    from utils.inject_roctx import registry as registry_pkg

    calls: list[list[str]] = []
    monkeypatch.setattr(
        registry_pkg, "install_many", functools.partial(record_call, calls)
    )
    return calls


def test_install_global_wraps_empty_input_is_noop(captured_install):
    from utils.inject_roctx.core import install_global_wraps

    install_global_wraps([])
    install_global_wraps(())
    assert captured_install == []


def test_install_global_wraps_list_input(captured_install):
    from utils.inject_roctx.core import install_global_wraps

    install_global_wraps(["torch"])
    install_global_wraps(["torch", "triton"])
    assert captured_install == [["torch"], ["torch", "triton"]]


# ---------------------------------------------------------------------------
# registry.install_many
# ---------------------------------------------------------------------------


@pytest.fixture
def fresh_registry(monkeypatch):
    """Provide an isolated registry for ``install_many`` tests."""
    from utils.inject_roctx import registry as registry_pkg

    monkeypatch.setattr(registry_pkg, "_REGISTRY", {})
    return registry_pkg


def test_install_many_invokes_registered_backends(fresh_registry):
    calls: list[str] = []
    fresh_registry.register(make_backend("alpha", lambda: calls.append("alpha")))
    fresh_registry.register(make_backend("beta", lambda: calls.append("beta")))

    fresh_registry.install_many(["alpha", "beta"])
    assert calls == ["alpha", "beta"]


def test_install_many_dedupes_duplicate_names(fresh_registry):
    calls: list[str] = []
    fresh_registry.register(make_backend("alpha", lambda: calls.append("alpha")))

    fresh_registry.install_many(["alpha", "alpha", "alpha"])
    assert calls == ["alpha"]


def test_install_many_continues_after_backend_failure(fresh_registry, monkeypatch):
    warnings: list[tuple] = []
    monkeypatch.setattr("utils.logger.console_warning", lambda *a: warnings.append(a))

    other_calls: list[str] = []
    fresh_registry.register(
        make_backend("bad", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    )
    fresh_registry.register(make_backend("good", lambda: other_calls.append("good")))

    fresh_registry.install_many(["bad", "good"])
    assert other_calls == ["good"]
    assert any("bad" in str(args[1]) for args in warnings if len(args) > 1)


def test_install_many_warns_on_unknown_backend(fresh_registry, monkeypatch):
    warnings: list[tuple] = []
    monkeypatch.setattr("utils.logger.console_warning", lambda *a: warnings.append(a))

    fresh_registry.install_many(["does_not_exist_zzz"])
    assert any(
        "does_not_exist_zzz" in str(args[1]) for args in warnings if len(args) > 1
    )


def test_install_many_warns_when_module_does_not_register(fresh_registry, monkeypatch):
    warnings: list[tuple] = []
    monkeypatch.setattr("utils.logger.console_warning", lambda *a: warnings.append(a))

    fake_name = "utils.inject_roctx._backends.ghost"
    sys.modules[fake_name] = types.ModuleType(fake_name)
    try:
        fresh_registry.install_many(["ghost"])
    finally:
        sys.modules.pop(fake_name, None)

    assert any("did not register" in str(args[1]) for args in warnings if len(args) > 1)


# ---------------------------------------------------------------------------
# TritonBackend
# ---------------------------------------------------------------------------


def test_triton_backend_skips_when_triton_missing(monkeypatch):
    from utils.inject_roctx._backends import triton as triton_backend

    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        functools.partial(find_spec_without_name, "triton", real_find_spec),
    )

    warnings: list[tuple] = []
    monkeypatch.setattr(
        triton_backend, "console_warning", lambda *a: warnings.append(a)
    )

    triton_backend.TritonBackend().install()
    assert any(
        "Triton is not installed" in str(args[1]) for args in warnings if len(args) > 1
    )


def test_triton_backend_wraps_compiled_kernel_run(monkeypatch):
    """CompiledKernel.run() is wrapped in preference to __call__."""
    from utils.inject_roctx._backends import triton as triton_backend

    pushes: list[tuple] = []
    monkeypatch.setattr(
        triton_backend,
        "_push_scope",
        lambda marker, ctx, backend="": pushes.append((marker, backend)),
    )
    monkeypatch.setattr(triton_backend, "_pop_scope", lambda: None)
    monkeypatch.setattr(triton_backend._STATE, "jit_function", None)

    class FakeCompiledKernel:
        name = "rk"

        def run(self, *a, **kw):
            return "ran"

    monkeypatch.setattr(triton_backend._STATE, "compiled_kernel", FakeCompiledKernel)
    triton_backend.patch_triton_launcher()

    assert FakeCompiledKernel().run() == "ran"
    assert pushes == [("triton.CompiledKernel.rk", "triton")]


def test_triton_backend_wraps_jitfunction_run(monkeypatch):
    """JITFunction.run is wrapped for eager launches."""
    from utils.inject_roctx._backends import triton as triton_backend

    pushes: list[str] = []
    monkeypatch.setattr(
        triton_backend,
        "_push_scope",
        lambda marker, ctx, backend="": pushes.append(marker),
    )
    monkeypatch.setattr(triton_backend, "_pop_scope", lambda: None)
    monkeypatch.setattr(triton_backend._STATE, "compiled_kernel", None)

    class FakeJIT:
        def __init__(self):
            self.fn = types.SimpleNamespace(__name__="add_kernel")

        def run(self, *a, **kw):
            return "launched"

    monkeypatch.setattr(triton_backend._STATE, "jit_function", FakeJIT)
    triton_backend.patch_triton_launcher()

    assert FakeJIT().run() == "launched"
    assert pushes == ["triton.JITFunction.add_kernel"]


def test_triton_backend_reentrancy_dedups_nested_launch(monkeypatch):
    """Nested JITFunction.run and CompiledKernel.run emit one marker."""
    from utils.inject_roctx._backends import triton as triton_backend

    pushes: list[str] = []
    monkeypatch.setattr(
        triton_backend,
        "_push_scope",
        lambda marker, ctx, backend="": pushes.append(marker),
    )
    monkeypatch.setattr(triton_backend, "_pop_scope", lambda: None)
    # Reset the per-thread guard.
    if hasattr(triton_backend._thread_local, "in_launch"):
        del triton_backend._thread_local.in_launch

    class FakeCompiledKernel:
        name = "inner"

        def run(self, *a, **kw):
            return "inner_ran"

    class FakeJIT:
        name = "outer"

        def __init__(self, compiled):
            self._compiled = compiled

        def run(self, *a, **kw):
            return self._compiled.run()

    monkeypatch.setattr(triton_backend._STATE, "compiled_kernel", FakeCompiledKernel)
    monkeypatch.setattr(triton_backend._STATE, "jit_function", FakeJIT)
    triton_backend.patch_triton_launcher()

    compiled = FakeCompiledKernel()
    out = FakeJIT(compiled).run()

    assert out == "inner_ran"
    assert pushes == ["triton.JITFunction.outer"]


def test_triton_backend_patch_is_idempotent(monkeypatch):
    """Patching twice does not re-wrap the launch entry point."""
    from utils.inject_roctx._backends import triton as triton_backend

    pushes: list[str] = []
    monkeypatch.setattr(
        triton_backend,
        "_push_scope",
        lambda marker, ctx, backend="": pushes.append(marker),
    )
    monkeypatch.setattr(triton_backend, "_pop_scope", lambda: None)
    # Reset the per-thread guard.
    if hasattr(triton_backend._thread_local, "in_launch"):
        del triton_backend._thread_local.in_launch

    class FakeJIT:
        name = "k"

        def run(self, *a, **kw):
            return "ran"

    monkeypatch.setattr(triton_backend._STATE, "compiled_kernel", None)
    monkeypatch.setattr(triton_backend._STATE, "jit_function", FakeJIT)

    triton_backend.patch_triton_launcher()
    wrapped_once = FakeJIT.__dict__["run"]
    triton_backend.patch_triton_launcher()

    assert FakeJIT.__dict__["run"] is wrapped_once, (
        "second patch re-wrapped the launcher"
    )
    assert FakeJIT().run() == "ran"
    assert pushes == ["triton.JITFunction.k"], "exactly one marker per launch"


def test_triton_backend_registers_framework_root(monkeypatch):
    """install() registers triton's package directory as a framework root."""
    from utils.inject_roctx._backends import triton as triton_backend

    monkeypatch.setattr(triton_backend, "_resolve_triton", lambda: True)
    monkeypatch.setattr(triton_backend, "patch_triton_launcher", lambda: None)

    fake_triton = types.ModuleType("triton")
    fake_triton.__file__ = "/opt/fake/triton/__init__.py"
    monkeypatch.setitem(sys.modules, "triton", fake_triton)

    roots: list[str] = []
    monkeypatch.setattr(
        triton_backend.core, "add_framework_root", lambda p: roots.append(p)
    )

    triton_backend.TritonBackend().install()
    assert roots == ["/opt/fake/triton"]


def test_triton_backend_skips_when_python_tier_unavailable(monkeypatch):
    from utils.inject_roctx._backends import triton as triton_backend

    monkeypatch.setattr(triton_backend, "_resolve_triton", lambda: True)
    monkeypatch.setattr(triton_backend.core, "ensure_python_tier", lambda: False)

    patched: list[bool] = []
    monkeypatch.setattr(
        triton_backend, "patch_triton_launcher", lambda: patched.append(True)
    )
    warnings: list[tuple] = []
    monkeypatch.setattr(
        triton_backend, "console_warning", lambda *a: warnings.append(a)
    )

    triton_backend.TritonBackend().install()
    assert patched == []
    assert any("ROCTX bindings not found" in str(a[1]) for a in warnings if len(a) > 1)


def test_ensure_python_tier_short_circuits_when_already_configured(monkeypatch):
    from utils.inject_roctx import core

    saved_push, saved_pop = core._STATE.range_push, core._STATE.range_pop
    try:
        core.set_python_tier_io(lambda _s: None, lambda: None)

        monkeypatch.setattr(core.importlib, "import_module", raise_import_skipped)
        assert core.ensure_python_tier() is True
    finally:
        core._STATE.range_push, core._STATE.range_pop = saved_push, saved_pop


def test_extract_kernel_name_prefers_attr_then_meta_then_fn():
    from utils.inject_roctx._backends import triton as triton_backend

    named = types.SimpleNamespace(name="direct")
    assert triton_backend._extract_kernel_name(named) == "direct"

    meta = types.SimpleNamespace(metadata={"name": "meta_name"})
    assert triton_backend._extract_kernel_name(meta) == "meta_name"

    via_fn = types.SimpleNamespace(fn=types.SimpleNamespace(__name__="fn_name"))
    assert triton_backend._extract_kernel_name(via_fn) == "fn_name"

    assert (
        triton_backend._extract_kernel_name(types.SimpleNamespace())
        == "<triton_kernel>"
    )
