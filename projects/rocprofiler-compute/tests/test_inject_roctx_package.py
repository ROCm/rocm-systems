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


def test_install_global_wraps_empty_inputs_are_noop(captured_install):
    from utils.inject_roctx.core import install_global_wraps

    install_global_wraps("")
    install_global_wraps([])
    assert captured_install == []


def test_install_global_wraps_single_name(captured_install):
    from utils.inject_roctx.core import install_global_wraps

    install_global_wraps("torch")
    assert captured_install == [["torch"]]


def test_install_global_wraps_comma_split_with_whitespace(captured_install):
    from utils.inject_roctx.core import install_global_wraps

    install_global_wraps("torch , , triton")
    assert captured_install == [["torch", "triton"]]


def test_install_global_wraps_iterable_input(captured_install):
    from utils.inject_roctx.core import install_global_wraps

    install_global_wraps(["torch", "triton"])
    assert captured_install == [["torch", "triton"]]


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


# ---------------------------------------------------------------------------
# core push/pop
# ---------------------------------------------------------------------------


@pytest.fixture
def core_with_python_tier():
    """Return ``core`` wired to in-memory push/pop sinks with empty stacks."""
    from utils.inject_roctx import core

    pushed: list[str] = []
    popped: list[None] = []
    core.set_python_tier_io(push=pushed.append, pop=lambda: popped.append(None))
    for attr in ("marker_stack", "context_stack"):
        if hasattr(core._thread_local, attr):
            delattr(core._thread_local, attr)
    return core, pushed, popped


@pytest.fixture
def torch_backend_tiers():
    """Return the torch backend wired to in-memory Python-tier sinks."""
    from utils.inject_roctx import core
    from utils.inject_roctx._backends import torch as torch_mod

    pushed: list[str] = []
    popped: list[None] = []
    core.set_python_tier_io(push=pushed.append, pop=lambda: popped.append(None))
    for attr in ("marker_stack", "context_stack"):
        if hasattr(core._thread_local, attr):
            delattr(core._thread_local, attr)
    if hasattr(torch_mod._thread_local, "tier_stack"):
        delattr(torch_mod._thread_local, "tier_stack")
    saved_hook = torch_mod._STATE.native_hook
    torch_mod._STATE.native_hook = None
    try:
        yield torch_mod, pushed, popped
    finally:
        torch_mod._STATE.native_hook = saved_hook


def test_push_scope_appends_backend_suffix(core_with_python_tier):
    core, pushed, _ = core_with_python_tier

    core._push_scope("op", "#1@x:1", backend="torch")
    assert pushed == ["op:#1@x:1|torch"]


def test_push_scope_omits_suffix_when_backend_empty(core_with_python_tier):
    core, pushed, _ = core_with_python_tier

    core._push_scope("op", "#1@x:1")
    assert pushed == ["op:#1@x:1"]


def test_torch_push_scope_routes_to_native_tier_when_active(torch_backend_tiers):
    torch_backend, pushed, _ = torch_backend_tiers

    seen: list[tuple] = []

    class Hook:
        def active(self):
            return True

        def push(self, marker, context, backend):
            seen.append((marker, context, backend))
            return True

        def pop(self):
            pass

    torch_backend._STATE.native_hook = Hook()
    torch_backend._push_scope("op", "#1@x:1", backend="torch")

    assert seen == [("op", "#1@x:1", "torch")]
    assert pushed == []


def test_torch_pop_scope_routes_each_frame_to_its_originating_tier(torch_backend_tiers):
    torch_backend, pushed, popped = torch_backend_tiers

    native_pops: list[None] = []

    class Hook:
        active_flag = True

        def active(self):
            return self.active_flag

        def push(self, marker, context, backend):
            return True

        def pop(self):
            native_pops.append(None)

    hook = Hook()
    torch_backend._STATE.native_hook = hook
    torch_backend._push_scope("native_op", "#1@x:1", backend="torch")
    hook.active_flag = False
    torch_backend._push_scope("py_op", "#2@x:2", backend="torch")
    torch_backend._pop_scope()
    torch_backend._pop_scope()

    assert pushed == ["native_op/py_op:#1@x:1/#2@x:2|torch"]
    assert popped == [None]
    assert native_pops == [None]
