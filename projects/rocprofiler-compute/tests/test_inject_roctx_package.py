# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for the ``utils.inject_roctx`` public surface:
``core.install_global_wraps``, ``registry.install_many``, ``TritonBackend``,
and ``core._push_scope`` / ``_pop_scope``."""

import importlib
import sys
import types

import common  # noqa: F401
import pytest

# ---------------------------------------------------------------------------
# install_global_wraps
# ---------------------------------------------------------------------------


@pytest.fixture
def captured_install(monkeypatch):
    """Replace ``registry.install_many`` with a recorder."""
    from utils.inject_roctx import registry as registry_pkg

    calls: list[list[str]] = []

    def _record(names):
        calls.append(list(names))

    monkeypatch.setattr(registry_pkg, "install_many", _record)
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


def test_install_global_wraps_api_alias_expands(captured_install):
    from utils.inject_roctx.core import install_global_wraps

    install_global_wraps("api")
    assert captured_install == [["torch", "triton"]]


def test_install_global_wraps_api_alongside_explicit_name(captured_install):
    from utils.inject_roctx.core import install_global_wraps

    install_global_wraps("api,torch")
    assert captured_install == [["torch", "triton", "torch"]]


# ---------------------------------------------------------------------------
# registry.install_many
# ---------------------------------------------------------------------------


@pytest.fixture
def fresh_registry(monkeypatch):
    """Provide an isolated registry for ``install_many`` tests."""
    from utils.inject_roctx import registry as registry_pkg

    monkeypatch.setattr(registry_pkg, "_REGISTRY", {})
    return registry_pkg


def _make_backend(name, install_fn=None):
    backend = types.SimpleNamespace()
    backend.name = name
    backend.install = install_fn or (lambda: None)
    return backend


def test_install_many_invokes_registered_backends(fresh_registry):
    calls: list[str] = []
    fresh_registry.register(_make_backend("alpha", lambda: calls.append("alpha")))
    fresh_registry.register(_make_backend("beta", lambda: calls.append("beta")))

    fresh_registry.install_many(["alpha", "beta"])
    assert calls == ["alpha", "beta"]


def test_install_many_dedupes_duplicate_names(fresh_registry):
    calls: list[str] = []
    fresh_registry.register(_make_backend("alpha", lambda: calls.append("alpha")))

    fresh_registry.install_many(["alpha", "alpha", "alpha"])
    assert calls == ["alpha"]


def test_install_many_continues_after_backend_failure(fresh_registry, monkeypatch):
    warnings: list[tuple] = []
    monkeypatch.setattr("utils.logger.console_warning", lambda *a: warnings.append(a))

    other_calls: list[str] = []
    fresh_registry.register(
        _make_backend("bad", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    )
    fresh_registry.register(_make_backend("good", lambda: other_calls.append("good")))

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

    def _no_triton(name, *args, **kwargs):
        if name == "triton":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", _no_triton)

    warnings: list[tuple] = []
    monkeypatch.setattr(
        triton_backend, "console_warning", lambda *a: warnings.append(a)
    )

    triton_backend.TritonBackend().install()
    assert any(
        "Triton is not installed" in str(args[1]) for args in warnings if len(args) > 1
    )


def test_triton_backend_wraps_compiled_kernel_call(monkeypatch):
    from utils.inject_roctx._backends import triton as triton_backend

    pushes: list[tuple] = []
    pops: list[None] = []
    monkeypatch.setattr(
        triton_backend,
        "_push_scope",
        lambda marker, ctx, backend="": pushes.append((marker, ctx, backend)),
    )
    monkeypatch.setattr(triton_backend, "_pop_scope", lambda: pops.append(None))

    class FakeKernel:
        name = "my_kernel"

        def __call__(self, *a, **kw):
            return ("ran", a, kw)

    backend = triton_backend.TritonBackend()
    backend._compiled_kernel = FakeKernel
    backend.patch_launcher()

    out = FakeKernel()(1, x=2)
    assert out == ("ran", (1,), {"x": 2})
    assert len(pushes) == 1
    marker, ctx, backend = pushes[0]
    assert marker == "triton.CompiledKernel.my_kernel"
    assert ctx.startswith("#1@")
    assert backend == "triton"
    assert pops == [None]


def test_triton_backend_kernel_name_fallbacks(monkeypatch):
    from utils.inject_roctx._backends import triton as triton_backend

    pushes: list[str] = []
    monkeypatch.setattr(
        triton_backend,
        "_push_scope",
        lambda marker, ctx, backend="": pushes.append(marker),
    )
    monkeypatch.setattr(triton_backend, "_pop_scope", lambda: None)

    class KernelWithDictMeta:
        metadata = {"name": "from_meta"}

        def __call__(self):
            pass

    class KernelNoName:
        def __call__(self):
            pass

    for cls, expected in (
        (KernelWithDictMeta, "triton.CompiledKernel.from_meta"),
        (KernelNoName, "triton.CompiledKernel.<triton_kernel>"),
    ):
        backend = triton_backend.TritonBackend()
        backend._compiled_kernel = cls
        backend.patch_launcher()
        cls()()
        assert pushes[-1] == expected


def test_triton_backend_patch_is_idempotent(monkeypatch):
    from utils.inject_roctx._backends import triton as triton_backend

    monkeypatch.setattr(triton_backend, "_push_scope", lambda *a, **k: None)
    monkeypatch.setattr(triton_backend, "_pop_scope", lambda: None)

    class FakeKernel:
        name = "k"

        def __call__(self):
            pass

    backend = triton_backend.TritonBackend()
    backend._compiled_kernel = FakeKernel
    backend.patch_launcher()
    first = FakeKernel.__call__
    backend.patch_launcher()
    assert FakeKernel.__call__ is first


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
