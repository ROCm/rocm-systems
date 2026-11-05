# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for Triton backend discovery and launch wrappers."""

import functools
import importlib
import inspect
import sys
import types
from pathlib import Path
from typing import List

from ml_api_tracing.marker_format import compose_marker
from ml_api_tracing.recording import core
from ml_api_tracing.recording.backends import triton as triton_backend


def find_spec_without_name(absent_name, real_find_spec, name, *args, **kwargs):
    """Behave like find_spec but report absent_name as missing."""
    if name == absent_name:
        return None
    return real_find_spec(name, *args, **kwargs)


def test_triton_backend_skips_when_triton_missing(monkeypatch):
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        functools.partial(find_spec_without_name, "triton", real_find_spec),
    )

    warnings: List[tuple] = []
    monkeypatch.setattr(
        triton_backend, "console_warning", lambda *a: warnings.append(a)
    )

    triton_backend.TritonBackend().install()
    assert any(
        "Triton is not installed" in str(args[1]) for args in warnings if len(args) > 1
    )


def test_triton_backend_wraps_compiled_kernel_run(monkeypatch):
    """CompiledKernel.run() is wrapped in preference to __call__."""
    pushes: List[tuple] = []
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
    pushes: List[str] = []
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


def test_triton_backend_patch_is_idempotent(monkeypatch):
    """Patching twice does not re-wrap the launch entry point."""
    pushes: List[str] = []
    monkeypatch.setattr(
        triton_backend,
        "_push_scope",
        lambda marker, ctx, backend="": pushes.append(marker),
    )
    monkeypatch.setattr(triton_backend, "_pop_scope", lambda: None)

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
    monkeypatch.setattr(triton_backend, "_resolve_triton", lambda: True)
    monkeypatch.setattr(triton_backend.core, "ensure_python_tier", lambda: True)
    monkeypatch.setattr(triton_backend, "patch_triton_launcher", lambda: None)

    fake_triton = types.ModuleType("triton")
    fake_triton.__file__ = "/opt/fake/triton/__init__.py"
    monkeypatch.setitem(sys.modules, "triton", fake_triton)

    roots: List[str] = []
    monkeypatch.setattr(
        triton_backend.core, "add_framework_root", lambda p: roots.append(p)
    )

    triton_backend.TritonBackend().install()
    assert roots == ["/opt/fake/triton"]


def test_triton_backend_skips_when_python_tier_unavailable(monkeypatch):
    monkeypatch.setattr(triton_backend, "_resolve_triton", lambda: True)
    monkeypatch.setattr(triton_backend.core, "ensure_python_tier", lambda: False)

    patched: List[bool] = []
    monkeypatch.setattr(
        triton_backend, "patch_triton_launcher", lambda: patched.append(True)
    )
    warnings: List[tuple] = []
    monkeypatch.setattr(
        triton_backend, "console_warning", lambda *a: warnings.append(a)
    )

    triton_backend.TritonBackend().install()
    assert patched == []
    assert any("ROCTX bindings not found" in str(a[1]) for a in warnings if len(a) > 1)


def test_extract_kernel_name_prefers_attr_then_meta_then_fn():
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


def test_launcher_marker_attributes_actual_recording_frames_to_user(monkeypatch):
    markers = []
    pops = []
    monkeypatch.setattr(core._STATE, "range_push", markers.append)
    monkeypatch.setattr(core._STATE, "range_pop", lambda: pops.append("pop"))
    monkeypatch.setattr(core._thread_local, "depth", 0, raising=False)
    caller_line = inspect.currentframe().f_lineno + 1
    value = triton_backend._run_with_marker(
        types.SimpleNamespace(name="kernel"),
        "triton.CompiledKernel",
        lambda: "launched",
    )
    assert value == "launched"
    assert markers == [
        compose_marker(
            "triton.CompiledKernel.kernel",
            f"{Path(__file__).name}:{caller_line}",
            backend="triton",
        )
    ]
    assert pops == ["pop"]
