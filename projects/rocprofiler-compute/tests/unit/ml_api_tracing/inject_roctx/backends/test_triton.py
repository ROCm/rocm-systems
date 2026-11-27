# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for Triton backend discovery and launch wrappers."""

import importlib
import inspect
import sys
import types
from pathlib import Path
from typing import List

from ml_api_tracing.inject_roctx import marker_ranges
from ml_api_tracing.inject_roctx.backends import triton as triton_backend
from ml_api_tracing.marker_format import MAX_TENSOR_ARGUMENT_ITEMS, compose_marker


class FindSpecWithout:
    """Behave like find_spec but report one name as missing."""

    def __init__(self, absent_name, real_find_spec):
        self.absent_name = absent_name
        self.real_find_spec = real_find_spec

    def __call__(self, name, *args, **kwargs):
        if name == self.absent_name:
            return None
        return self.real_find_spec(name, *args, **kwargs)


def test_triton_backend_skips_when_triton_missing(monkeypatch):
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        FindSpecWithout("triton", real_find_spec),
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
        marker_ranges,
        "push_scope",
        lambda marker, ctx, backend="", args="n/a": pushes.append((marker, backend)),
    )
    monkeypatch.setattr(marker_ranges, "pop_scope", lambda: None)
    monkeypatch.setattr(triton_backend._CoreState, "jit_function", None)

    class FakeCompiledKernel:
        name = "rk"

        def run(self, *a, **kw):
            return "ran"

    monkeypatch.setattr(
        triton_backend._CoreState, "compiled_kernel", FakeCompiledKernel
    )
    triton_backend.patch_triton_launcher()

    assert FakeCompiledKernel().run() == "ran"
    assert pushes == [("triton.CompiledKernel.rk", "triton")]


def test_triton_backend_wraps_jitfunction_run(monkeypatch):
    """JITFunction.run is wrapped for eager launches."""
    pushes: List[str] = []
    monkeypatch.setattr(
        marker_ranges,
        "push_scope",
        lambda marker, ctx, backend="", args="n/a": pushes.append(marker),
    )
    monkeypatch.setattr(marker_ranges, "pop_scope", lambda: None)
    monkeypatch.setattr(triton_backend._CoreState, "compiled_kernel", None)

    class FakeJIT:
        def __init__(self):
            self.fn = types.SimpleNamespace(__name__="add_kernel")

        def run(self, *a, **kw):
            return "launched"

    monkeypatch.setattr(triton_backend._CoreState, "jit_function", FakeJIT)
    triton_backend.patch_triton_launcher()

    assert FakeJIT().run() == "launched"
    assert pushes == ["triton.JITFunction.add_kernel"]


def test_triton_wrappers_pass_launch_keywords_named_like_wrapper_parameters(
    monkeypatch,
):
    """Launch keywords named like the wrapper parameters reach the kernel intact."""
    monkeypatch.setattr(marker_ranges, "push_scope", lambda *a, **kw: None)
    monkeypatch.setattr(marker_ranges, "pop_scope", lambda: None)
    monkeypatch.setattr(triton_backend._CoreState, "jit_function", None)
    received = {}

    class FakeCompiledKernel:
        name = "keywords"

        def run(self, *args, **kwargs):
            received["method"] = (args, kwargs)

        @property
        def launcher(self):
            def launch(*args, **kwargs):
                received["launcher"] = (args, kwargs)

            return launch

    monkeypatch.setattr(
        triton_backend._CoreState, "compiled_kernel", FakeCompiledKernel
    )
    triton_backend.patch_triton_launcher()
    assert triton_backend._wrap_launch(FakeCompiledKernel, "launcher", True)
    keywords = {
        "instance": 1,
        "is_compiled": 2,
        "original": 3,
        "call": 4,
    }

    FakeCompiledKernel().run(7, **keywords)
    FakeCompiledKernel().launcher(8, **keywords)

    assert received["method"] == ((7,), keywords)
    assert received["launcher"] == ((8,), keywords)


def test_triton_backend_patch_is_idempotent(monkeypatch):
    """Patching twice does not re-wrap the launch entry point."""
    pushes: List[str] = []
    monkeypatch.setattr(
        marker_ranges,
        "push_scope",
        lambda marker, ctx, backend="", args="n/a": pushes.append(marker),
    )
    monkeypatch.setattr(marker_ranges, "pop_scope", lambda: None)

    class FakeJIT:
        name = "k"

        def run(self, *a, **kw):
            return "ran"

    monkeypatch.setattr(triton_backend._CoreState, "compiled_kernel", None)
    monkeypatch.setattr(triton_backend._CoreState, "jit_function", FakeJIT)

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
    monkeypatch.setattr(
        triton_backend.marker_ranges, "ensure_python_tier", lambda: True
    )
    monkeypatch.setattr(triton_backend, "patch_triton_launcher", lambda: None)

    fake_triton = types.ModuleType("triton")
    fake_triton.__file__ = "/opt/fake/triton/__init__.py"
    monkeypatch.setitem(sys.modules, "triton", fake_triton)

    roots: List[str] = []
    monkeypatch.setattr(
        triton_backend.marker_ranges, "add_framework_root", lambda p: roots.append(p)
    )

    triton_backend.TritonBackend().install()
    assert roots == ["/opt/fake/triton"]


def test_triton_backend_skips_when_python_tier_unavailable(monkeypatch):
    monkeypatch.setattr(triton_backend, "_resolve_triton", lambda: True)
    monkeypatch.setattr(
        triton_backend.marker_ranges, "ensure_python_tier", lambda: False
    )

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
    monkeypatch.setattr(marker_ranges._CoreState, "range_push", markers.append)
    monkeypatch.setattr(
        marker_ranges._CoreState, "range_pop", lambda: pops.append("pop")
    )
    monkeypatch.setattr(marker_ranges._thread_local, "depth", 0, raising=False)
    caller_line = inspect.currentframe().f_lineno + 1
    value = triton_backend._run_with_marker(
        types.SimpleNamespace(name="kernel"),
        True,
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


def test_format_triton_args_jit_ignores_extra_trailing_positionals():
    kernel = types.SimpleNamespace(params=["x_ptr", "n_elements"])
    tensor = types.SimpleNamespace(shape=(8,), dtype="float32")
    blob = triton_backend._format_triton_args(
        kernel,
        False,
        (tensor, 8, 256),
        {},
    )
    assert blob == "(x_ptr=float32[8], n_elements=8)"


def test_format_triton_args_compiled_reads_names_from_src_fn():
    tensor = types.SimpleNamespace(shape=(8,), dtype="float32")
    kernel = types.SimpleNamespace(
        src=types.SimpleNamespace(
            fn=types.SimpleNamespace(arg_names=["x_ptr", "n_elements"])
        )
    )
    blob = triton_backend._format_triton_args(
        kernel,
        True,
        (1, 1, 1, None, None, None, None, None, None, tensor, 8),
        {},
    )
    assert blob == "(x_ptr=float32[8], n_elements=8)"


def test_format_triton_args_uses_arg_names_when_params_empty():
    tensor = types.SimpleNamespace(shape=(2,), dtype="float32")
    blob = triton_backend._format_triton_args(
        types.SimpleNamespace(params=(), arg_names=["x_ptr"]),
        False,
        (tensor,),
        {},
    )
    assert blob == "(x_ptr=float32[2])"


def test_format_triton_args_without_names_is_n_a():
    blob = triton_backend._format_triton_args(
        types.SimpleNamespace(),
        False,
        (object(), 1024),
        {"grid": (8,)},
    )
    assert blob == "n/a"


def test_format_triton_args_exception_is_n_a():
    class Boom:
        @property
        def params(self):
            raise RuntimeError("cannot walk params")

        arg_names = []

    blob = triton_backend._format_triton_args(
        Boom(),
        False,
        (1,),
        {},
    )
    assert blob == "n/a"


def test_format_triton_args_renders_float_and_bool():
    blob = triton_backend._format_triton_args(
        types.SimpleNamespace(params=["scale", "flag"]),
        False,
        (0.5, True),
        {},
    )
    assert blob == "(scale=0.5, flag=True)"


def test_format_triton_args_caps_at_32_items():
    names = ["p%d" % index for index in range(40)]
    blob = triton_backend._format_triton_args(
        types.SimpleNamespace(params=names),
        False,
        tuple(range(40)),
        {},
    )
    assert blob.count("=") == MAX_TENSOR_ARGUMENT_ITEMS
    assert "p31=" in blob
    assert "p32=" not in blob


def test_triton_wrap_records_kernel_operand_on_marker(monkeypatch):
    recorded = []
    monkeypatch.setattr(
        marker_ranges,
        "push_scope",
        lambda marker, ctx, backend="", args="n/a": recorded.append(args),
    )
    monkeypatch.setattr(marker_ranges, "pop_scope", lambda: None)
    monkeypatch.setattr(triton_backend._CoreState, "compiled_kernel", None)

    class FakeJIT:
        def __init__(self):
            self.params = [
                types.SimpleNamespace(name="x_ptr"),
                types.SimpleNamespace(name="n_elements"),
                types.SimpleNamespace(name="BLOCK_SIZE"),
            ]

        def run(self, *a, **kw):
            return "launched"

    monkeypatch.setattr(triton_backend._CoreState, "jit_function", FakeJIT)
    triton_backend.patch_triton_launcher()

    tensor = types.SimpleNamespace(shape=(2, 4), dtype="float32")
    assert FakeJIT().run(tensor, 8, grid=(8,), BLOCK_SIZE=256) == "launched"
    assert recorded
    assert recorded[0] == "(x_ptr=float32[2x4], n_elements=8, BLOCK_SIZE=256)"


def test_triton_wrap_records_compiled_kernel_operands_from_src_fn(monkeypatch):
    recorded = []
    monkeypatch.setattr(
        marker_ranges,
        "push_scope",
        lambda marker, ctx, backend="", args="n/a": recorded.append((marker, args)),
    )
    monkeypatch.setattr(marker_ranges, "pop_scope", lambda: None)
    monkeypatch.setattr(triton_backend._CoreState, "jit_function", None)

    class FakeCompiledKernel:
        name = "rk"

        def __init__(self):
            self.src = types.SimpleNamespace(
                fn=types.SimpleNamespace(arg_names=["x_ptr", "n_elements"])
            )

        @property
        def run(self):
            def launch(*args, **kwargs):
                return "ran"

            return launch

    monkeypatch.setattr(
        triton_backend._CoreState, "compiled_kernel", FakeCompiledKernel
    )
    triton_backend.patch_triton_launcher()

    tensor = types.SimpleNamespace(shape=(2, 4), dtype="float32")
    prefix = (1, 1, 1, None, None, None, None, None, None)
    assert FakeCompiledKernel().run(*prefix, tensor, 8) == "ran"
    assert recorded
    marker, args = recorded[0]
    assert marker == "triton.CompiledKernel.rk"
    assert args == "(x_ptr=float32[2x4], n_elements=8)"
