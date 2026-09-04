# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for the ``utils.inject_roctx`` public surface:
``core.install_global_wraps``, ``registry.install_many``, ``TritonBackend``,
and ``core._push_scope`` / ``_pop_scope``."""

import functools
import importlib
import json
import sys
import types
from pathlib import Path

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


@pytest.fixture(autouse=True)
def reset_args_capture():
    """Reset operator-argument capture configuration to defaults around each
    test."""
    from utils.inject_roctx import core

    core.set_args_capture(True, False)
    yield
    core.set_args_capture(True, False)


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


def test_install_global_wraps_threads_capture_config(captured_install):
    from utils.inject_roctx import core

    core.install_global_wraps("torch", capture_args=True, capture_arg_values=True)
    assert core._STATE.capture_args is True
    assert core._STATE.capture_arg_values is True

    core.install_global_wraps("torch", capture_args=False, capture_arg_values=False)
    assert core._STATE.capture_args is False
    assert core._STATE.capture_arg_values is False


def test_install_global_wraps_empty_backends_leaves_config_untouched(captured_install):
    from utils.inject_roctx import core

    core.set_args_capture(True, False)
    core.install_global_wraps("", capture_args=False, capture_arg_values=True)
    # No backends: the call is a no-op and must not mutate capture config.
    assert core._STATE.capture_args is True
    assert core._STATE.capture_arg_values is False
    assert captured_install == []


# ---------------------------------------------------------------------------
# launch.parse_launcher_options
# ---------------------------------------------------------------------------


def test_parse_launcher_options_defaults_capture_on_shapes():
    from utils.inject_roctx import launch

    frameworks, capture_args, capture_arg_values, remaining = (
        launch.parse_launcher_options(["--frameworks", "torch", "--", "t.py", "-n"])
    )
    assert frameworks == ["torch"]
    assert capture_args is True
    assert capture_arg_values is False
    assert remaining == ["t.py", "-n"]


def test_parse_launcher_options_parses_capture_flags():
    from utils.inject_roctx import launch

    frameworks, capture_args, capture_arg_values, remaining = (
        launch.parse_launcher_options([
            "--frameworks",
            "torch",
            "triton",
            "--capture-args",
            "1",
            "--capture-arg-values",
            "1",
            "--",
            "t.py",
        ])
    )
    assert frameworks == ["torch", "triton"]
    assert capture_args is True
    assert capture_arg_values is True
    assert remaining == ["t.py"]


def test_parse_launcher_options_capture_off():
    from utils.inject_roctx import launch

    _frameworks, capture_args, capture_arg_values, _remaining = (
        launch.parse_launcher_options([
            "--capture-args",
            "0",
            "--capture-arg-values",
            "0",
            "--",
            "t.py",
        ])
    )
    assert capture_args is False
    assert capture_arg_values is False


@pytest.mark.parametrize(
    "value,expected",
    [
        ("1", True),
        ("on", True),
        ("true", True),
        ("YES", True),
        ("0", False),
        ("off", False),
        ("", False),
        ("nonsense", False),
    ],
)
def test_launch_flag_parsing(value, expected):
    from utils.inject_roctx import launch

    assert launch._flag(value) is expected


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
        lambda marker, ctx, backend="", args="": pushes.append((marker, backend)),
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
        lambda marker, ctx, backend="", args="": pushes.append(marker),
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
        lambda marker, ctx, backend="", args="": pushes.append(marker),
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
        lambda marker, ctx, backend="", args="": pushes.append(marker),
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

        def push(self, marker, context, backend, args=""):
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

        def push(self, marker, context, backend, args=""):
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


# ---------------------------------------------------------------------------
# Operator args capture
# ---------------------------------------------------------------------------

_MARKER_ARGS_CODEC_PATH = Path(__file__).resolve().parents[3] / "marker_args_codec.json"
_MARKER_ARGS_CODEC = json.loads(_MARKER_ARGS_CODEC_PATH.read_text(encoding="utf-8"))


def test_marker_args_codec_limits():
    from utils.inject_roctx import marker_format

    assert marker_format.MAX_ARGS_LEN == _MARKER_ARGS_CODEC["max_args_len"]
    assert marker_format.MAX_ARG_ITEMS == _MARKER_ARGS_CODEC["max_arg_items"]
    assert (
        marker_format.MAX_NESTED_ARG_ITEMS == _MARKER_ARGS_CODEC["max_nested_arg_items"]
    )


def test_python_arg_formatters_limit_nested_items():
    from utils.inject_roctx import core, marker_format
    from utils.inject_roctx._backends import torch as torch_backend
    from utils.inject_roctx._backends import triton as triton_backend

    core.set_args_capture(True, False)
    item_limit = marker_format.MAX_NESTED_ARG_ITEMS
    items = list(range(item_limit + 1))
    expected = f"[{', '.join(['int'] * item_limit)}]"

    assert torch_backend._format_dispatch_arg(items) == expected
    assert triton_backend._format_triton_arg(items) == expected


@pytest.mark.parametrize(
    "case",
    _MARKER_ARGS_CODEC["cases"],
    ids=[case["name"] for case in _MARKER_ARGS_CODEC["cases"]],
)
def test_marker_args_codec_encode_decode(case):
    from utils.inject_roctx import marker_format

    encoded = marker_format.encode_args(case["plaintext"])
    assert encoded == case["encoded"]
    assert marker_format.decode_args(encoded) == case["plaintext"]


def test_core_push_scope_appends_args_segment_before_backend(core_with_python_tier):
    core, pushed, _ = core_with_python_tier

    core._push_scope("op", "#1@x:1", backend="torch", args="(f32[2x2])")
    assert pushed == ["op:#1@x:1|args=(f32[2x2])|torch"]


def test_core_push_scope_args_segment_without_backend(core_with_python_tier):
    core, pushed, _ = core_with_python_tier

    core._push_scope("op", "#1@x:1", args="(f32[2x2])")
    assert pushed == ["op:#1@x:1|args=(f32[2x2])"]


def test_core_push_scope_encodes_pipe_in_args(core_with_python_tier):
    core, pushed, _ = core_with_python_tier

    core._push_scope("op", "#1@x:1", backend="torch", args="a|b")
    assert pushed == ["op:#1@x:1|args=a%7Cb|torch"]


def test_torch_push_scope_forwards_args_to_native_tier(torch_backend_tiers):
    torch_backend, pushed, _ = torch_backend_tiers

    seen: list[tuple] = []

    class Hook:
        def active(self):
            return True

        def push(self, marker, context, backend, args=""):
            seen.append((marker, context, backend, args))
            return True

        def pop(self):
            pass

    torch_backend._STATE.native_hook = Hook()
    torch_backend._push_scope("op", "#1@x:1", backend="torch", args="(f32[2x2])")

    assert seen == [("op", "#1@x:1", "torch", "(f32[2x2])")]
    assert pushed == []


def test_args_capture_config_gate():
    from utils.inject_roctx import core

    core.set_args_capture(True, False)
    assert core.args_capture_enabled() is True
    assert core.args_values_enabled() is False

    core.set_args_capture(False, False)
    assert core.args_capture_enabled() is False

    core.set_args_capture(False, True)
    assert core.args_values_enabled() is False

    core.set_args_capture(True, True)
    assert core.args_values_enabled() is True


def test_cap_args_truncates_long_blobs():
    from utils.inject_roctx import marker_format

    long_blob = "x" * (marker_format.MAX_ARGS_LEN + 50)
    capped = marker_format.cap_args(long_blob)
    assert capped.endswith("...")
    assert len(capped) == marker_format.MAX_ARGS_LEN + len("...")
    assert marker_format.cap_args("short") == "short"


def test_cap_args_keeps_closing_paren_when_parenthesized():
    from utils.inject_roctx import marker_format

    long_blob = "(" + "x" * (marker_format.MAX_ARGS_LEN + 50) + ")"
    capped = marker_format.cap_args(long_blob)
    assert capped.endswith("...)")
    assert len(capped) == marker_format.MAX_ARGS_LEN + len("...)")


def test_triton_build_args_tensor_and_scalar():
    from utils.inject_roctx import core
    from utils.inject_roctx._backends import triton as triton_backend

    fake_tensor = types.SimpleNamespace(shape=(2, 3), dtype="torch.float32")
    params = [
        types.SimpleNamespace(name="x_ptr"),
        types.SimpleNamespace(name="n_elements"),
    ]
    self_obj = types.SimpleNamespace(params=params)

    blob = triton_backend._build_triton_args(
        self_obj, (fake_tensor, 1024), {"grid": (8,), "BLOCK_SIZE": 256}
    )
    assert "x_ptr=float32[2x3]" in blob
    assert "n_elements=int" in blob
    assert "grid" not in blob
    assert "BLOCK_SIZE=int" in blob

    core.set_args_capture(True, True)
    blob_values = triton_backend._build_triton_args(
        self_obj, (fake_tensor, 1024), {"grid": (8,), "BLOCK_SIZE": 256}
    )
    assert "n_elements=1024" in blob_values
    assert "BLOCK_SIZE=256" in blob_values


def test_triton_build_args_drops_compiled_kernel_preamble():
    from utils.inject_roctx._backends import triton as triton_backend

    class _LazyDict:
        pass

    class _HookChain:
        pass

    fn = types.SimpleNamespace(
        params=[
            types.SimpleNamespace(name="x_ptr"),
            types.SimpleNamespace(name="out_ptr"),
            types.SimpleNamespace(name="n_elements"),
            types.SimpleNamespace(name="BLOCK_SIZE"),
        ]
    )
    self_obj = types.SimpleNamespace(params=None, src=types.SimpleNamespace(fn=fn))

    x = types.SimpleNamespace(shape=(8,), dtype="torch.float32")
    out = types.SimpleNamespace(shape=(8,), dtype="torch.float32")
    preamble = (8, 1, 1, 0, 12345, (4, 1, 0), _LazyDict(), _HookChain(), _HookChain())
    blob = triton_backend._build_triton_args(self_obj, preamble + (x, out, 8, 256), {})

    assert blob == (
        "(x_ptr=float32[8], out_ptr=float32[8], n_elements=int, BLOCK_SIZE=int)"
    )
    assert "LazyDict" not in blob
    assert "HookChain" not in blob


def test_triton_build_args_drops_internal_types_without_names():
    from utils.inject_roctx import core
    from utils.inject_roctx._backends import triton as triton_backend

    core.set_args_capture(True, True)

    class _LazyDict:
        pass

    _LazyDict.__name__ = "LazyDict"
    blob = triton_backend._build_triton_args(
        types.SimpleNamespace(), (_LazyDict(), 1024), {}
    )
    assert "LazyDict" not in blob
    assert "1024" in blob


def test_triton_build_args_caps_combined_positional_and_keyword():
    from utils.inject_roctx import marker_format
    from utils.inject_roctx._backends import triton as triton_backend

    cap = marker_format.MAX_ARG_ITEMS
    self_obj = types.SimpleNamespace(params=None)

    blob = triton_backend._build_triton_args(
        self_obj, tuple(range(cap)), {f"k{i}": i for i in range(cap)}
    )
    assert len(blob[1:-1].split(", ")) == cap
    assert "k0=" not in blob

    blob = triton_backend._build_triton_args(
        self_obj, (1, 2), {f"k{i}": i for i in range(cap)}
    )
    assert len(blob[1:-1].split(", ")) == cap
    assert "k0=int" in blob
    assert f"k{cap - 2}=" not in blob


def test_triton_build_args_respects_gate():
    from utils.inject_roctx import core
    from utils.inject_roctx._backends import triton as triton_backend

    core.set_args_capture(False, False)
    blob = triton_backend._build_triton_args(
        types.SimpleNamespace(params=None), (1, 2), {}
    )
    assert blob == ""


def test_torch_build_dispatch_args_formats(monkeypatch):
    from utils.inject_roctx import core
    from utils.inject_roctx._backends import torch as torch_backend

    class FakeTensor:
        def __init__(self, shape, dtype):
            self.shape = shape
            self.dtype = dtype

    fake_torch = types.SimpleNamespace(Tensor=FakeTensor)
    monkeypatch.setattr(torch_backend._STATE, "torch", fake_torch)

    func = types.SimpleNamespace(
        _schema=types.SimpleNamespace(arguments=[types.SimpleNamespace(name="self")])
    )

    t = FakeTensor((4, 8), "torch.float16")
    blob = torch_backend._build_dispatch_args(func, (t,), {"dim": 1})
    assert "self=float16[4x8]" in blob
    assert "dim=int" in blob

    core.set_args_capture(True, True)
    blob_values = torch_backend._build_dispatch_args(func, (t,), {"dim": 1})
    assert "dim=1" in blob_values


def test_torch_build_dispatch_args_without_schema(monkeypatch):
    from utils.inject_roctx._backends import torch as torch_backend

    class FakeTensor:
        def __init__(self, shape, dtype):
            self.shape = shape
            self.dtype = dtype

    fake_torch = types.SimpleNamespace(Tensor=FakeTensor)
    monkeypatch.setattr(torch_backend._STATE, "torch", fake_torch)

    t = FakeTensor((2, 3), "torch.float32")
    blob = torch_backend._build_dispatch_args(object(), (t,), {})
    assert blob == "(float32[2x3])"


def test_torch_build_dispatch_args_for_mm_relu_cat(monkeypatch):
    torch = pytest.importorskip("torch")
    from utils.inject_roctx import core
    from utils.inject_roctx._backends import torch as torch_backend

    monkeypatch.setattr(torch_backend._STATE, "torch", torch)
    core.set_args_capture(True, False)

    a = torch.zeros(2, 3, dtype=torch.float32)
    b = torch.zeros(3, 4, dtype=torch.float32)
    t = torch.zeros(2, 3, dtype=torch.float32)

    assert (
        torch_backend._build_dispatch_args(torch.ops.aten.mm.default, (a, b), {})
        == "(self=float32[2x3], mat2=float32[3x4])"
    )
    assert (
        torch_backend._build_dispatch_args(torch.ops.aten.relu.default, (t,), {})
        == "(self=float32[2x3])"
    )
    assert (
        torch_backend._build_dispatch_args(
            torch.ops.aten.cat.default, ([t, t],), {"dim": 0}
        )
        == "(tensors=[float32[2x3], float32[2x3]], dim=int)"
    )

    core.set_args_capture(True, True)
    assert (
        torch_backend._build_dispatch_args(
            torch.ops.aten.cat.default, ([t, t],), {"dim": 1}
        )
        == "(tensors=[float32[2x3], float32[2x3]], dim=1)"
    )
