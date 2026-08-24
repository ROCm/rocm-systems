# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for utils.inject_roctx._backends.torch_cpp_loader. No GPU."""

import importlib
import sys
import types
from pathlib import Path

import common  # noqa: F401
import pytest

from utils.inject_roctx._backends import torch_cpp_loader as inject_roctx_loader

_FAKE_TORCH_VERSION = "2.9.0"


# ---------------------------------------------------------------------------
# torch_version
# ---------------------------------------------------------------------------


def test_torch_version_strips_the_local_build_segment(monkeypatch):
    """``torch_version()`` drops a local ``+...`` build suffix."""
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(__version__="2.9.0+rocm7.1"),
    )
    assert inject_roctx_loader.torch_version() == "2.9.0"


def test_torch_version_is_none_without_torch(monkeypatch):
    """``torch_version()`` returns ``None`` when torch is not importable."""
    import builtins

    real_import = builtins.__import__

    def _import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch" or (
            isinstance(name, str) and name.startswith("torch.")
        ):
            raise ImportError("torch missing")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _import)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    assert inject_roctx_loader.torch_version() is None


# ---------------------------------------------------------------------------
# Artifact discovery
# ---------------------------------------------------------------------------


def _install_tree_with_artifacts(tmp_path: Path, *versions: str) -> Path:
    """Create ``<prefix>/lib/rocprofiler-compute/torch_trace_collector-*.so``."""
    install_root = tmp_path / "opt" / "rocm"
    artifact_dir = install_root / "lib" / "rocprofiler-compute"
    artifact_dir.mkdir(parents=True)
    for version in versions:
        (artifact_dir / f"torch_trace_collector-{version}.so").write_bytes(b"stub")
    return install_root


def test_torch_version_from_artifact_name():
    path = Path("torch_trace_collector-2.9.0.so")
    assert inject_roctx_loader._torch_version_from_artifact_name(path) == "2.9.0"
    assert (
        inject_roctx_loader._torch_version_from_artifact_name(Path("other.so")) is None
    )


def test_supported_torch_versions_lists_installed_artifacts(tmp_path, monkeypatch):
    install_root = _install_tree_with_artifacts(tmp_path, "2.8.0", "2.9.0", "2.8.0")
    monkeypatch.setattr(inject_roctx_loader, "_install_prefix", lambda: install_root)
    assert inject_roctx_loader.supported_torch_versions() == ("2.8.0", "2.9.0")


def test_supported_torch_versions_is_empty_without_artifacts(tmp_path, monkeypatch):
    install_root = tmp_path / "opt" / "rocm"
    (install_root / "lib" / "rocprofiler-compute").mkdir(parents=True)
    monkeypatch.setattr(inject_roctx_loader, "_install_prefix", lambda: install_root)
    assert inject_roctx_loader.supported_torch_versions() == ()


# ---------------------------------------------------------------------------
# Tier surface
# ---------------------------------------------------------------------------


def test_c_tier_names_is_prebuilt_only():
    assert inject_roctx_loader.C_TIER_NAMES == frozenset((
        inject_roctx_loader.TIER_PREBUILT,
    ))


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------


def test_force_python_fallback_returns_none():
    assert inject_roctx_loader.load(force_python_fallback=True).module is None


def test_load_does_not_raise_when_torch_missing(monkeypatch):
    monkeypatch.setattr(inject_roctx_loader, "torch_version", lambda: None)
    assert inject_roctx_loader.load().module is None


def test_load_raises_when_torch_version_is_unsupported(monkeypatch, tmp_path):
    install_root = _install_tree_with_artifacts(tmp_path, "2.8.0")
    monkeypatch.setattr(inject_roctx_loader, "_install_prefix", lambda: install_root)
    monkeypatch.setattr(
        inject_roctx_loader, "torch_version", lambda: _FAKE_TORCH_VERSION
    )

    with pytest.raises(inject_roctx_loader.UnsupportedTorchVersionError) as raised:
        inject_roctx_loader.load()

    error = raised.value
    assert error.workload_torch_version == _FAKE_TORCH_VERSION
    assert error.supported_torch_versions == ("2.8.0",)
    assert _FAKE_TORCH_VERSION in str(error)
    assert "2.8.0" in str(error)


def test_load_raises_when_no_artifacts_are_installed(monkeypatch, tmp_path):
    install_root = tmp_path / "opt" / "rocm"
    (install_root / "lib" / "rocprofiler-compute").mkdir(parents=True)
    monkeypatch.setattr(inject_roctx_loader, "_install_prefix", lambda: install_root)
    monkeypatch.setattr(
        inject_roctx_loader, "torch_version", lambda: _FAKE_TORCH_VERSION
    )

    with pytest.raises(inject_roctx_loader.UnsupportedTorchVersionError) as raised:
        inject_roctx_loader.load()

    assert raised.value.supported_torch_versions == ()
    assert "none" in str(raised.value)


def test_load_selects_the_matching_prebuilt_artifact(monkeypatch, tmp_path):
    install_root = _install_tree_with_artifacts(tmp_path, "2.8.0", _FAKE_TORCH_VERSION)
    monkeypatch.setattr(inject_roctx_loader, "_install_prefix", lambda: install_root)
    monkeypatch.setattr(
        inject_roctx_loader, "torch_version", lambda: _FAKE_TORCH_VERSION
    )
    sentinel = object()
    loaded_paths = []

    def _import_module(_name, path):
        loaded_paths.append(path)
        return sentinel

    monkeypatch.setattr(inject_roctx_loader, "_import_module_from_path", _import_module)

    result = inject_roctx_loader.load()
    assert result.module is sentinel
    assert result.tier == inject_roctx_loader.TIER_PREBUILT
    assert loaded_paths == [
        install_root
        / "lib"
        / "rocprofiler-compute"
        / f"torch_trace_collector-{_FAKE_TORCH_VERSION}.so"
    ]


def test_load_raises_import_error_when_matching_artifact_fails(
    monkeypatch, tmp_path
):
    install_root = _install_tree_with_artifacts(tmp_path, _FAKE_TORCH_VERSION)
    monkeypatch.setattr(inject_roctx_loader, "_install_prefix", lambda: install_root)
    monkeypatch.setattr(
        inject_roctx_loader, "torch_version", lambda: _FAKE_TORCH_VERSION
    )
    monkeypatch.setattr(
        inject_roctx_loader,
        "_import_module_from_path",
        lambda _n, _p: (_ for _ in ()).throw(ImportError("bad so")),
    )

    with pytest.raises(ImportError, match="bad so"):
        inject_roctx_loader.load()


def test_load_returns_independent_diagnostics(monkeypatch, tmp_path):
    install_root = _install_tree_with_artifacts(tmp_path, _FAKE_TORCH_VERSION)
    monkeypatch.setattr(inject_roctx_loader, "_install_prefix", lambda: install_root)
    monkeypatch.setattr(
        inject_roctx_loader, "torch_version", lambda: _FAKE_TORCH_VERSION
    )
    monkeypatch.setattr(
        inject_roctx_loader, "_import_module_from_path", lambda _n, _p: object()
    )

    first = inject_roctx_loader.load().diagnostics
    second = inject_roctx_loader.load().diagnostics
    assert first is not second


def test_safe_log_appends_to_provided_diagnostics():
    diagnostics = []
    inject_roctx_loader._safe_log("log", "tier A skipped", diagnostics)
    inject_roctx_loader._safe_log("warning", "tier B failed", diagnostics)
    assert [lvl for lvl, _ in diagnostics] == ["log", "warning"]
    assert [msg for _, msg in diagnostics] == ["tier A skipped", "tier B failed"]


def test_format_load_diagnostic_trail_handles_empty():
    assert inject_roctx_loader.format_load_diagnostic_trail([]) == ""


def test_format_load_diagnostic_trail_caps_lines():
    trail = [("log", f"line {i}") for i in range(100)]
    rendered = inject_roctx_loader.format_load_diagnostic_trail(trail, max_lines=12)
    lines = rendered.splitlines()
    assert len(lines) == 12
    assert "line 99" in rendered
    assert "line 0" not in rendered


def test_format_load_diagnostic_trail_includes_level_per_line():
    trail = [("log", "skipped tier A"), ("warning", "tier B failed")]
    rendered = inject_roctx_loader.format_load_diagnostic_trail(trail)
    assert "[log]" in rendered
    assert "[warning]" in rendered
    assert "skipped tier A" in rendered
    assert "tier B failed" in rendered


# ---------------------------------------------------------------------------
# Python fallback integration
# ---------------------------------------------------------------------------


def test_python_fallback_path_still_works_without_so(monkeypatch):
    """With the loader returning ``None``, the Python fallback still works."""
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("torch not importable")
    monkeypatch.setattr(inject_roctx_loader, "load", lambda **kw: None)
    if "utils.inject_roctx" in sys.modules:
        del sys.modules["utils.inject_roctx"]
    importlib.import_module("utils.inject_roctx")
    try:
        from utils.inject_roctx import core
        from utils.inject_roctx._backends import torch as _torch_backend

        assert _torch_backend.using_c_tier() is False
        assert _torch_backend.dump_torch_trace_stats() is None

        pushed = []
        original_io = core.get_python_tier_io()
        core.set_python_tier_io(push=pushed.append, pop=lambda: None)
        try:
            core._push_scope("py.tier.test", "#1@test:1")
            core._pop_scope()
        finally:
            core.set_python_tier_io(*original_io)
        assert pushed, "Python-tier rangePush was not invoked in fallback mode"
    finally:
        sys.modules.pop("utils.inject_roctx", None)


def test_import_does_not_apply_global_patches(monkeypatch):
    """Importing ``utils.inject_roctx`` does not patch PyTorch."""
    monkeypatch.setattr(inject_roctx_loader, "load", lambda **kw: None)
    if "utils.inject_roctx" in sys.modules:
        del sys.modules["utils.inject_roctx"]

    try:
        import torch  # noqa: F401
    except Exception:
        pytest.skip("torch not importable")

    import torch as _torch

    pre = {"compile": getattr(_torch, "compile", None)}
    importlib.import_module("utils.inject_roctx")
    try:
        post = {"compile": getattr(_torch, "compile", None)}
        from utils.inject_roctx import core
        from utils.inject_roctx._backends import torch as _torch_backend

        assert hasattr(core, "install_global_wraps")
        assert hasattr(_torch_backend, "using_c_tier")
        assert post["compile"] is pre["compile"]
    finally:
        sys.modules.pop("utils.inject_roctx", None)
