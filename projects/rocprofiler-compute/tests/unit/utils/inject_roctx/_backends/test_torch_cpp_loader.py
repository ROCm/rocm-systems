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


def test_torch_version_exits_when_torch_is_missing(monkeypatch):
    """``torch_version()`` exits when torch is not importable."""
    import builtins

    real_import = builtins.__import__

    def _import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch" or (isinstance(name, str) and name.startswith("torch.")):
            raise ImportError("torch missing")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _import)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    with pytest.raises(SystemExit) as raised:
        inject_roctx_loader.torch_version()
    assert raised.value.code == 1


# ---------------------------------------------------------------------------
# Artifact discovery
# ---------------------------------------------------------------------------


def collector_artifact_dir(tmp_path: Path, *versions: str) -> Path:
    """Create ``tmp_path/lib/rocprofiler-compute/*.so`` and return that directory."""
    artifact_dir = tmp_path / "lib" / "rocprofiler-compute"
    artifact_dir.mkdir(parents=True)
    for version in versions:
        (artifact_dir / f"torch_trace_collector-{version}.so").write_bytes(b"stub")
    return artifact_dir


def test_list_collector_artifacts_returns_paths_and_versions(tmp_path, monkeypatch):
    artifact_dir = collector_artifact_dir(tmp_path, "2.8.0", "2.9.0", "2.8.0")
    monkeypatch.setattr(inject_roctx_loader, "_ARTIFACT_DIR", artifact_dir)
    assert inject_roctx_loader.list_collector_artifacts() == {
        "2.8.0": artifact_dir / "torch_trace_collector-2.8.0.so",
        "2.9.0": artifact_dir / "torch_trace_collector-2.9.0.so",
    }


def test_list_collector_artifacts_is_empty_without_artifacts(tmp_path, monkeypatch):
    artifact_dir = tmp_path / "lib" / "rocprofiler-compute"
    artifact_dir.mkdir(parents=True)
    monkeypatch.setattr(inject_roctx_loader, "_ARTIFACT_DIR", artifact_dir)
    assert inject_roctx_loader.list_collector_artifacts() == {}


def test_list_collector_artifacts_skips_legacy_names(tmp_path, monkeypatch):
    artifact_dir = collector_artifact_dir(tmp_path, "2.8.0", _FAKE_TORCH_VERSION)
    (
        artifact_dir / "torch_trace_collector-py3.12_torch2.13.0_srcdeadbeef.so"
    ).write_bytes(b"legacy")
    monkeypatch.setattr(inject_roctx_loader, "_ARTIFACT_DIR", artifact_dir)
    assert inject_roctx_loader.list_collector_artifacts() == {
        "2.8.0": artifact_dir / "torch_trace_collector-2.8.0.so",
        _FAKE_TORCH_VERSION: (
            artifact_dir / f"torch_trace_collector-{_FAKE_TORCH_VERSION}.so"
        ),
    }


def test_list_collector_artifacts_uses_build_lib_when_install_dir_missing(
    tmp_path, monkeypatch
):
    package_root = tmp_path / "src"
    build_dir = package_root / "lib" / "_build" / "lib"
    build_dir.mkdir(parents=True)
    so_path = build_dir / "torch_trace_collector-2.13.0.so"
    so_path.write_bytes(b"stub")
    monkeypatch.setattr(
        inject_roctx_loader, "_ARTIFACT_DIR", tmp_path / "missing-install"
    )
    monkeypatch.setattr(inject_roctx_loader, "_PACKAGE_ROOT", package_root)
    assert inject_roctx_loader.list_collector_artifacts() == {"2.13.0": so_path}


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------


def test_load_raises_when_torch_version_is_unsupported(monkeypatch, tmp_path):
    artifact_dir = collector_artifact_dir(tmp_path, "2.8.0")
    monkeypatch.setattr(inject_roctx_loader, "_ARTIFACT_DIR", artifact_dir)
    monkeypatch.setattr(
        inject_roctx_loader, "torch_version", lambda: _FAKE_TORCH_VERSION
    )

    with pytest.raises(inject_roctx_loader.UnsupportedTorchVersionError) as raised:
        inject_roctx_loader.load()

    error = raised.value
    assert _FAKE_TORCH_VERSION in str(error)
    assert "2.8.0" in str(error)


def test_load_raises_when_no_artifacts_are_discovered(monkeypatch, tmp_path):
    artifact_dir = tmp_path / "lib" / "rocprofiler-compute"
    artifact_dir.mkdir(parents=True)
    monkeypatch.setattr(inject_roctx_loader, "_ARTIFACT_DIR", artifact_dir)
    monkeypatch.setattr(
        inject_roctx_loader, "torch_version", lambda: _FAKE_TORCH_VERSION
    )

    with pytest.raises(inject_roctx_loader.UnsupportedTorchVersionError) as raised:
        inject_roctx_loader.load()

    assert _FAKE_TORCH_VERSION in str(raised.value)


def test_load_exits_when_matching_artifact_fails(monkeypatch, tmp_path):
    artifact_dir = collector_artifact_dir(tmp_path, _FAKE_TORCH_VERSION)
    monkeypatch.setattr(inject_roctx_loader, "_ARTIFACT_DIR", artifact_dir)
    monkeypatch.setattr(
        inject_roctx_loader, "torch_version", lambda: _FAKE_TORCH_VERSION
    )

    with pytest.raises(SystemExit) as raised:
        inject_roctx_loader.load()
    assert raised.value.code == 1


# ---------------------------------------------------------------------------
# Python fallback integration
# ---------------------------------------------------------------------------


def test_python_fallback_path_still_works_without_so(monkeypatch):
    """With the loader returning ``None``, the Python fallback still works."""
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("torch not importable")
    monkeypatch.setattr(inject_roctx_loader, "load", lambda: None)
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
    monkeypatch.setattr(inject_roctx_loader, "load", lambda: None)
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
