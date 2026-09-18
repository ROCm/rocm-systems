# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for utils.inject_roctx._backends.torch_cpp_loader. No GPU."""

import sys
import types
from pathlib import Path

import common  # noqa: F401
import pytest
from packaging.version import Version

from utils.inject_roctx._backends import torch_cpp_loader as inject_roctx_loader

_FAKE_TORCH_VERSION = "2.9"
_FAKE_ABI = "cpython-312-x86_64-linux-gnu"
_SUPPORTED = ("2.13", "2.14")


# ---------------------------------------------------------------------------
# torch_version
# ---------------------------------------------------------------------------


def stub_torch(monkeypatch, version: str) -> None:
    """Install a torch stub exposing ``__version__`` and ``torch_version``."""
    monkeypatch.setitem(
        sys.modules, "torch", types.SimpleNamespace(__version__=version)
    )
    monkeypatch.setitem(
        sys.modules,
        "torch.torch_version",
        types.SimpleNamespace(Version=Version),
    )


def test_torch_version_drops_the_local_build_segment(monkeypatch):
    """``torch_version()`` ignores a local ``+...`` build suffix."""
    stub_torch(monkeypatch, "2.9.0+rocm7.1")
    assert inject_roctx_loader.torch_version() == "2.9"


def test_torch_version_drops_a_prerelease_marker(monkeypatch):
    """``torch_version()`` ignores a prerelease marker such as ``a0``."""
    stub_torch(monkeypatch, "2.9.0a0+rocm7.1")
    assert inject_roctx_loader.torch_version() == "2.9"


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


def write_collector_so(directory: Path, abi: str = _FAKE_ABI) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"torch_trace_collector.{abi}.so"
    path.write_bytes(b"stub")
    return path


def installed_package_root(tmp_path: Path, libdir: str, with_collector: bool):
    """Return ``(package_root, artifact_dir)`` for an install-prefix layout."""
    package_root = tmp_path / "libexec" / "rocprofiler-compute"
    package_root.mkdir(parents=True)
    artifact_dir = tmp_path / libdir / "rocprofiler-compute"
    if with_collector:
        write_collector_so(artifact_dir)
    return package_root, artifact_dir


def test_list_collector_artifacts_returns_the_installed_artifact(tmp_path, monkeypatch):
    package_root, artifact_dir = installed_package_root(tmp_path, "lib", True)
    monkeypatch.setattr(inject_roctx_loader, "_PACKAGE_ROOT", package_root)
    assert inject_roctx_loader.list_collector_artifacts() == [
        artifact_dir / f"torch_trace_collector.{_FAKE_ABI}.so"
    ]


def test_list_collector_artifacts_is_empty_without_artifacts(tmp_path, monkeypatch):
    package_root, _artifact_dir = installed_package_root(tmp_path, "lib", False)
    monkeypatch.setattr(inject_roctx_loader, "_PACKAGE_ROOT", package_root)
    assert inject_roctx_loader.list_collector_artifacts() == []


def test_list_collector_artifacts_skips_legacy_names(tmp_path, monkeypatch):
    """The per-version artifact name predates the stub headers."""
    package_root, artifact_dir = installed_package_root(tmp_path, "lib", True)
    (artifact_dir / f"torch_trace_collector-2.13.{_FAKE_ABI}.so").write_bytes(b"legacy")
    monkeypatch.setattr(inject_roctx_loader, "_PACKAGE_ROOT", package_root)
    assert inject_roctx_loader.list_collector_artifacts() == [
        artifact_dir / f"torch_trace_collector.{_FAKE_ABI}.so"
    ]


def test_list_collector_artifacts_finds_lib64_install(tmp_path, monkeypatch):
    package_root, artifact_dir = installed_package_root(tmp_path, "lib64", True)
    monkeypatch.setattr(inject_roctx_loader, "_PACKAGE_ROOT", package_root)
    assert inject_roctx_loader.list_collector_artifacts() == [
        artifact_dir / f"torch_trace_collector.{_FAKE_ABI}.so"
    ]


def test_list_collector_artifacts_finds_lib64_when_lib_has_no_collector(
    tmp_path, monkeypatch
):
    package_root, artifact_dir = installed_package_root(tmp_path, "lib64", True)
    lib_install_dir = tmp_path / "lib" / "rocprofiler-compute"
    lib_install_dir.mkdir(parents=True)
    (lib_install_dir / "librocprofiler-compute-tool.so").write_bytes(b"tool")
    monkeypatch.setattr(inject_roctx_loader, "_PACKAGE_ROOT", package_root)
    assert inject_roctx_loader.list_collector_artifacts() == [
        artifact_dir / f"torch_trace_collector.{_FAKE_ABI}.so"
    ]


def test_list_collector_artifacts_uses_src_lib_build_when_install_dir_missing(
    tmp_path, monkeypatch
):
    package_root = tmp_path / "src"
    so_path = write_collector_so(package_root / "lib" / "_build" / "lib")
    monkeypatch.setattr(inject_roctx_loader, "_PACKAGE_ROOT", package_root)
    assert inject_roctx_loader.list_collector_artifacts() == [so_path]


def test_list_collector_artifacts_uses_root_build_lib(tmp_path, monkeypatch):
    package_root = tmp_path / "src"
    package_root.mkdir(parents=True)
    so_path = write_collector_so(tmp_path / "build" / "lib")
    monkeypatch.setattr(inject_roctx_loader, "_PACKAGE_ROOT", package_root)
    assert inject_roctx_loader.list_collector_artifacts() == [so_path]


def test_list_collector_artifacts_finds_source_build_when_prefix_lib_is_empty(
    tmp_path, monkeypatch
):
    package_root = tmp_path / "src"
    (tmp_path / "lib" / "rocprofiler-compute").mkdir(parents=True)
    so_path = write_collector_so(package_root / "lib" / "_build" / "lib")
    monkeypatch.setattr(inject_roctx_loader, "_PACKAGE_ROOT", package_root)
    assert inject_roctx_loader.list_collector_artifacts() == [so_path]


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------


def stub_extension(monkeypatch, supported=_SUPPORTED) -> types.SimpleNamespace:
    """Make ``load()`` return a stub instead of dlopening the artifact."""
    module = types.SimpleNamespace(supported_torch_versions=supported)
    monkeypatch.setattr(inject_roctx_loader, "publish_torch_symbols", lambda: None)

    class _Loader:
        @staticmethod
        def exec_module(_module):
            return None

    monkeypatch.setattr(
        inject_roctx_loader.importlib.util,
        "spec_from_file_location",
        lambda *_args, **_kwargs: types.SimpleNamespace(loader=_Loader()),
    )
    monkeypatch.setattr(
        inject_roctx_loader.importlib.util,
        "module_from_spec",
        lambda _spec: module,
    )
    return module


def test_load_returns_the_extension_for_a_supported_version(monkeypatch, tmp_path):
    package_root, _artifact_dir = installed_package_root(tmp_path, "lib", True)
    monkeypatch.setattr(inject_roctx_loader, "_PACKAGE_ROOT", package_root)
    monkeypatch.setattr(inject_roctx_loader, "torch_version", lambda: "2.13")
    module = stub_extension(monkeypatch)

    assert inject_roctx_loader.load() is module


def test_load_raises_when_torch_version_is_unsupported(monkeypatch, tmp_path):
    package_root, _artifact_dir = installed_package_root(tmp_path, "lib", True)
    monkeypatch.setattr(inject_roctx_loader, "_PACKAGE_ROOT", package_root)
    monkeypatch.setattr(
        inject_roctx_loader, "torch_version", lambda: _FAKE_TORCH_VERSION
    )
    stub_extension(monkeypatch)

    with pytest.raises(inject_roctx_loader.UnsupportedTorchVersionError) as raised:
        inject_roctx_loader.load()

    message = str(raised.value)
    assert _FAKE_TORCH_VERSION in message
    assert "2.13" in message


def test_load_raises_a_distinct_error_when_nothing_was_built(monkeypatch, tmp_path):
    """A build with no collector is reported apart from an unsupported version."""
    package_root, _artifact_dir = installed_package_root(tmp_path, "lib", False)
    monkeypatch.setattr(inject_roctx_loader, "_PACKAGE_ROOT", package_root)
    monkeypatch.setattr(
        inject_roctx_loader, "torch_version", lambda: _FAKE_TORCH_VERSION
    )

    with pytest.raises(inject_roctx_loader.CollectorNotBuiltError) as raised:
        inject_roctx_loader.load()

    message = str(raised.value)
    assert _FAKE_TORCH_VERSION in message
    assert "Supported PyTorch versions" not in message


def test_load_raises_when_publishing_torch_symbols_fails(monkeypatch, tmp_path):
    """A dlopen failure falls back to the Python tier instead of exiting."""
    package_root, _artifact_dir = installed_package_root(tmp_path, "lib", True)
    monkeypatch.setattr(inject_roctx_loader, "_PACKAGE_ROOT", package_root)
    monkeypatch.setattr(inject_roctx_loader, "torch_version", lambda: "2.13")

    def _fail() -> None:
        raise OSError("libtorch_cpu.so: cannot open shared object file")

    monkeypatch.setattr(inject_roctx_loader, "publish_torch_symbols", _fail)

    with pytest.raises(inject_roctx_loader.CollectorLoadError) as raised:
        inject_roctx_loader.load()

    assert isinstance(raised.value, inject_roctx_loader.CollectorUnavailableError)
    assert "libtorch_cpu.so" in str(raised.value)


def test_load_raises_when_the_extension_fails_to_load(monkeypatch, tmp_path):
    package_root, _artifact_dir = installed_package_root(tmp_path, "lib", True)
    monkeypatch.setattr(inject_roctx_loader, "_PACKAGE_ROOT", package_root)
    monkeypatch.setattr(inject_roctx_loader, "torch_version", lambda: "2.13")
    monkeypatch.setattr(inject_roctx_loader, "publish_torch_symbols", lambda: None)

    with pytest.raises(inject_roctx_loader.CollectorLoadError):
        inject_roctx_loader.load()
