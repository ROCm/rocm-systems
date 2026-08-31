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


@pytest.fixture(autouse=True)
def clear_cmake_binary_dir(monkeypatch):
    monkeypatch.delenv("CMAKE_BINARY_DIR", raising=False)


def write_collector_so(directory: Path, version: str, abi: str = _FAKE_ABI) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"torch_trace_collector-{version}.{abi}.so"
    path.write_bytes(b"stub")
    return path


def installed_package_root(tmp_path: Path, libdir: str, *versions: str):
    """Return ``(package_root, artifact_dir)`` for an install-prefix layout."""
    package_root = tmp_path / "libexec" / "rocprofiler-compute"
    package_root.mkdir(parents=True)
    artifact_dir = tmp_path / libdir / "rocprofiler-compute"
    for version in versions:
        write_collector_so(artifact_dir, version)
    return package_root, artifact_dir


def test_list_collector_artifacts_returns_paths_and_versions(tmp_path, monkeypatch):
    package_root, artifact_dir = installed_package_root(
        tmp_path, "lib", "2.8", "2.9", "2.8"
    )
    monkeypatch.setattr(inject_roctx_loader, "_PACKAGE_ROOT", package_root)
    assert inject_roctx_loader.list_collector_artifacts() == {
        "2.8": artifact_dir / f"torch_trace_collector-2.8.{_FAKE_ABI}.so",
        "2.9": artifact_dir / f"torch_trace_collector-2.9.{_FAKE_ABI}.so",
    }


def test_list_collector_artifacts_is_empty_without_artifacts(tmp_path, monkeypatch):
    package_root, _artifact_dir = installed_package_root(tmp_path, "lib")
    monkeypatch.setattr(inject_roctx_loader, "_PACKAGE_ROOT", package_root)
    assert inject_roctx_loader.list_collector_artifacts() == {}


def test_list_collector_artifacts_skips_legacy_names(tmp_path, monkeypatch):
    package_root, artifact_dir = installed_package_root(
        tmp_path, "lib", "2.8", _FAKE_TORCH_VERSION
    )
    (
        artifact_dir / "torch_trace_collector-py3.12_torch2.13.0_srcdeadbeef.so"
    ).write_bytes(b"legacy")
    monkeypatch.setattr(inject_roctx_loader, "_PACKAGE_ROOT", package_root)
    assert inject_roctx_loader.list_collector_artifacts() == {
        "2.8": artifact_dir / f"torch_trace_collector-2.8.{_FAKE_ABI}.so",
        _FAKE_TORCH_VERSION: (
            artifact_dir / f"torch_trace_collector-{_FAKE_TORCH_VERSION}.{_FAKE_ABI}.so"
        ),
    }


def test_list_collector_artifacts_finds_lib64_install(tmp_path, monkeypatch):
    package_root, artifact_dir = installed_package_root(tmp_path, "lib64", "2.13")
    monkeypatch.setattr(inject_roctx_loader, "_PACKAGE_ROOT", package_root)
    so_path = artifact_dir / f"torch_trace_collector-2.13.{_FAKE_ABI}.so"
    assert inject_roctx_loader.list_collector_artifacts() == {"2.13": so_path}


def test_list_collector_artifacts_finds_lib64_when_lib_has_no_collector(
    tmp_path, monkeypatch
):
    package_root, artifact_dir = installed_package_root(tmp_path, "lib64", "2.13")
    lib_install_dir = tmp_path / "lib" / "rocprofiler-compute"
    lib_install_dir.mkdir(parents=True)
    (lib_install_dir / "librocprofiler-compute-tool.so").write_bytes(b"tool")
    monkeypatch.setattr(inject_roctx_loader, "_PACKAGE_ROOT", package_root)
    so_path = artifact_dir / f"torch_trace_collector-2.13.{_FAKE_ABI}.so"
    assert inject_roctx_loader.list_collector_artifacts() == {"2.13": so_path}


def test_list_collector_artifacts_uses_src_lib_build_when_install_dir_missing(
    tmp_path, monkeypatch
):
    package_root = tmp_path / "src"
    so_path = write_collector_so(package_root / "lib" / "_build" / "lib", "2.13")
    monkeypatch.setattr(inject_roctx_loader, "_PACKAGE_ROOT", package_root)
    assert inject_roctx_loader.list_collector_artifacts() == {"2.13": so_path}


def test_list_collector_artifacts_uses_root_build_lib(tmp_path, monkeypatch):
    package_root = tmp_path / "src"
    package_root.mkdir(parents=True)
    so_path = write_collector_so(tmp_path / "build" / "lib", "2.13")
    monkeypatch.setattr(inject_roctx_loader, "_PACKAGE_ROOT", package_root)
    assert inject_roctx_loader.list_collector_artifacts() == {"2.13": so_path}


def test_list_collector_artifacts_uses_cmake_binary_dir_lib(tmp_path, monkeypatch):
    package_root = tmp_path / "src"
    package_root.mkdir(parents=True)
    cmake_binary_dir = tmp_path / "out"
    so_path = write_collector_so(cmake_binary_dir / "lib", "2.13")
    monkeypatch.setenv("CMAKE_BINARY_DIR", str(cmake_binary_dir))
    monkeypatch.setattr(inject_roctx_loader, "_PACKAGE_ROOT", package_root)
    assert inject_roctx_loader.list_collector_artifacts() == {"2.13": so_path}


def test_list_collector_artifacts_finds_source_build_when_prefix_lib_is_empty(
    tmp_path, monkeypatch
):
    package_root = tmp_path / "src"
    (tmp_path / "lib" / "rocprofiler-compute").mkdir(parents=True)
    so_path = write_collector_so(package_root / "lib" / "_build" / "lib", "2.13")
    monkeypatch.setattr(inject_roctx_loader, "_PACKAGE_ROOT", package_root)
    assert inject_roctx_loader.list_collector_artifacts() == {"2.13": so_path}


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------


def test_load_raises_when_torch_version_is_unsupported(monkeypatch, tmp_path):
    package_root, _artifact_dir = installed_package_root(tmp_path, "lib", "2.8")
    monkeypatch.setattr(inject_roctx_loader, "_PACKAGE_ROOT", package_root)
    monkeypatch.setattr(
        inject_roctx_loader, "torch_version", lambda: _FAKE_TORCH_VERSION
    )

    with pytest.raises(inject_roctx_loader.UnsupportedTorchVersionError) as raised:
        inject_roctx_loader.load()

    error = raised.value
    assert _FAKE_TORCH_VERSION in str(error)
    assert "2.8" in str(error)


def test_load_raises_a_distinct_error_when_nothing_was_built(monkeypatch, tmp_path):
    """A build with no collector is reported apart from an unsupported version."""
    package_root, _artifact_dir = installed_package_root(tmp_path, "lib")
    monkeypatch.setattr(inject_roctx_loader, "_PACKAGE_ROOT", package_root)
    monkeypatch.setattr(
        inject_roctx_loader, "torch_version", lambda: _FAKE_TORCH_VERSION
    )

    with pytest.raises(inject_roctx_loader.CollectorNotBuiltError) as raised:
        inject_roctx_loader.load()

    message = str(raised.value)
    assert _FAKE_TORCH_VERSION in message
    assert "Supported PyTorch versions" not in message


def test_load_exits_when_matching_artifact_fails(monkeypatch, tmp_path):
    package_root, _artifact_dir = installed_package_root(
        tmp_path, "lib", _FAKE_TORCH_VERSION
    )
    monkeypatch.setattr(inject_roctx_loader, "_PACKAGE_ROOT", package_root)
    monkeypatch.setattr(
        inject_roctx_loader, "torch_version", lambda: _FAKE_TORCH_VERSION
    )

    with pytest.raises(SystemExit) as raised:
        inject_roctx_loader.load()
    assert raised.value.code == 1
