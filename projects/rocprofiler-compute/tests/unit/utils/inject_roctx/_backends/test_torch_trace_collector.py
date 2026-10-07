# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for the generic Torch collector ctypes loader. No GPU."""

import builtins
import ctypes
import os
import sys
from types import SimpleNamespace

import pytest

from utils.inject_roctx._backends import torch_trace_collector as loader


class FakeTorchVersion:
    """Stand-in for the release components used from torch.torch_version."""

    def __init__(self, version):
        self.release = tuple(int(part) for part in version.split(".")[:2])


class FakeNativeFunction:
    def __init__(self, return_value=0):
        self.return_value = return_value
        self.error = None
        self.argtypes = None
        self.restype = None
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        if self.error is not None:
            raise self.error
        return self.return_value


def make_native_library(revision=2):
    return SimpleNamespace(
        torch_trace_collector_abi_revision=FakeNativeFunction(revision),
        torch_trace_collector_install=FakeNativeFunction(),
        torch_trace_collector_push_launcher_tid=FakeNativeFunction(),
        torch_trace_collector_pop_launcher_tid=FakeNativeFunction(),
    )


def stub_torch(monkeypatch, version="2.13.0+cpu"):
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(__file__="/opt/fake/torch/__init__.py", __version__=version),
    )
    monkeypatch.setitem(
        sys.modules, "torch.torch_version", SimpleNamespace(Version=FakeTorchVersion)
    )


@pytest.fixture(autouse=True)
def reset_loader_state(monkeypatch):
    monkeypatch.setattr(loader, "_collector_library", None)
    monkeypatch.setattr(loader, "_torch_cpu_library", None)
    monkeypatch.setattr(loader, "console_log", lambda *_args: None)
    warnings = []
    monkeypatch.setattr(
        loader, "console_warning", lambda _category, message: warnings.append(message)
    )
    return warnings


@pytest.fixture
def native_environment(monkeypatch, tmp_path):
    stub_torch(monkeypatch)
    collector_path = tmp_path / "torch_trace_collector.so"
    library = make_native_library()
    torch_library = object()
    loads = []
    monkeypatch.setattr(loader, "_find_collector", lambda: collector_path)

    def load_library(path, mode):
        loads.append((path, mode))
        return library if path == str(collector_path) else torch_library

    monkeypatch.setattr(loader.ctypes, "CDLL", load_library)
    return SimpleNamespace(
        collector_path=collector_path,
        library=library,
        torch_library=torch_library,
        loads=loads,
    )


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("2.13.0+cpu", "2.13"),
        ("2.13.1+rocm10.2.0a20260922", "2.13"),
        ("2.14.0.dev20260928+rocm10.2.0a20260928", "2.14"),
        ("2.12.1", "2.12"),
    ],
)
def test_workload_torch_version_returns_major_minor(monkeypatch, version, expected):
    stub_torch(monkeypatch, version)
    assert loader._workload_torch_version() == expected


@pytest.mark.parametrize("version", ["not-a-version", "2"])
def test_unparsable_version_uses_fallback(monkeypatch, reset_loader_state, version):
    stub_torch(monkeypatch, version)
    assert not loader.install()
    assert "Could not determine the PyTorch version" in reset_loader_state[0]


def test_missing_torch_uses_fallback(monkeypatch, reset_loader_state):
    real_import = builtins.__import__

    def fail_torch_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch missing")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fail_torch_import)
    assert not loader.install()
    assert "torch missing" in reset_loader_state[0]


@pytest.mark.parametrize("libdir", ["lib", "lib64"])
def test_find_collector_uses_installed_generic_artifact(monkeypatch, tmp_path, libdir):
    package_root = tmp_path / "libexec" / "rocprofiler-compute"
    package_root.mkdir(parents=True)
    artifact_dir = tmp_path / libdir / "rocprofiler-compute"
    artifact_dir.mkdir(parents=True)
    artifact = artifact_dir / "torch_trace_collector.so"
    artifact.write_bytes(b"stub")
    monkeypatch.setattr(loader, "_PACKAGE_ROOT", package_root)

    assert loader._find_collector() == artifact.resolve()


def test_find_collector_uses_source_build_artifact(monkeypatch, tmp_path):
    package_root = tmp_path / "project" / "src"
    artifact = package_root / "lib" / "_build" / "lib" / "torch_trace_collector.so"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"stub")
    monkeypatch.setattr(loader, "_PACKAGE_ROOT", package_root)

    assert loader._find_collector() == artifact.resolve()


@pytest.mark.parametrize("version", ["2.12.1", "2.15.0", "3.13.0"])
def test_unsupported_torch_is_rejected_before_lookup_or_dlopen(
    monkeypatch, reset_loader_state, version
):
    stub_torch(monkeypatch, version)
    monkeypatch.setattr(
        loader, "_find_collector", lambda: pytest.fail("unexpected collector lookup")
    )
    monkeypatch.setattr(
        loader.ctypes, "CDLL", lambda *_a, **_k: pytest.fail("unexpected dlopen")
    )

    assert not loader.install()
    assert "Supported versions: 2.13, 2.14" in reset_loader_state[0]


@pytest.mark.parametrize("version", ["2.13.0+cpu", "2.14.1+rocm10.2"])
def test_install_promotes_real_torch_then_loads_generic_collector_once(
    monkeypatch, native_environment, version
):
    stub_torch(monkeypatch, version)
    assert loader.install()
    assert loader.install()

    assert native_environment.loads == [
        (
            "/opt/fake/torch/lib/libtorch_cpu.so",
            os.RTLD_GLOBAL | os.RTLD_LAZY | os.RTLD_NODELETE,
        ),
        (
            str(native_environment.collector_path),
            ctypes.RTLD_LOCAL | os.RTLD_NOW | os.RTLD_NODELETE,
        ),
    ]
    assert loader._torch_cpu_library is native_environment.torch_library
    assert loader._collector_library is native_environment.library
    assert native_environment.library.torch_trace_collector_install.calls == [()]
    signatures = {
        "abi_revision": ([], ctypes.c_uint32),
        "install": ([], ctypes.c_int32),
        "push_launcher_tid": ([ctypes.c_uint64], ctypes.c_int32),
        "pop_launcher_tid": ([], ctypes.c_int32),
    }
    for name, (argtypes, restype) in signatures.items():
        function = getattr(native_environment.library, f"torch_trace_collector_{name}")
        assert function.argtypes == argtypes
        assert function.restype is restype


def test_missing_collector_uses_fallback_before_dlopen(
    monkeypatch, native_environment, reset_loader_state
):
    monkeypatch.setattr(loader, "_find_collector", lambda: None)
    assert not loader.install()
    assert native_environment.loads == []
    assert "was not built" in reset_loader_state[0]


@pytest.mark.parametrize("failed_path", ["torch", "collector"])
def test_dlopen_failure_warns_and_uses_fallback(
    monkeypatch, native_environment, reset_loader_state, failed_path
):
    load_library = loader.ctypes.CDLL

    def fail_load(path, mode):
        is_collector = path == str(native_environment.collector_path)
        if is_collector == (failed_path == "collector"):
            raise OSError("missing symbol")
        return load_library(path, mode)

    monkeypatch.setattr(loader.ctypes, "CDLL", fail_load)
    assert not loader.install()
    assert loader._collector_library is None
    assert "missing symbol" in reset_loader_state[0]
    if failed_path == "collector":
        assert loader._torch_cpu_library is native_environment.torch_library


def test_wrong_revision_is_rejected_before_install(
    native_environment, reset_loader_state
):
    native_environment.library.torch_trace_collector_abi_revision.return_value = 1
    assert not loader.install()
    assert native_environment.library.torch_trace_collector_install.calls == []
    assert loader._collector_library is None
    assert "interface revision 1; expected 2" in reset_loader_state[0]


@pytest.mark.parametrize(
    "missing_symbol",
    ["abi_revision", "install", "push_launcher_tid", "pop_launcher_tid"],
)
def test_incomplete_interface_is_rejected_before_install(
    native_environment, reset_loader_state, missing_symbol
):
    install_function = native_environment.library.torch_trace_collector_install
    delattr(native_environment.library, f"torch_trace_collector_{missing_symbol}")
    assert not loader.install()
    assert install_function.calls == []
    assert loader._collector_library is None
    assert missing_symbol in reset_loader_state[0]


def test_failed_install_can_retry_without_promoting_torch_again(
    native_environment, reset_loader_state
):
    install_function = native_environment.library.torch_trace_collector_install
    install_function.return_value = 1
    assert not loader.install()
    assert loader._collector_library is None
    assert "torch_trace_collector_install failed" in reset_loader_state[0]

    install_function.return_value = 0
    assert loader.install()
    assert len(native_environment.loads) == 3
    assert install_function.calls == [(), ()]


def test_launcher_calls_return_false_without_a_collector():
    assert not loader.push_launcher_tid()
    assert not loader.pop_launcher_tid()


def test_launcher_calls_use_native_thread_id(monkeypatch, native_environment):
    assert loader.install()
    monkeypatch.setattr(loader.threading, "get_native_id", lambda: 123456)

    assert loader.push_launcher_tid()
    assert loader.pop_launcher_tid()
    assert native_environment.library.torch_trace_collector_push_launcher_tid.calls == [
        (123456,)
    ]
    assert native_environment.library.torch_trace_collector_pop_launcher_tid.calls == [
        ()
    ]


@pytest.mark.parametrize("operation", ["push_launcher_tid", "pop_launcher_tid"])
@pytest.mark.parametrize("raises", [False, True])
def test_launcher_failures_do_not_escape(native_environment, operation, raises):
    assert loader.install()
    function = getattr(native_environment.library, f"torch_trace_collector_{operation}")
    if raises:
        function.error = RuntimeError("launcher failure")
    else:
        function.return_value = 1

    assert not getattr(loader, operation)()
