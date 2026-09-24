# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for the generic Torch collector ctypes loader. No GPU."""

import builtins
import ctypes
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, List, Optional, Tuple

import pytest

from utils.inject_roctx._backends import torch_cpp_loader

CPU_213_VERSION = "2.13.0+cpu"
ROCM_214_VERSION = "2.14.0+rocm10.2.0a20260917"


class FakeTorchVersion:
    """Stand-in for torch.torch_version.Version."""

    def __init__(self, value: str) -> None:
        self.release = tuple(int(part) for part in value.split("+")[0].split("."))


class FakeNativeFunction:
    def __init__(
        self,
        return_value: int = 0,
        callback: Optional[Callable[..., int]] = None,
    ) -> None:
        self.return_value = return_value
        self.callback = callback
        self.argtypes = None
        self.restype = None
        self.calls: List[Tuple[object, ...]] = []

    def __call__(self, *args: object) -> int:
        self.calls.append(args)
        if self.callback is not None:
            return self.callback(*args)
        return self.return_value


def make_native_library(
    revision: int = 1,
    install_result: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        torch_trace_collector_abi_revision=FakeNativeFunction(revision),
        torch_trace_collector_install=FakeNativeFunction(install_result),
        torch_trace_collector_uninstall=FakeNativeFunction(),
        torch_trace_collector_is_installed=FakeNativeFunction(return_value=1),
        torch_trace_collector_push_user_scope=FakeNativeFunction(),
        torch_trace_collector_pop_user_scope=FakeNativeFunction(),
        torch_trace_collector_get_stats=FakeNativeFunction(),
    )


def populate_native_stats(stats_pointer: object) -> int:
    stats = ctypes.cast(
        stats_pointer,
        ctypes.POINTER(torch_cpp_loader._CollectorStats),
    ).contents
    assert stats.struct_size == ctypes.sizeof(torch_cpp_loader._CollectorStats)
    stats.installed = 1
    stats.pushes = 8
    stats.pops = 7
    stats.user_scope_pushes = 3
    stats.user_scope_pops = 2
    stats.user_scope_inherits = 1
    stats.snapshots_saved = 5
    stats.snapshots_consumed = 4
    stats.snapshots_dropped = 0
    stats.snapshots_overwritten = 1
    stats.callback_errors = 2
    stats.snapshots_pending = 1
    return 0


def stub_torch(monkeypatch: pytest.MonkeyPatch, version: str) -> None:
    torch_module = types.SimpleNamespace(
        __file__="/opt/fake/torch/__init__.py",
        __version__=version,
    )
    version_module = types.SimpleNamespace(Version=FakeTorchVersion)
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "torch.torch_version", version_module)


def write_collector_so(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "torch_trace_collector.so"
    path.write_bytes(b"stub")
    return path


def create_installed_package_layout(tmp_path: Path, libdir: str) -> Tuple[Path, Path]:
    package_root = tmp_path / "libexec" / "rocprofiler-compute"
    package_root.mkdir(parents=True)
    artifact_dir = tmp_path / libdir / "rocprofiler-compute"
    artifact_dir.mkdir(parents=True)
    return package_root, artifact_dir


@pytest.fixture(autouse=True)
def reset_loader_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch_cpp_loader, "_torch_cpu_library", None)


@pytest.mark.parametrize(
    ("torch_version", "expected"),
    [
        (CPU_213_VERSION, "2.13"),
        (ROCM_214_VERSION, "2.14"),
        ("2.13.0+rocm10.2.0a20260922", "2.13"),
        ("2.12.1", "2.12"),
    ],
)
def test_workload_torch_version_returns_major_minor(
    monkeypatch: pytest.MonkeyPatch,
    torch_version: str,
    expected: str,
) -> None:
    stub_torch(monkeypatch, torch_version)

    assert torch_cpp_loader._workload_torch_version() == expected


def test_workload_torch_version_fails_closed_when_version_is_unparsable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warnings = []
    stub_torch(monkeypatch, "not-a-version")
    monkeypatch.setattr(
        torch_cpp_loader,
        "console_warning",
        lambda _category, message: warnings.append(message),
    )

    assert torch_cpp_loader._workload_torch_version() == ""
    assert "Could not determine the PyTorch version" in warnings[0]


def test_workload_torch_version_fails_closed_when_torch_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warnings = []
    real_import = builtins.__import__

    def fail_torch_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch missing")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fail_torch_import)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr(
        torch_cpp_loader,
        "console_warning",
        lambda _category, message: warnings.append(message),
    )

    assert torch_cpp_loader._workload_torch_version() == ""
    assert "torch missing" in warnings[0]


@pytest.mark.parametrize("libdir", ["lib", "lib64"])
def test_discover_collector_uses_installed_generic_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    libdir: str,
) -> None:
    package_root, artifact_dir = create_installed_package_layout(tmp_path, libdir)
    write_collector_so(artifact_dir)
    monkeypatch.setattr(torch_cpp_loader, "_PACKAGE_ROOT", package_root)

    assert (
        torch_cpp_loader._discover_collector_artifact()
        == (artifact_dir / "torch_trace_collector.so").resolve()
    )


def test_discover_collector_ignores_versioned_and_soabi_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    package_root, artifact_dir = create_installed_package_layout(tmp_path, "lib")
    (artifact_dir / "torch_trace_collector-2.13.so").write_bytes(b"legacy")
    (
        artifact_dir / "torch_trace_collector.cpython-312-x86_64-linux-gnu.so"
    ).write_bytes(b"legacy")
    monkeypatch.setattr(torch_cpp_loader, "_PACKAGE_ROOT", package_root)

    assert torch_cpp_loader._discover_collector_artifact() is None


def test_promote_torch_cpu_reopens_workload_library_globally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_torch(monkeypatch, CPU_213_VERSION)
    loaded_libraries: List[Tuple[str, int]] = []
    fake_library = object()
    monkeypatch.setattr(
        torch_cpp_loader.ctypes,
        "CDLL",
        lambda path, mode: loaded_libraries.append((path, mode)) or fake_library,
    )

    assert torch_cpp_loader._promote_torch_cpu() is fake_library
    assert loaded_libraries == [
        (
            "/opt/fake/torch/lib/libtorch_cpu.so",
            torch_cpp_loader._TORCH_LIBRARY_LOAD_MODE,
        )
    ]


@pytest.mark.parametrize("torch_version", ["2.13", "2.14"])
def test_load_returns_plain_c_wrapper_for_supported_torch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    torch_version: str,
) -> None:
    collector_path = tmp_path / "torch_trace_collector.so"
    native_library = make_native_library()
    torch_library = object()
    load_order = []
    monkeypatch.setattr(
        torch_cpp_loader,
        "_workload_torch_version",
        lambda: torch_version,
    )
    monkeypatch.setattr(
        torch_cpp_loader,
        "_discover_collector_artifact",
        lambda: collector_path,
    )
    monkeypatch.setattr(
        torch_cpp_loader,
        "_promote_torch_cpu",
        lambda: load_order.append("torch") or torch_library,
    )
    monkeypatch.setattr(
        torch_cpp_loader.ctypes,
        "CDLL",
        lambda path, mode: load_order.append((path, mode)) or native_library,
    )
    monkeypatch.setattr(torch_cpp_loader, "console_log", lambda *_args: None)

    collector = torch_cpp_loader.load()

    assert isinstance(collector, torch_cpp_loader.TorchTraceCollector)
    assert load_order == [
        "torch",
        (str(collector_path), torch_cpp_loader._COLLECTOR_LOAD_MODE),
    ]
    assert torch_cpp_loader._torch_cpu_library is torch_library
    revision = native_library.torch_trace_collector_abi_revision
    assert revision.argtypes == []
    assert revision.restype is ctypes.c_uint32
    assert revision.calls == [()]


@pytest.mark.parametrize(
    "torch_version",
    ["2.12", "2.15", "3.0", ""],
)
def test_load_rejects_unsupported_torch_before_artifact_lookup(
    monkeypatch: pytest.MonkeyPatch,
    torch_version: str,
) -> None:
    monkeypatch.setattr(
        torch_cpp_loader,
        "_workload_torch_version",
        lambda: torch_version,
    )
    monkeypatch.setattr(
        torch_cpp_loader,
        "_discover_collector_artifact",
        lambda: pytest.fail("collector lookup must not run"),
    )

    with pytest.raises(torch_cpp_loader.UnsupportedTorchVersionError) as raised:
        torch_cpp_loader.load()
    assert "2.13, 2.14" in str(raised.value)


def test_load_reports_missing_generic_collector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        torch_cpp_loader,
        "_workload_torch_version",
        lambda: "2.13",
    )
    monkeypatch.setattr(torch_cpp_loader, "_discover_collector_artifact", lambda: None)

    with pytest.raises(torch_cpp_loader.CollectorNotBuiltError):
        torch_cpp_loader.load()


def test_load_wraps_collector_dlopen_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    collector_path = tmp_path / "torch_trace_collector.so"
    torch_library = object()
    monkeypatch.setattr(
        torch_cpp_loader,
        "_workload_torch_version",
        lambda: "2.13",
    )
    monkeypatch.setattr(
        torch_cpp_loader,
        "_discover_collector_artifact",
        lambda: collector_path,
    )
    monkeypatch.setattr(
        torch_cpp_loader,
        "_promote_torch_cpu",
        lambda: torch_library,
    )

    def fail_load(_path: str, mode: int) -> None:
        assert mode == torch_cpp_loader._COLLECTOR_LOAD_MODE
        raise OSError("missing symbol")

    monkeypatch.setattr(torch_cpp_loader.ctypes, "CDLL", fail_load)

    with pytest.raises(torch_cpp_loader.CollectorLoadError, match="missing symbol"):
        torch_cpp_loader.load()
    assert torch_cpp_loader._torch_cpu_library is torch_library


def test_load_rejects_incompatible_plain_c_abi(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    collector_path = tmp_path / "torch_trace_collector.so"
    native_library = make_native_library(revision=2)
    monkeypatch.setattr(
        torch_cpp_loader,
        "_workload_torch_version",
        lambda: "2.13",
    )
    monkeypatch.setattr(
        torch_cpp_loader,
        "_discover_collector_artifact",
        lambda: collector_path,
    )
    monkeypatch.setattr(torch_cpp_loader, "_promote_torch_cpu", object)
    monkeypatch.setattr(
        torch_cpp_loader.ctypes,
        "CDLL",
        lambda _path, mode: native_library,
    )

    with pytest.raises(
        torch_cpp_loader.CollectorLoadError,
        match="incompatible interface revision 2",
    ):
        torch_cpp_loader.load()


def test_wrapper_calls_installation_api(tmp_path: Path) -> None:
    native_library = make_native_library()
    collector = torch_cpp_loader.TorchTraceCollector(
        native_library,
        tmp_path / "torch_trace_collector.so",
    )

    collector.install()
    assert collector.is_installed()
    collector.uninstall()

    assert native_library.torch_trace_collector_install.calls == [()]
    assert native_library.torch_trace_collector_is_installed.calls == [()]
    assert native_library.torch_trace_collector_uninstall.calls == [()]


def test_wrapper_encodes_user_scope_arguments(tmp_path: Path) -> None:
    native_library = make_native_library()
    collector = torch_cpp_loader.TorchTraceCollector(
        native_library,
        tmp_path / "torch_trace_collector.so",
    )

    collector.push_user_scope("outer/µ", "#1@file.py:7", "torch")
    collector.pop_user_scope()

    assert native_library.torch_trace_collector_push_user_scope.calls == [
        (b"outer/\xc2\xb5", b"#1@file.py:7", b"torch")
    ]
    assert native_library.torch_trace_collector_pop_user_scope.calls == [()]


def test_wrapper_returns_native_stats(tmp_path: Path) -> None:
    native_library = make_native_library()
    native_library.torch_trace_collector_get_stats.callback = populate_native_stats
    collector = torch_cpp_loader.TorchTraceCollector(
        native_library,
        tmp_path / "torch_trace_collector.so",
    )

    stats = collector.dump_stats()

    assert stats == {
        "installed": True,
        "pushes": 8,
        "pops": 7,
        "user_scope_pushes": 3,
        "user_scope_pops": 2,
        "user_scope_inherits": 1,
        "snapshots_saved": 5,
        "snapshots_consumed": 4,
        "snapshots_dropped": 0,
        "snapshots_overwritten": 1,
        "callback_errors": 2,
        "snapshots_pending": 1,
    }


@pytest.mark.parametrize(
    ("method_name", "native_name"),
    [
        ("install", "torch_trace_collector_install"),
        ("uninstall", "torch_trace_collector_uninstall"),
        ("push_user_scope", "torch_trace_collector_push_user_scope"),
        ("pop_user_scope", "torch_trace_collector_pop_user_scope"),
        ("dump_stats", "torch_trace_collector_get_stats"),
    ],
)
def test_wrapper_raises_when_native_operation_fails(
    tmp_path: Path,
    method_name: str,
    native_name: str,
) -> None:
    native_library = make_native_library()
    getattr(native_library, native_name).return_value = 1
    collector = torch_cpp_loader.TorchTraceCollector(
        native_library,
        tmp_path / "torch_trace_collector.so",
    )

    method = getattr(collector, method_name)
    args = ("marker", "context", "torch") if method_name == "push_user_scope" else ()
    expected_operation = "get_stats" if method_name == "dump_stats" else method_name
    with pytest.raises(RuntimeError, match=expected_operation):
        method(*args)
