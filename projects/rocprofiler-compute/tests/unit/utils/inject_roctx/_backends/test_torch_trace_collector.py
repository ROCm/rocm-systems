# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for the generic native Torch collector loader."""

import builtins
import ctypes
import re
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple

import pytest

from utils.inject_roctx._backends import torch_trace_collector

PROJECT_ROOT = Path(__file__).resolve().parents[5]
CPU_213_IDENTITY = torch_trace_collector._TorchIdentity(
    version="2.13.0+cpu",
    git_version="cf30153c4c131c8164ee7798e5022d810682e2cb",
    debug=False,
    uses_cxx11_abi=True,
)
ROCM_214_IDENTITY = torch_trace_collector._TorchIdentity(
    version="2.14.0+rocm10.2.0a20260917",
    git_version="50011224068dd1fbc41573a6e43c9d9ffa053a63",
    debug=False,
    uses_cxx11_abi=True,
)


class FakeNativeFunction:
    def __init__(self, return_value: int = 0) -> None:
        self.return_value = return_value
        self.argtypes = None
        self.restype = None
        self.call_count = 0
        self.calls: List[Tuple[object, ...]] = []

    def __call__(self, *args: object) -> int:
        self.call_count += 1
        self.calls.append(args)
        return self.return_value


def stub_torch(
    monkeypatch: pytest.MonkeyPatch,
    identity: torch_trace_collector._TorchIdentity,
) -> None:
    torch_module = types.SimpleNamespace(
        __file__="/opt/fake/torch/__init__.py",
        __version__=identity.version,
        version=types.SimpleNamespace(
            git_version=identity.git_version,
            debug=identity.debug,
        ),
        _C=types.SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=identity.uses_cxx11_abi),
    )
    monkeypatch.setitem(sys.modules, "torch", torch_module)


def raise_load_error(*_args: object, **_kwargs: object) -> None:
    raise OSError("missing symbol")


def raise_lookup_error() -> None:
    raise OSError("artifact lookup failed")


@pytest.fixture(autouse=True)
def reset_loader_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test with empty process-lifetime loader handles."""
    monkeypatch.setattr(torch_trace_collector, "_collector_library", None)
    monkeypatch.setattr(torch_trace_collector, "_torch_cpu_library", None)


@pytest.fixture
def warning_messages(monkeypatch: pytest.MonkeyPatch) -> List[str]:
    """Capture loader warning messages."""
    messages = []
    monkeypatch.setattr(
        torch_trace_collector,
        "console_warning",
        lambda _category, message: messages.append(message),
    )
    return messages


@pytest.fixture
def native_loader_scenario(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    warning_messages: List[str],
) -> SimpleNamespace:
    """Configure the common supported-build loader path for failure tests."""
    scenario = SimpleNamespace(
        collector_path=tmp_path / "torch_trace_collector.so",
        torch_cpu_library=object(),
        warning_messages=warning_messages,
    )
    monkeypatch.setattr(
        torch_trace_collector,
        "_workload_torch_identity",
        lambda: CPU_213_IDENTITY,
    )
    monkeypatch.setattr(
        torch_trace_collector,
        "_find_collector",
        lambda: scenario.collector_path,
    )
    monkeypatch.setattr(
        torch_trace_collector,
        "_promote_torch_cpu",
        lambda: scenario.torch_cpu_library,
    )
    return scenario


def test_workload_torch_identity_reports_version_revision_debug_and_abi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_torch(monkeypatch, CPU_213_IDENTITY)

    assert torch_trace_collector._workload_torch_identity() == CPU_213_IDENTITY


def test_loader_abi_revision_matches_native_header() -> None:
    source_header = (
        PROJECT_ROOT / "src/lib/torch_trace_collector/torch_trace_collector.h"
    )
    installed_test_header = PROJECT_ROOT / "tests/torch_trace_collector_abi.h"
    header_path = source_header if source_header.is_file() else installed_test_header
    assert header_path.is_file()
    header = header_path.read_text(encoding="utf-8")
    match = re.search(
        r"^#define TORCH_TRACE_COLLECTOR_ABI_REVISION ([0-9]+)U$",
        header,
        flags=re.MULTILINE,
    )

    assert match is not None
    assert int(match.group(1)) == torch_trace_collector._EXPECTED_COLLECTOR_ABI_REVISION


def test_workload_torch_identity_warns_when_torch_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    warning_messages: List[str],
) -> None:
    real_import = builtins.__import__

    def fail_torch_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch missing")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fail_torch_import)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    assert torch_trace_collector._workload_torch_identity() == (
        "",
        "",
        False,
        False,
    )
    assert "torch missing" in warning_messages[0]


def test_workload_torch_identity_rejects_missing_debug_metadata(
    monkeypatch: pytest.MonkeyPatch,
    warning_messages: List[str],
) -> None:
    torch_module = types.SimpleNamespace(
        __file__="/opt/fake/torch/__init__.py",
        __version__=CPU_213_IDENTITY.version,
        version=types.SimpleNamespace(git_version=CPU_213_IDENTITY.git_version),
        _C=types.SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=True),
    )
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    assert torch_trace_collector._workload_torch_identity() == (
        "",
        "",
        False,
        False,
    )
    assert "build identity" in warning_messages[0]


def test_find_collector_uses_generic_artifact_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    collector_path = tmp_path / "torch_trace_collector.so"
    lookups = []
    monkeypatch.setattr(
        torch_trace_collector,
        "find_prebuilt_artifacts",
        lambda package_root, artifact_name: (
            lookups.append((package_root, artifact_name)) or [collector_path]
        ),
    )

    assert torch_trace_collector._find_collector() == collector_path
    assert lookups == [
        (
            torch_trace_collector._PACKAGE_ROOT,
            "torch_trace_collector.so",
        )
    ]


def test_promote_torch_cpu_reopens_workload_library_globally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_torch(monkeypatch, CPU_213_IDENTITY)
    loaded_libraries: List[Tuple[str, int]] = []
    fake_library = object()
    monkeypatch.setattr(
        torch_trace_collector.ctypes,
        "CDLL",
        lambda path, mode: loaded_libraries.append((path, mode)) or fake_library,
    )

    assert torch_trace_collector._promote_torch_cpu() is fake_library
    assert loaded_libraries == [
        (
            "/opt/fake/torch/lib/libtorch_cpu.so",
            torch_trace_collector._TORCH_LIBRARY_LOAD_MODE,
        )
    ]


@pytest.mark.parametrize("identity", [CPU_213_IDENTITY, ROCM_214_IDENTITY])
def test_install_loads_generic_collector_for_supported_torch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    identity: torch_trace_collector._TorchIdentity,
) -> None:
    collector_path = tmp_path / "torch_trace_collector.so"
    loaded_collectors: List[Tuple[str, int]] = []
    load_order = []
    fake_torch_cpu_library = object()
    fake_abi_revision = FakeNativeFunction(return_value=2)
    fake_install = FakeNativeFunction()
    fake_push_launcher_tid = FakeNativeFunction()
    fake_pop_launcher_tid = FakeNativeFunction()
    fake_library = SimpleNamespace(
        torch_trace_collector_abi_revision=fake_abi_revision,
        torch_trace_collector_install=fake_install,
        torch_trace_collector_push_launcher_tid=fake_push_launcher_tid,
        torch_trace_collector_pop_launcher_tid=fake_pop_launcher_tid,
    )

    monkeypatch.setattr(
        torch_trace_collector,
        "_workload_torch_identity",
        lambda: identity,
    )
    monkeypatch.setattr(
        torch_trace_collector,
        "_find_collector",
        lambda: collector_path,
    )
    monkeypatch.setattr(
        torch_trace_collector,
        "_promote_torch_cpu",
        lambda: load_order.append("torch_cpu") or fake_torch_cpu_library,
    )
    monkeypatch.setattr(
        torch_trace_collector.ctypes,
        "CDLL",
        lambda path, mode: (
            load_order.append("collector")
            or loaded_collectors.append((path, mode))
            or fake_library
        ),
    )
    monkeypatch.setattr(torch_trace_collector, "console_log", lambda *_: None)

    assert torch_trace_collector.install()
    assert loaded_collectors == [
        (str(collector_path), torch_trace_collector._COLLECTOR_LOAD_MODE)
    ]
    assert load_order == ["torch_cpu", "collector"]
    assert torch_trace_collector._torch_cpu_library is fake_torch_cpu_library
    assert fake_abi_revision.restype is ctypes.c_uint32
    assert fake_abi_revision.argtypes == []
    assert fake_abi_revision.call_count == 1
    assert fake_install.restype is ctypes.c_int
    assert fake_install.argtypes == []
    assert fake_install.call_count == 1
    assert fake_push_launcher_tid.restype is ctypes.c_int
    assert fake_push_launcher_tid.argtypes == [ctypes.c_uint64]
    assert fake_pop_launcher_tid.restype is ctypes.c_int
    assert fake_pop_launcher_tid.argtypes == []
    assert torch_trace_collector._collector_library is fake_library


@pytest.mark.parametrize(
    ("identity", "message_fragment"),
    [
        pytest.param(
            torch_trace_collector._TorchIdentity(
                version="2.12.0+cpu",
                git_version="old",
                debug=False,
                uses_cxx11_abi=True,
            ),
            "version=2.12.0+cpu",
            id="version",
        ),
        pytest.param(
            torch_trace_collector._TorchIdentity(
                version="2.13.0+self-built",
                git_version=CPU_213_IDENTITY.git_version,
                debug=False,
                uses_cxx11_abi=True,
            ),
            "version=2.13.0+self-built",
            id="build-tag",
        ),
        pytest.param(
            torch_trace_collector._TorchIdentity(
                version=CPU_213_IDENTITY.version,
                git_version="unvalidated",
                debug=False,
                uses_cxx11_abi=True,
            ),
            "git=unvalidated",
            id="git-revision",
        ),
        pytest.param(
            torch_trace_collector._TorchIdentity(
                version=CPU_213_IDENTITY.version,
                git_version=CPU_213_IDENTITY.git_version,
                debug=True,
                uses_cxx11_abi=True,
            ),
            "debug=True",
            id="debug-build",
        ),
        pytest.param(
            torch_trace_collector._TorchIdentity(
                version=CPU_213_IDENTITY.version,
                git_version=CPU_213_IDENTITY.git_version,
                debug=False,
                uses_cxx11_abi=False,
            ),
            "cxx11_abi=False",
            id="cxx11-abi",
        ),
    ],
)
def test_install_uses_fallback_for_unsupported_torch(
    monkeypatch: pytest.MonkeyPatch,
    identity: torch_trace_collector._TorchIdentity,
    message_fragment: str,
    warning_messages: List[str],
) -> None:
    monkeypatch.setattr(
        torch_trace_collector,
        "_workload_torch_identity",
        lambda: identity,
    )
    monkeypatch.setattr(
        torch_trace_collector,
        "_find_collector",
        lambda: pytest.fail("collector lookup must not run"),
    )
    assert not torch_trace_collector.install()
    assert message_fragment in warning_messages[0]


def test_install_uses_fallback_when_collector_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    warning_messages: List[str],
) -> None:
    monkeypatch.setattr(
        torch_trace_collector,
        "_workload_torch_identity",
        lambda: ROCM_214_IDENTITY,
    )
    monkeypatch.setattr(torch_trace_collector, "_find_collector", lambda: None)
    assert not torch_trace_collector.install()
    assert "was not built" in warning_messages[0]


def test_install_uses_fallback_when_collector_lookup_fails(
    monkeypatch: pytest.MonkeyPatch,
    warning_messages: List[str],
) -> None:
    monkeypatch.setattr(
        torch_trace_collector,
        "_workload_torch_identity",
        lambda: CPU_213_IDENTITY,
    )
    monkeypatch.setattr(
        torch_trace_collector,
        "_find_collector",
        raise_lookup_error,
    )
    monkeypatch.setattr(
        torch_trace_collector,
        "_promote_torch_cpu",
        lambda: pytest.fail("Torch must not be promoted after lookup failure"),
    )
    assert not torch_trace_collector.install()
    assert "artifact lookup failed" in warning_messages[0]


def test_install_uses_fallback_when_native_install_fails(
    monkeypatch: pytest.MonkeyPatch,
    native_loader_scenario: SimpleNamespace,
) -> None:
    fake_library = SimpleNamespace(
        torch_trace_collector_abi_revision=FakeNativeFunction(return_value=2),
        torch_trace_collector_install=FakeNativeFunction(return_value=1),
        torch_trace_collector_push_launcher_tid=FakeNativeFunction(),
        torch_trace_collector_pop_launcher_tid=FakeNativeFunction(),
    )

    monkeypatch.setattr(
        torch_trace_collector.ctypes,
        "CDLL",
        lambda _path, mode: fake_library,
    )

    assert not torch_trace_collector.install()
    assert "install failed" in native_loader_scenario.warning_messages[0]
    assert fake_library.torch_trace_collector_install.call_count == 1
    assert torch_trace_collector._collector_library is None


def test_install_uses_fallback_when_abi_symbol_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    native_loader_scenario: SimpleNamespace,
) -> None:
    monkeypatch.setattr(
        torch_trace_collector.ctypes,
        "CDLL",
        lambda _path, mode: SimpleNamespace(),
    )

    assert not torch_trace_collector.install()
    assert "AttributeError" in native_loader_scenario.warning_messages[0]
    assert torch_trace_collector._collector_library is None


def test_install_uses_fallback_for_incompatible_collector_abi(
    monkeypatch: pytest.MonkeyPatch,
    native_loader_scenario: SimpleNamespace,
) -> None:
    fake_library = SimpleNamespace(
        torch_trace_collector_abi_revision=FakeNativeFunction(return_value=3)
    )

    monkeypatch.setattr(
        torch_trace_collector.ctypes,
        "CDLL",
        lambda _path, mode: fake_library,
    )

    assert not torch_trace_collector.install()
    assert (
        "incompatible interface revision 3"
        in native_loader_scenario.warning_messages[0]
    )
    assert torch_trace_collector._collector_library is None


def test_install_uses_fallback_when_install_symbol_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    native_loader_scenario: SimpleNamespace,
) -> None:
    fake_library = SimpleNamespace(
        torch_trace_collector_abi_revision=FakeNativeFunction(return_value=2),
        torch_trace_collector_push_launcher_tid=FakeNativeFunction(),
        torch_trace_collector_pop_launcher_tid=FakeNativeFunction(),
    )

    monkeypatch.setattr(
        torch_trace_collector.ctypes,
        "CDLL",
        lambda _path, mode: fake_library,
    )

    assert not torch_trace_collector.install()
    assert "AttributeError" in native_loader_scenario.warning_messages[0]
    assert torch_trace_collector._collector_library is None


@pytest.mark.parametrize(
    "missing_symbol",
    [
        "torch_trace_collector_push_launcher_tid",
        "torch_trace_collector_pop_launcher_tid",
    ],
)
def test_install_uses_fallback_when_launcher_tid_symbol_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    native_loader_scenario: SimpleNamespace,
    missing_symbol: str,
) -> None:
    fake_library = SimpleNamespace(
        torch_trace_collector_abi_revision=FakeNativeFunction(return_value=2),
        torch_trace_collector_install=FakeNativeFunction(),
        torch_trace_collector_push_launcher_tid=FakeNativeFunction(),
        torch_trace_collector_pop_launcher_tid=FakeNativeFunction(),
    )
    delattr(fake_library, missing_symbol)
    monkeypatch.setattr(
        torch_trace_collector.ctypes,
        "CDLL",
        lambda _path, mode: fake_library,
    )

    assert not torch_trace_collector.install()
    assert "AttributeError" in native_loader_scenario.warning_messages[0]
    assert fake_library.torch_trace_collector_install.call_count == 0
    assert torch_trace_collector._collector_library is None


def test_install_uses_fallback_when_torch_cpu_cannot_be_promoted(
    monkeypatch: pytest.MonkeyPatch,
    native_loader_scenario: SimpleNamespace,
) -> None:
    monkeypatch.setattr(
        torch_trace_collector,
        "_promote_torch_cpu",
        raise_load_error,
    )

    assert not torch_trace_collector.install()
    assert "missing symbol" in native_loader_scenario.warning_messages[0]
    assert torch_trace_collector._torch_cpu_library is None
    assert torch_trace_collector._collector_library is None


def test_install_uses_fallback_when_collector_cannot_load(
    monkeypatch: pytest.MonkeyPatch,
    native_loader_scenario: SimpleNamespace,
) -> None:
    monkeypatch.setattr(
        torch_trace_collector.ctypes,
        "CDLL",
        raise_load_error,
    )

    assert not torch_trace_collector.install()
    assert "missing symbol" in native_loader_scenario.warning_messages[0]
    assert (
        torch_trace_collector._torch_cpu_library
        is native_loader_scenario.torch_cpu_library
    )
    assert torch_trace_collector._collector_library is None


def test_install_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    existing_library = object()
    monkeypatch.setattr(torch_trace_collector, "_collector_library", existing_library)
    monkeypatch.setattr(
        torch_trace_collector,
        "_workload_torch_identity",
        lambda: pytest.fail("identity lookup must not run"),
    )

    assert torch_trace_collector.install()


def test_launcher_tid_helpers_call_the_installed_collector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    push = FakeNativeFunction()
    pop = FakeNativeFunction()
    fake_library = SimpleNamespace(
        torch_trace_collector_push_launcher_tid=push,
        torch_trace_collector_pop_launcher_tid=pop,
    )
    monkeypatch.setattr(torch_trace_collector, "_collector_library", fake_library)
    monkeypatch.setattr(torch_trace_collector.threading, "get_native_id", lambda: 73)

    assert torch_trace_collector.push_launcher_tid()
    assert torch_trace_collector.pop_launcher_tid()
    assert push.calls == [(73,)]
    assert pop.calls == [()]


def test_launcher_tid_helpers_fail_closed_without_collector() -> None:
    assert not torch_trace_collector.push_launcher_tid()
    assert not torch_trace_collector.pop_launcher_tid()


def test_launcher_tid_helpers_propagate_native_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_library = SimpleNamespace(
        torch_trace_collector_push_launcher_tid=FakeNativeFunction(return_value=1),
        torch_trace_collector_pop_launcher_tid=FakeNativeFunction(return_value=1),
    )
    monkeypatch.setattr(torch_trace_collector, "_collector_library", fake_library)

    assert not torch_trace_collector.push_launcher_tid()
    assert not torch_trace_collector.pop_launcher_tid()
