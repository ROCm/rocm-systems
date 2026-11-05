# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for workload-time backend discovery and installation."""

import functools
import importlib
import subprocess
import sys
import types
from pathlib import Path

import pytest

from ml_api_tracing.recording import installer
from ml_api_tracing.recording.installer import install_global_wraps

CANONICAL_RECORDING_PROBE = """
from typing import get_type_hints
package_name = 'ml_api_tracing.recording.backends'
if sys.argv[3] == 'ordinary':
    __import__(package_name, fromlist=('*',))
else:
    importlib.import_module(package_name)
assert {name for name in sys.modules if name.startswith('ml_api_tracing')} == {
    'ml_api_tracing', 'ml_api_tracing.recording', 'ml_api_tracing.recording.backends'
}
modules = {}
for name in ('backend_contract', 'core', 'installer',
             'backends.torch_trace_collector', 'backends.torch', 'backends.triton'):
    module_name = 'ml_api_tracing.recording.' + name
    assert module_name not in sys.modules, module_name
    if sys.argv[3] == 'ordinary':
        module = __import__(module_name, fromlist=('*',))
    else:
        module = importlib.import_module(module_name)
    modules[name] = module
backend_contract = modules['backend_contract']
core = modules['core']
torch = modules['backends.torch']
triton = modules['backends.triton']
torch_trace_collector = modules['backends.torch_trace_collector']
assert torch.core is core and triton.core is core
assert torch.torch_trace_collector is torch_trace_collector
assert isinstance(backend_contract.get_backend('torch'), torch.TorchBackend)
assert isinstance(backend_contract.get_backend('triton'), triton.TritonBackend)
assert backend_contract.get_backend('missing') is None
for module in modules.values():
    assert importlib.import_module(module.__name__) is module
    for value in vars(module).values():
        if callable(value) and getattr(value, '__module__', None) == module.__name__:
            get_type_hints(value)
assert_project_origins()
assert not forbidden_attempts, forbidden_attempts
print('canonical state and optional imports preserved')
"""


def record_call(calls, names):
    """Append a snapshot of names to calls as an install_many spy."""
    calls.append(list(names))


def raise_import_skipped(*args, **kwargs):
    """Raise to assert that an import path is never reached."""
    raise AssertionError("backend discovery should be skipped")


def make_backend(name, install_fn=None):
    backend = types.SimpleNamespace()
    backend.name = name
    backend.install = install_fn or (lambda: None)
    return backend


class FalseBackend(types.SimpleNamespace):
    def __bool__(self):
        return False


@pytest.fixture
def captured_warnings(monkeypatch):
    """Capture complete installer warning arguments for each test."""
    warnings = []
    monkeypatch.setattr(
        installer, "console_warning", lambda *args: warnings.append(args)
    )
    return warnings


@pytest.fixture
def captured_install(monkeypatch):
    """Replace installer.install_many with a recorder."""
    calls = []
    monkeypatch.setattr(
        installer, "install_many", functools.partial(record_call, calls)
    )
    return calls


def test_install_global_wraps_empty_input_is_noop(captured_install):
    install_global_wraps([])
    install_global_wraps(())
    assert captured_install == []


def test_install_global_wraps_list_input(captured_install):
    install_global_wraps(["torch"])
    install_global_wraps(["torch", "triton"])
    assert captured_install == [["torch"], ["torch", "triton"]]


def test_install_global_wraps_normalizes_names(captured_install):
    install_global_wraps(name for name in [" torch ", "", "\t", 7, " triton"])
    assert captured_install == [["torch", "7", "triton"]]


def test_install_many_invokes_registered_backends(fresh_registry):
    calls = []
    fresh_registry.register(make_backend("alpha", lambda: calls.append("alpha")))
    fresh_registry.register(make_backend("beta", lambda: calls.append("beta")))

    installer.install_many(["alpha", "beta"])
    assert calls == ["alpha", "beta"]


def test_install_many_dedupes_duplicate_names(fresh_registry):
    calls = []
    fresh_registry.register(make_backend("alpha", lambda: calls.append("alpha")))

    installer.install_many(["alpha", "alpha", "alpha"])
    assert calls == ["alpha"]


def test_install_many_continues_after_backend_failure(
    fresh_registry, captured_warnings
):
    other_calls = []
    fresh_registry.register(
        make_backend("bad", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    )
    fresh_registry.register(make_backend("good", lambda: other_calls.append("good")))

    installer.install_many(["bad", "good"])
    assert other_calls == ["good"]
    assert captured_warnings == [
        ("inject_roctx", "backend 'bad' install failed; continuing: boom")
    ]


def test_install_many_warns_on_unknown_backend(fresh_registry, captured_warnings):
    installer.install_many(["does_not_exist_zzz"])
    assert any(
        "does_not_exist_zzz" in str(args[1])
        for args in captured_warnings
        if len(args) > 1
    )


def test_install_many_warns_when_module_does_not_register(
    fresh_registry, captured_warnings
):
    fake_name = "ml_api_tracing.recording.backends.ghost"
    sys.modules[fake_name] = types.ModuleType(fake_name)
    try:
        installer.install_many(["ghost"])
    finally:
        sys.modules.pop(fake_name, None)

    assert any(
        "did not register" in str(args[1])
        for args in captured_warnings
        if len(args) > 1
    )


def test_install_many_installs_false_backend_without_discovery(
    fresh_registry, monkeypatch
):
    calls = []
    fresh_registry.register(FalseBackend(name="alpha", install=lambda: calls.append(1)))
    monkeypatch.setattr(importlib, "import_module", raise_import_skipped)
    installer.install_many(["", "alpha", "alpha"])
    assert calls == [1]


@pytest.mark.parametrize("failure", ["none", "install", "import", "register-import"])
def test_install_many_retries_on_later_calls(
    fresh_registry, monkeypatch, captured_warnings, failure
):
    calls = []
    attempts = []

    def install():
        calls.append("alpha")
        if failure == "install" and len(calls) == 1:
            raise RuntimeError("install failed")

    backend = make_backend("alpha", install)

    def discover(name, package):
        attempts.append((name, package))
        if failure == "register-import" or len(attempts) > 1:
            fresh_registry.register(backend)
        if len(attempts) == 1:
            raise ImportError("discovery failed")

    if failure in ("none", "install"):
        fresh_registry.register(backend)
    else:
        monkeypatch.setattr(importlib, "import_module", discover)
    installer.install_many(["alpha", "alpha"])
    installer.install_many(["alpha"])
    assert calls == ["alpha"] * (2 if failure in ("none", "install") else 1)
    assert len(captured_warnings) == (0 if failure == "none" else 1)
    assert (
        len(attempts)
        == {"none": 0, "install": 0, "import": 2, "register-import": 1}[failure]
    )


def test_install_many_discovers_new_module_and_warns_without_registration(
    fresh_registry, monkeypatch, captured_warnings
):
    imports = []
    monkeypatch.setattr(importlib, "import_module", lambda *args: imports.append(args))
    installer.install_many(["ghost", "ghost"])
    assert imports == [(".ghost", installer._BACKENDS_PKG)]
    assert len(captured_warnings) == 1
    assert "did not register 'ghost'" in captured_warnings[0][1]


def test_install_many_preserves_partial_install_state(
    fresh_registry, captured_warnings
):
    calls = []

    def install_bad():
        calls.append("partial")
        raise RuntimeError("partial install")

    fresh_registry.register(make_backend("bad", install_bad))
    fresh_registry.register(make_backend("good", lambda: calls.append("good")))
    installer.install_many(["bad", "good", "bad"])
    assert calls == ["partial", "good"]
    assert len(captured_warnings) == 1


@pytest.mark.parametrize("import_mode", ["ordinary", "dynamic"])
def test_recording_uses_canonical_state_without_framework_imports(
    tmp_path, profile_import_setup, profile_import_tests_root, import_mode
):
    python_root = Path(installer.__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-I",
            "-S",
            "-c",
            profile_import_setup + CANONICAL_RECORDING_PROBE,
            str(python_root),
            str(profile_import_tests_root),
            import_mode,
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=True,
    )
    assert completed.stdout.strip() == "canonical state and optional imports preserved"
