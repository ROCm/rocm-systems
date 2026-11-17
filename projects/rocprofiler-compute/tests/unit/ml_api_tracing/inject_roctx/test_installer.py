# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for workload-time backend discovery and installation."""

import importlib
import sys
import types

import pytest

from ml_api_tracing.inject_roctx import installer
from ml_api_tracing.inject_roctx.installer import install_global_wraps


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


class WarningRecorder:
    """Stand in for console_warning: keep every complete argument tuple."""

    def __init__(self):
        self.warnings = []

    def __call__(self, *args):
        self.warnings.append(args)


@pytest.fixture
def captured_warnings(monkeypatch):
    """Capture complete installer warning arguments for each test."""
    recorder = WarningRecorder()
    monkeypatch.setattr(installer, "console_warning", recorder)
    return recorder.warnings


def test_install_global_wraps_empty_input_is_noop(fresh_registry):
    calls = []
    fresh_registry.register(make_backend("alpha", lambda: calls.append("alpha")))

    install_global_wraps([])
    install_global_wraps(())

    assert calls == []


def test_install_global_wraps_normalizes_names(fresh_registry):
    calls = []
    for name in ("torch", "7", "triton"):
        fresh_registry.register(
            make_backend(name, lambda name=name: calls.append(name))
        )

    install_global_wraps(name for name in [" torch ", "", "\t", 7, " triton"])

    assert calls == ["torch", "7", "triton"]


def test_install_global_wraps_invokes_registered_backends(fresh_registry):
    calls = []
    fresh_registry.register(make_backend("alpha", lambda: calls.append("alpha")))
    fresh_registry.register(make_backend("beta", lambda: calls.append("beta")))

    install_global_wraps(["alpha", "beta"])
    assert calls == ["alpha", "beta"]


def test_install_global_wraps_dedupes_duplicate_names(fresh_registry):
    calls = []
    fresh_registry.register(make_backend("alpha", lambda: calls.append("alpha")))

    install_global_wraps(["alpha", "alpha", "alpha"])
    assert calls == ["alpha"]


def test_install_global_wraps_continues_after_backend_failure(
    fresh_registry, captured_warnings
):
    other_calls = []
    fresh_registry.register(
        make_backend("bad", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    )
    fresh_registry.register(make_backend("good", lambda: other_calls.append("good")))

    install_global_wraps(["bad", "good"])
    assert other_calls == ["good"]
    assert captured_warnings == [
        ("inject_roctx", "backend 'bad' install failed; continuing: boom")
    ]


def test_install_global_wraps_warns_on_unknown_backend(
    fresh_registry, captured_warnings
):
    install_global_wraps(["does_not_exist_zzz"])
    assert any(
        "does_not_exist_zzz" in str(args[1])
        for args in captured_warnings
        if len(args) > 1
    )


def test_install_global_wraps_warns_when_module_does_not_register(
    fresh_registry, captured_warnings, monkeypatch
):
    fake_name = "ml_api_tracing.inject_roctx.backends.ghost"
    monkeypatch.setitem(sys.modules, fake_name, types.ModuleType(fake_name))

    install_global_wraps(["ghost"])

    assert any(
        "did not register" in str(args[1])
        for args in captured_warnings
        if len(args) > 1
    )


def test_install_global_wraps_installs_false_backend_without_discovery(
    fresh_registry, monkeypatch
):
    calls = []
    fresh_registry.register(FalseBackend(name="alpha", install=lambda: calls.append(1)))
    monkeypatch.setattr(importlib, "import_module", raise_import_skipped)
    install_global_wraps(["", "alpha", "alpha"])
    assert calls == [1]


@pytest.mark.parametrize("failure", ["none", "install", "import", "register-import"])
def test_install_global_wraps_retries_on_later_calls(
    fresh_registry, monkeypatch, captured_warnings, failure
):
    calls = []
    attempts = []

    def install():
        calls.append("alpha")
        if failure == "install" and len(calls) == 1:
            raise RuntimeError("install failed")

    backend = make_backend("alpha", install)

    def discover(name):
        attempts.append(name)
        if failure == "register-import" or len(attempts) > 1:
            fresh_registry.register(backend)
        if len(attempts) == 1:
            raise ImportError("discovery failed")

    if failure in ("none", "install"):
        fresh_registry.register(backend)
    else:
        monkeypatch.setattr(importlib, "import_module", discover)
    install_global_wraps(["alpha", "alpha"])
    install_global_wraps(["alpha"])
    assert calls == ["alpha"] * (2 if failure in ("none", "install") else 1)
    assert len(captured_warnings) == (0 if failure == "none" else 1)
    assert (
        len(attempts)
        == {"none": 0, "install": 0, "import": 2, "register-import": 1}[failure]
    )


def test_install_global_wraps_discovers_new_module_and_warns_without_registration(
    fresh_registry, monkeypatch, captured_warnings
):
    imports = []
    monkeypatch.setattr(importlib, "import_module", lambda *args: imports.append(args))
    install_global_wraps(["ghost", "ghost"])
    assert imports == [(f"{installer._BACKENDS_PKG}.ghost",)]
    assert len(captured_warnings) == 1
    assert "did not register 'ghost'" in captured_warnings[0][1]


def test_install_global_wraps_preserves_partial_install_state(
    fresh_registry, captured_warnings
):
    calls = []

    def install_bad():
        calls.append("partial")
        raise RuntimeError("partial install")

    fresh_registry.register(make_backend("bad", install_bad))
    fresh_registry.register(make_backend("good", lambda: calls.append("good")))
    install_global_wraps(["bad", "good", "bad"])
    assert calls == ["partial", "good"]
    assert len(captured_warnings) == 1
