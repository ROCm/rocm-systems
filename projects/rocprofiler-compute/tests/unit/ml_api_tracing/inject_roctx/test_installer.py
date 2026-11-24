# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for workload-time backend discovery and installation."""

import sys
import types

import pytest

from ml_api_tracing.inject_roctx import installer
from ml_api_tracing.inject_roctx.installer import install_global_wraps


def make_backend(name, install_fn=None):
    backend = types.SimpleNamespace()
    backend.name = name
    backend.install = install_fn or (lambda: None)
    return backend


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
