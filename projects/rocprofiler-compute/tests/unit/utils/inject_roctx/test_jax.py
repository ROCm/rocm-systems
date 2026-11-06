# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for the JAX inject_roctx backend."""

from utils.inject_roctx._backends import jax as jax_backend


def test_jax_backend_skips_when_jax_missing(monkeypatch):
    warnings = []

    def capture_warning(prefix, message=""):
        warnings.append((prefix, message))

    monkeypatch.setattr(
        jax_backend.importlib.util, "find_spec", lambda name, package=None: None
    )
    monkeypatch.setattr(jax_backend, "console_warning", capture_warning)
    jax_backend.JaxBackend().install()
    assert warnings
    assert warnings[0][0] == "ml api trace"
    assert "JAX is not installed" in warnings[0][1]
