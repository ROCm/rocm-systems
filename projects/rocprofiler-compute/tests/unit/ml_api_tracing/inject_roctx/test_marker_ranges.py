# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for recording ROCTX state, scopes, and caller attribution."""

import os
from pathlib import Path

import pytest

from ml_api_tracing.inject_roctx import marker_ranges
from ml_api_tracing.marker_format import compose_marker


def raise_import_skipped(*args, **kwargs):
    """Raise to assert that an import path is never reached."""
    raise AssertionError("roctx import should be skipped")


def raise_scope_error(*args):
    raise RuntimeError("scope callback failed")


def test_ensure_python_tier_short_circuits_when_already_configured(monkeypatch):
    saved_push, saved_pop = (
        marker_ranges._CoreState.range_push,
        marker_ranges._CoreState.range_pop,
    )
    try:
        marker_ranges.set_python_tier_io(lambda _s: None, lambda: None)

        monkeypatch.setattr(marker_ranges.PathFinder, "find_spec", raise_import_skipped)
        assert marker_ranges.ensure_python_tier() is True
    finally:
        marker_ranges._CoreState.range_push, marker_ranges._CoreState.range_pop = (
            saved_push,
            saved_pop,
        )


def test_push_scope_emits_composed_marker_and_pop_is_idempotent():
    previous = marker_ranges.get_python_tier_io()
    pushes = []
    pops = []
    marker_ranges.set_python_tier_io(pushes.append, lambda: pops.append("pop"))
    marker_ranges._thread_local.depth = 0
    try:
        marker_ranges.push_scope("aten::addmm", "n/a", backend="torch")
        assert pushes == [compose_marker("aten::addmm", "n/a", backend="torch")]
        marker_ranges.pop_scope()
        marker_ranges.pop_scope()
        assert pops == ["pop"]
    finally:
        marker_ranges.set_python_tier_io(*previous)
        marker_ranges._thread_local.depth = 0


def test_resolve_user_caller_location_skips_package_and_framework_roots():
    assert (
        marker_ranges._PACKAGE_ROOT
        == str(Path(marker_ranges.__file__).resolve().parent) + os.sep
    )
    location = marker_ranges.resolve_user_caller_location()
    assert location.startswith(Path(__file__).name)
    assert ":" in location
    parent = str(Path(__file__).resolve().parent)
    marker_ranges.add_framework_root(parent)
    stored = parent if parent.endswith(os.sep) else parent + os.sep
    try:
        skipped = marker_ranges.resolve_user_caller_location()
        assert ":" in skipped
        assert not skipped.startswith(Path(__file__).name)
    finally:
        if stored in marker_ranges._CoreState.framework_roots:
            marker_ranges._CoreState.framework_roots.remove(stored)


def test_failed_push_preserves_outer_scope(monkeypatch):
    pops = []
    monkeypatch.setattr(marker_ranges._CoreState, "range_push", raise_scope_error)
    monkeypatch.setattr(
        marker_ranges._CoreState, "range_pop", lambda: pops.append("outer")
    )
    monkeypatch.setattr(marker_ranges._thread_local, "depth", 1, raising=False)

    with pytest.raises(RuntimeError, match="scope callback failed"):
        marker_ranges.push_scope("inner", "workload.py:1", backend="torch")

    assert marker_ranges._thread_local.depth == 1
    marker_ranges.pop_scope()
    assert pops == ["outer"]
    assert marker_ranges._thread_local.depth == 0


def test_failed_pop_is_not_retried(monkeypatch):
    monkeypatch.setattr(marker_ranges._CoreState, "range_pop", raise_scope_error)
    monkeypatch.setattr(marker_ranges._thread_local, "depth", 1, raising=False)

    with pytest.raises(RuntimeError, match="scope callback failed"):
        marker_ranges.pop_scope()

    assert marker_ranges._thread_local.depth == 0
    marker_ranges.pop_scope()
