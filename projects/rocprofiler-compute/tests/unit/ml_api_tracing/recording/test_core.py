# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for recording ROCTX state, scopes, and caller attribution."""

import os
from pathlib import Path

from ml_api_tracing.marker_format import compose_marker
from ml_api_tracing.recording import core


def raise_import_skipped(*args, **kwargs):
    """Raise to assert that an import path is never reached."""
    raise AssertionError("roctx import should be skipped")


def test_ensure_python_tier_short_circuits_when_already_configured(monkeypatch):
    saved_push, saved_pop = core._STATE.range_push, core._STATE.range_pop
    try:
        core.set_python_tier_io(lambda _s: None, lambda: None)

        monkeypatch.setattr(core.PathFinder, "find_spec", raise_import_skipped)
        assert core.ensure_python_tier() is True
    finally:
        core._STATE.range_push, core._STATE.range_pop = saved_push, saved_pop


def test_push_scope_emits_composed_marker_and_pop_is_idempotent():
    previous = core.get_python_tier_io()
    pushes = []
    pops = []
    core.set_python_tier_io(pushes.append, lambda: pops.append("pop"))
    core._thread_local.depth = 0
    try:
        core._push_scope("aten::addmm", "n/a", backend="torch")
        assert pushes == [compose_marker("aten::addmm", "n/a", backend="torch")]
        core._pop_scope()
        core._pop_scope()
        assert pops == ["pop"]
    finally:
        core.set_python_tier_io(*previous)
        core._thread_local.depth = 0


def test_resolve_user_caller_location_skips_package_and_framework_roots():
    assert core._PACKAGE_ROOT == str(Path(core.__file__).resolve().parent) + os.sep
    location = core.resolve_user_caller_location()
    assert location.startswith(Path(__file__).name)
    assert ":" in location
    parent = str(Path(__file__).resolve().parent)
    core.add_framework_root(parent)
    stored = parent if parent.endswith(os.sep) else parent + os.sep
    try:
        skipped = core.resolve_user_caller_location()
        assert ":" in skipped
        assert not skipped.startswith(Path(__file__).name)
    finally:
        if stored in core._STATE.framework_roots:
            core._STATE.framework_roots.remove(stored)
