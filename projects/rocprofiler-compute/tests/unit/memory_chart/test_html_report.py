# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Loader capability checks used by the HTML report."""

import json

import pytest

from memory_chart import loader
from tests.unit.memory_chart.layout_cases import LAYOUT_ARCHS


@pytest.mark.parametrize("arch", sorted({arch for _, arch in LAYOUT_ARCHS}))
def test_has_layout_accepts_every_loader_architecture(arch):
    assert loader.has_layout(arch)


@pytest.mark.parametrize("arch", ["gfx1201", "gfx1030"])
def test_has_layout_rejects_unsupported_architectures(arch):
    assert not loader.has_layout(arch)


def test_has_layout_preserves_loader_validation_errors(monkeypatch, tmp_path):
    path = tmp_path / "broken.json"
    path.write_text(json.dumps({"archs": ["gfx908"]}), encoding="utf-8")
    monkeypatch.setattr(loader, "layout_files", lambda: [path])
    monkeypatch.setattr(loader.Layouts, "_by_arch", None)
    with pytest.raises(SystemExit):
        loader.has_layout("gfx908")


@pytest.mark.parametrize("arch", ["gfx1150", "gfx1151", "gfx1152", "gfx1153"])
def test_has_layout_reuses_loader_architecture_mapping(arch):
    assert loader.has_layout(arch) == (loader.Layouts.for_arch(arch) is not None)
