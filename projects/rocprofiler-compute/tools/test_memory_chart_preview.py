#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Tests for the memory chart layout preview."""

import pytest
from memory_chart_preview import main

from memory_chart.loader import layout_files
from utils.utils_common import strip_ansi


def preview(capsys, *args):
    assert main(list(args)) == 0
    return strip_ansi(capsys.readouterr().out)


@pytest.mark.parametrize("path", layout_files(), ids=lambda p: p.stem)
def test_placeholders_fill_every_metric_of_a_layout_file(path, capsys):
    output = preview(capsys, str(path))
    assert path.name in output.splitlines()[0]
    assert "N/A" not in output


def test_an_arch_previews_its_shipped_layout(capsys):
    assert "gfx950.json" in preview(capsys, "gfx950").splitlines()[0]
