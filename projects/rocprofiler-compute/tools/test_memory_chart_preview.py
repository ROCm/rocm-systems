#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Tests for the memory chart layout preview."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from memory_chart_preview import main  # noqa: E402

import config  # noqa: E402
from memory_chart.loader import layout_files  # noqa: E402
from memory_chart.mem_chart import strip_ansi  # noqa: E402

GFX942_CONFIG = (
    config.rocprof_compute_home
    / "rocprof_compute_soc"
    / "analysis_configs"
    / "gfx942"
    / "0300_memory_chart.yaml"
)


def preview(capsys, *args):
    assert main(list(args)) == 0
    return strip_ansi(capsys.readouterr().out)


@pytest.mark.parametrize("path", layout_files(), ids=lambda p: p.stem)
def test_placeholders_fill_every_metric_of_a_layout_file(path, capsys):
    output = preview(capsys, str(path))
    assert path.name in output.splitlines()[0]
    assert "N/A" not in output


def test_values_file_and_empty(tmp_path, capsys):
    values = tmp_path / "values.json"
    values.write_text(json.dumps({"L2 Hit": 12.5}), encoding="utf-8")
    assert "Hit 12.5%" in preview(capsys, "gfx950", "--values", str(values))
    assert "N/A" in preview(capsys, "gfx950", "--empty")


def test_units_come_from_the_given_config(capsys):
    shipped = preview(capsys, "gfx950")
    # gfx942's config has no units for gfx950-only metrics such as HBM Read BW
    other = preview(capsys, "gfx950", "--config", str(GFX942_CONFIG))
    assert "Hit 42.0%" in other
    assert other.count("GB/s") < shipped.count("GB/s")
