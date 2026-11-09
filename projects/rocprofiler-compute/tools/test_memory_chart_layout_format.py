#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Tests for the memory chart layout formatter."""

import json

from memory_chart_layout_format import format_layout, with_metric_ids

from memory_chart.loader import layout_files, metric_entries


def reversed_keys(value):
    """*value* with the key order of every object reversed."""
    if isinstance(value, dict):
        return {k: reversed_keys(value[k]) for k in reversed(list(value))}
    if isinstance(value, list):
        return [reversed_keys(v) for v in value]
    return value


def test_keys_and_blocks_are_written_in_canonical_order():
    for path in layout_files():
        text = path.read_text(encoding="utf-8")
        data = reversed_keys(json.loads(text))
        data["blocks"].reverse()
        assert format_layout(data) == text, path.name


def test_metric_ids_are_filled_in_from_the_panel_config():
    for path in layout_files():
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
        for entry in metric_entries(data):
            del entry["metric_id"]
        assert with_metric_ids(path, data) == json.loads(text)
