#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Tests for the memory chart layout formatter."""

import json

from memory_chart_layout_format import format_layout

from memory_chart.loader import layout_files


def reversed_keys(value):
    """*value* with the key order of every object reversed."""
    if isinstance(value, dict):
        return {k: reversed_keys(value[k]) for k in reversed(list(value))}
    if isinstance(value, list):
        return [reversed_keys(v) for v in value]
    return value


def test_keys_are_written_in_loader_order_at_every_level():
    for path in layout_files():
        text = path.read_text(encoding="utf-8")
        assert format_layout(reversed_keys(json.loads(text))) == text, path.name
