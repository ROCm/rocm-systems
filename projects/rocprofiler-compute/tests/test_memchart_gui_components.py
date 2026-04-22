# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Unit tests for utils/gui_components/memchart.py - Dash/web-UI memory chart.

Regression coverage for commit 43ad1138d1, which corrected the data lookup
in insert_chart_data() and the visible label in get_memchart() from the
legacy "Matrix Ops" string to the panel-YAML key "MFMA".
"""

from typing import Any, Iterator

import pandas as pd
import pytest
from dash_svg import Text

from utils import schema
from utils.gui_components import memchart as gui_memchart

MEM_CHART_TABLE_ID = 301


def _make_workload(
    rows: list[tuple[str, Any]],
) -> tuple[list[dict[str, Any]], schema.Workload]:
    """Build a minimal (mem_data, base_data) fixture for insert_chart_data."""
    df = pd.DataFrame(rows, columns=["Metric", "Value"])
    base_data = schema.Workload(dfs={MEM_CHART_TABLE_ID: df})
    mem_data = [{"metric_table": {"id": MEM_CHART_TABLE_ID}}]
    return mem_data, base_data


def _walk_text(node: Any) -> Iterator[Text]:  # noqa: ANN401
    """Recursively yield every dash_svg.Text in a Dash component tree."""
    if isinstance(node, Text):
        yield node
    children = getattr(node, "children", None)
    if children is None or isinstance(children, str):
        return
    if isinstance(children, (list, tuple)):
        for child in children:
            yield from _walk_text(child)
    else:
        yield from _walk_text(children)


def _find_text_by_id(root: Any, text_id: str) -> Text:  # noqa: ANN401
    """Return the single Text element whose id matches text_id."""
    matches = [t for t in _walk_text(root) if getattr(t, "id", None) == text_id]
    assert len(matches) == 1, (
        f"expected exactly one Text with id={text_id!r}, found {len(matches)}"
    )
    return matches[0]


# =============================================================================
# Regression: insert_chart_data routes the YAML "MFMA" value to id="matrix_ops"
# =============================================================================


class TestInsertChartDataMfma:
    """Regression for the insert_chart_data MFMA lookup fix (commit 43ad1138d1)."""

    def test_mfma_value_lookup(self):
        """The 'MFMA' value flows into the Text element with id='matrix_ops'."""
        mem_data, base_data = _make_workload([("MFMA", 42)])
        result = gui_memchart.insert_chart_data(mem_data, base_data)
        matrix_ops = _find_text_by_id(result, "matrix_ops")
        assert matrix_ops.children == "42"

    def test_legacy_matrix_ops_key_ignored(self):
        """Legacy 'Matrix Ops' key must NOT be used as a fallback lookup.

        Prior to commit 43ad1138d1 the renderer looked up
        memchart_values.get('Matrix Ops'), so a legacy key would have
        rendered through. After the fix it only honors 'MFMA'.
        """
        mem_data, base_data = _make_workload([("Matrix Ops", 99)])
        result = gui_memchart.insert_chart_data(mem_data, base_data)
        matrix_ops = _find_text_by_id(result, "matrix_ops")
        assert matrix_ops.children == "N/A"

    def test_missing_mfma_renders_na(self):
        """When MFMA is absent entirely, matrix_ops renders as 'N/A'."""
        mem_data, base_data = _make_workload([("SALU", 1)])
        result = gui_memchart.insert_chart_data(mem_data, base_data)
        matrix_ops = _find_text_by_id(result, "matrix_ops")
        assert matrix_ops.children == "N/A"

    @pytest.mark.parametrize(
        ("yaml_key", "text_id", "value"),
        [
            ("SALU", "salu", 11),
            ("SMEM", "smem", 22),
            ("VALU", "valu", 33),
            ("VMEM", "vmem", 55),
            ("LDS", "lds", 66),
            ("GWS", "gws", 77),
            ("BR", "br", 88),
        ],
    )
    def test_other_instr_ids_unaffected(self, yaml_key: str, text_id: str, value: int):
        """Sanity check: every other InstrDispatch row keeps its YAML mapping."""
        mem_data, base_data = _make_workload([(yaml_key, value)])
        result = gui_memchart.insert_chart_data(mem_data, base_data)
        text = _find_text_by_id(result, text_id)
        assert text.children == str(value)


# =============================================================================
# Regression: get_memchart visible label is "MFMA:" (not "Matrix Ops:")
# =============================================================================


class TestGetMemchartLabels:
    """Regression for the get_memchart label fix (commit 43ad1138d1)."""

    def test_mfma_label_present(self):
        """The InstrDispatch column header includes a 'MFMA:' label."""
        mem_data, base_data = _make_workload([("MFMA", 42)])
        section = gui_memchart.get_memchart(mem_data, base_data)
        labels = [t.children for t in _walk_text(section)]
        assert "MFMA:" in labels

    def test_legacy_matrix_ops_label_absent(self):
        """The legacy 'Matrix Ops:' label must not appear anywhere."""
        mem_data, base_data = _make_workload([("MFMA", 42)])
        section = gui_memchart.get_memchart(mem_data, base_data)
        labels = [t.children for t in _walk_text(section)]
        assert "Matrix Ops:" not in labels
