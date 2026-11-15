#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Tests for YAML metric equation formatter."""

import textwrap

import pytest
from format_yaml import check_file

# A panel with one equation that needs formatting and one equation
# with a dangling operator.
PANEL_WITH_BROKEN_EQUATION = textwrap.dedent("""\
    Panel Config:
      id: 400
      data source:
      - metric_table:
          id: 401
          metric:
            L2 Hit:
              value: SUM(TCC_HIT_sum * 2)
            AI LDS:
              value: SUM(SQ_INSTS_VALU_MFMA_MOPS_F64 * 512 + ) / SUM(SQ_LDS_IDX_ACTIVE)
""")


def write_config(tmp_path, relative_path, text):
    path = tmp_path / relative_path
    path.parent.mkdir(parents=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_unparseable_equation_reported(tmp_path):
    """A broken equation is reported by key; the other equations are still fixed."""
    panel = write_config(
        tmp_path,
        "analysis_configs/gfx942/0400_roofline.yaml",
        PANEL_WITH_BROKEN_EQUATION,
    )

    is_valid, issues = check_file(panel, fix=True)

    assert not is_valid
    assert issues[0].startswith(
        "  Panel Config.data source[0].metric_table.metric.AI LDS.value: "
        "cannot parse equation: "
    )
    content = panel.read_text(encoding="utf-8")
    assert "value: 2 * SUM(TCC_HIT_sum)" in content
    assert "MOPS_F64 * 512 + ) /" in content


def test_unformatted_library_formula_reported(tmp_path):
    """A metric library formula is checked like a panel equation."""
    path = write_config(
        tmp_path,
        "metric_library/gfx908/l2.yaml",
        "l2.ai:\n  formula: ( SUM( (a * 2) ) / SUM(b) )\n",
    )
    is_valid, issues = check_file(path)
    assert not is_valid
    assert "l2.ai.formula: equation needs normalization" in issues[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
