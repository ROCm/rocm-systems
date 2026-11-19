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

# A per-channel table and a simple-box table, each with a dangling operator in a
# metric entry; their headers hold display text under the same keys
PANEL_WITH_STD_DEV_AND_EXPR = textwrap.dedent("""\
    Panel Config:
      id: 1800
      data source:
      - metric_table:
          id: 1801
          header:
            metric: Metric
            avg: Avg
            std dev: Std Dev
          metric:
            L2 Cache Hit Rate:
              avg: 100 * SUM(TCC_HIT_sum) / SUM(TCC_HIT_sum + TCC_MISS_sum)
              std dev: 100 * STD(TCC_HIT_sum / ) )
      - metric_table:
          id: 1802
          header:
            metric: Metric
            expr: Expression
          metric:
            Stalled on L2 Data:
              expr: 100 * TCP_PENDING_STALL_CYCLES_sum /
""")

# Metric library descriptions: one too long for its line, one folded although it fits
# the key line, one folded too narrow, and an empty one
LDS_UTIL = (
    "Percentage of the kernel's duration the LDS was busy executing LDS instructions, "
    "averaged over compute units."
)
UNFILLED_DESCRIPTIONS = textwrap.dedent(f"""\
    lds.util:
      description: {LDS_UTIL}
    lds.insts:
      description: >-
        Number of LDS instructions issued.
    lds.bank_conflict_rate:
      description: >-
        Percentage of LDS cycles spent
        on bank conflicts. High conflict rates reduce effective LDS bandwidth.
      unit: pct
    lds.unused:
      description: ''
""")

FILLED_DESCRIPTIONS = textwrap.dedent("""\
    lds.util:
      description: >-
        Percentage of the kernel's duration the LDS was busy executing LDS instructions,
        averaged over compute units.
    lds.insts:
      description: Number of LDS instructions issued.
    lds.bank_conflict_rate:
      description: >-
        Percentage of LDS cycles spent on bank conflicts. High conflict rates reduce
        effective LDS bandwidth.
      unit: pct
    lds.unused:
      description: ''
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


def test_std_dev_and_expr_checked_but_not_headers(tmp_path):
    """std dev and simple-box expr values are equations; header text is not."""
    panel = write_config(
        tmp_path,
        "analysis_configs/gfx942/1800_l2_cache_per_channel.yaml",
        PANEL_WITH_STD_DEV_AND_EXPR,
    )
    _, issues = check_file(panel)
    assert [issue.split(": cannot parse")[0] for issue in issues] == [
        "  Panel Config.data source[0].metric_table.metric.L2 Cache Hit Rate.std dev",
        "  Panel Config.data source[1].metric_table.metric.Stalled on L2 Data.expr",
    ]


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


def test_library_descriptions_filled(tmp_path):
    """Metric library descriptions are filled to 88 columns, on the key line when they
    fit there."""
    path = write_config(
        tmp_path, "metric_library/gfx950/lds.yaml", UNFILLED_DESCRIPTIONS
    )
    is_valid, issues = check_file(path)
    assert not is_valid
    assert issues == ["  descriptions need filling to 88 columns"]
    check_file(path, fix=True)
    assert path.read_text(encoding="utf-8") == FILLED_DESCRIPTIONS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
