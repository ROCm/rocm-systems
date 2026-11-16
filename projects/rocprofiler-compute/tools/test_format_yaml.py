#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Tests for YAML metric equation formatter."""

import textwrap
from typing import List, Set

import pytest
import yaml
from format_yaml import check_file, file_equations

import config
from rocprof_compute_soc.metric_library.loader import default_metric_library_dir
from utils.equation_format import (
    CHANNEL_PLACEHOLDER,
    LITERAL_VALUES,
    ASTNode,
    BinaryOpNode,
    FunctionCallNode,
    NumberNode,
    UnaryOpNode,
    normalize_equation,
    parse_equation,
)
from utils.metrics.expression import build_eval_string

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

# Panel collectables: top-level name: formula pairs, with one formula to format
COLLECTABLES = textwrap.dedent("""\
    # Named formulas that metric formulas use as $name.
    GRBM_GUI_ACTIVE_PER_XCD: GRBM_GUI_ACTIVE / $num_xcd
    hbm_write_bytes: ((TCC_EA0_WRREQ_sum-TCC_EA0_WRREQ_64B_sum)*32)
""")


def real_equations() -> Set[str]:
    """Return every equation of the panel configs, metric library and collectables."""
    files = list(config.analysis_configs_dir.glob("gfx*/*.yaml"))
    files += config.analysis_configs_dir.glob("collectables/*.yaml")
    files += default_metric_library_dir().glob("gfx*/*.yaml")
    equations = set()
    for path in files:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        equations.update(equation for _, equation in file_equations(path, document))
    return equations


def fully_parenthesized(node: ASTNode) -> str:
    """Write a syntax tree with parentheses around every operation and no spaces."""
    if isinstance(node, UnaryOpNode):
        return f"{node.op}({fully_parenthesized(node.operand)})"
    if isinstance(node, BinaryOpNode):
        left = fully_parenthesized(node.left)
        right = fully_parenthesized(node.right)
        return f"({left}{node.op}{right})"
    if isinstance(node, FunctionCallNode):
        return f"{node.name}({','.join(fully_parenthesized(arg) for arg in node.args)})"
    if isinstance(node, NumberNode):
        return node.value
    return node.name


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


def test_panel_collectables_fixed_in_place(tmp_path):
    """Each top-level value of a panel collectables file is an equation."""
    path = write_config(
        tmp_path, "analysis_configs/collectables/gfx942_collectables.yaml", COLLECTABLES
    )
    check_file(path, fix=True)
    assert path.read_text(encoding="utf-8") == COLLECTABLES.replace(
        "((TCC_EA0_WRREQ_sum-TCC_EA0_WRREQ_64B_sum)*32)",
        "(TCC_EA0_WRREQ_sum - TCC_EA0_WRREQ_64B_sum) * 32",
    )


def test_real_equations_round_trip():
    """Each real equation parses as Python reads it, and normalizing it with every
    parenthesis written out and no spaces gives back the equation."""
    failures: List[str] = []
    for equation in sorted(real_equations()):
        if equation in LITERAL_VALUES or CHANNEL_PLACEHOLDER in equation:
            continue
        written_out = fully_parenthesized(parse_equation(equation))
        normalized = normalize_equation(written_out)
        if build_eval_string(written_out) != build_eval_string(equation):
            failures.append(f"parsed as {written_out}: {equation}")
        elif normalized != equation:
            failures.append(f"normalized to {normalized}: {equation}")
    assert failures == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
