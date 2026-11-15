#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Tests for YAML metric equation formatter."""

import textwrap

from format_yaml import check_file

# A metric library file with a formula, a peak and a measured peak that need
# formatting, and a formula with a dangling operator.
LIBRARY_WITH_BROKEN_FORMULA = textwrap.dedent("""\
    l2.hit:
      formula: SUM(TCC_HIT_sum * 2)
      peak: ( $max_sclk * 2 )
      peak_empirical: ( $HBMBw_empirical_peak )
    lds.ai:
      formula: SUM(SQ_INSTS_VALU_MFMA_MOPS_F64 * 512 + ) / SUM(SQ_LDS_IDX_ACTIVE)
""")


def write_config(tmp_path, relative_path, text):
    path = tmp_path / relative_path
    path.parent.mkdir(parents=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_library_equations_reported_and_fixed(tmp_path):
    """Each equation key is checked: a broken equation is reported by key, and the
    others need normalization; a fix rewrites them and keeps the broken one."""
    path = write_config(
        tmp_path, "metric_library/gfx942/l2.yaml", LIBRARY_WITH_BROKEN_FORMULA
    )

    is_valid, issues = check_file(path)
    assert not is_valid
    assert issues[0].startswith("  lds.ai.formula: cannot parse equation: ")
    assert issues[1:] == [
        f"  l2.hit.{key}: equation needs normalization"
        for key in ("formula", "peak", "peak_empirical")
    ]

    is_valid, issues = check_file(path, fix=True)
    assert not is_valid
    assert len(issues) == 1
    content = path.read_text(encoding="utf-8")
    assert "formula: 2 * SUM(TCC_HIT_sum)" in content
    assert "peak: $max_sclk * 2" in content
    assert "peak_empirical: $HBMBw_empirical_peak" in content
    assert "MOPS_F64 * 512 + ) /" in content
