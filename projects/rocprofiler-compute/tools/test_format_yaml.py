#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Tests for YAML metric equation formatter."""

import textwrap

import pytest
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
