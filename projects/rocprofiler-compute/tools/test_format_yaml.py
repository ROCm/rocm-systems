#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Tests for YAML metric equation formatter."""

import textwrap

import pytest
from format_yaml import check_file

# A panel with one per-channel equation that needs formatting and one equation
# with a dangling operator.
PANEL_WITH_BROKEN_EQUATION = textwrap.dedent("""\
    Panel Config:
      id: 400
      data source:
      - metric_table:
          id: 401
          metric:
            L2 Hit:
              value: SUM(TCC_HIT[0] * 2)
            AI LDS:
              value: SUM(SQ_INSTS_VALU_MFMA_MOPS_F64 * 512 + ) / SUM(SQ_LDS_IDX_ACTIVE)
""")


class TestErrorHandling:
    """Error reporting."""

    def test_unparseable_equation_reported(self, tmp_path):
        """A broken equation is reported by key; the other equations are still fixed."""
        panel = tmp_path / "analysis_configs" / "gfx942" / "0400_roofline.yaml"
        panel.parent.mkdir(parents=True)
        panel.write_text(PANEL_WITH_BROKEN_EQUATION, encoding="utf-8")

        is_valid, issues = check_file(panel, fix=True)

        assert not is_valid
        assert issues[0].startswith(
            "  Panel Config.data source[0].metric_table.metric.AI LDS.value: "
            "cannot parse equation: "
        )
        content = panel.read_text(encoding="utf-8")
        assert "value: 2 * SUM(TCC_HIT[0])" in content
        assert "MOPS_F64 * 512 + ) /" in content


class TestMetricLibraryFiles:
    """Metric library files: formula and peak_empirical are checked like panel
    equations."""

    def write_library_file(self, tmp_path, text):
        path = tmp_path / "metric_library" / "gfx908" / "l2.yaml"
        path.parent.mkdir(parents=True)
        path.write_text(text, encoding="utf-8")
        return path

    def test_unformatted_formula_fails(self, tmp_path):
        path = self.write_library_file(
            tmp_path, "l2.ai:\n  formula: ( SUM( (a * 2) ) / SUM(b) )\n"
        )
        is_valid, issues = check_file(path)
        assert not is_valid
        assert "l2.ai.formula: equation needs normalization" in issues[0]

    def test_formatted_formula_template_and_remove_list_pass(self, tmp_path):
        path = self.write_library_file(
            tmp_path,
            "remove:\n  - l2.old\n"
            "l2.ai:\n  formula: SUM(a) / SUM(b)\n"
            "  peak_empirical: $L2Bw_empirical_peak\n"
            "l2.chan_lat:\n"
            "  formula: A[::_1] / B[::_1] if B[::_1] != 0 else 0\n",
        )
        assert check_file(path) == (True, [])

    def test_fix_rewrites_formula(self, tmp_path):
        path = self.write_library_file(tmp_path, "l2.ai:\n  formula: (a+b)\n")
        check_file(path, fix=True)
        assert path.read_text(encoding="utf-8") == "l2.ai:\n  formula: a + b\n"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
