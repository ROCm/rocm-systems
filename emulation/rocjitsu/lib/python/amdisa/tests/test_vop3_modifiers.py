# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for VOP3 source and destination modifier helpers."""

from amdisa.codegen.execute.vop3_modifiers import (
    vop3_dst_mod,
    vop3_dst_mod_f64,
    vop3_src_mod,
)


class TestVop3SrcMod:
    """Tests for vop3_src_mod (input modifier: abs then neg)."""

    def test_has_abs_true_includes_abs_line(self):
        lines = vop3_src_mod('s', 0, has_abs=True)
        assert any('std::fabs' in line and '1u << 0' in line for line in lines)

    def test_has_abs_false_omits_abs_line(self):
        lines = vop3_src_mod('s', 0, has_abs=False)
        assert not any('std::fabs' in line for line in lines)

    def test_always_includes_neg_line(self):
        for has_abs in (True, False):
            lines = vop3_src_mod('s', 0, has_abs=has_abs)
            assert any('inst_.neg' in line and '= -s' in line for line in lines)

    def test_src_idx_zero(self):
        lines = vop3_src_mod('x', 0, has_abs=True)
        assert any('1u << 0' in line for line in lines)
        assert any('1u << 0' in line and 'inst_.neg' in line for line in lines)

    def test_src_idx_one(self):
        lines = vop3_src_mod('x', 1, has_abs=True)
        assert any('1u << 1' in line for line in lines)
        assert any('1u << 1' in line and 'inst_.neg' in line for line in lines)

    def test_src_idx_two(self):
        lines = vop3_src_mod('x', 2, has_abs=True)
        assert any('1u << 2' in line for line in lines)
        assert any('1u << 2' in line and 'inst_.neg' in line for line in lines)

    def test_varname_appears_in_lines(self):
        lines = vop3_src_mod('myvar', 0, has_abs=True)
        assert all('myvar' in line for line in lines)

    def test_has_abs_true_returns_two_lines(self):
        lines = vop3_src_mod('s', 0, has_abs=True)
        assert len(lines) == 2

    def test_has_abs_false_returns_one_line(self):
        lines = vop3_src_mod('s', 0, has_abs=False)
        assert len(lines) == 1

    def test_default_indent(self):
        lines = vop3_src_mod('s', 0, has_abs=True)
        assert all(line.startswith('    ') for line in lines)

    def test_custom_indent(self):
        lines = vop3_src_mod('s', 0, has_abs=True, indent='  ')
        assert all(line.startswith('  ') for line in lines)

    def test_works_for_float_varname(self):
        lines = vop3_src_mod('sv0', 0, has_abs=True)
        assert any('sv0' in line and 'std::fabs' in line for line in lines)

    def test_works_for_double_varname(self):
        # vop3_src_mod is type-generic (uses std::fabs and unary negation)
        lines = vop3_src_mod('d', 0, has_abs=True)
        assert any('d' in line and 'std::fabs' in line for line in lines)


class TestVop3DstMod:
    """Tests for vop3_dst_mod (float output modifier: omod then clamp)."""

    def test_returns_four_lines(self):
        lines = vop3_dst_mod('result')
        assert len(lines) == 4

    def test_omod_1_multiplies_by_2f(self):
        lines = vop3_dst_mod('result')
        assert any('omod == 1' in line and '*= 2.0f' in line for line in lines)

    def test_omod_2_multiplies_by_4f(self):
        lines = vop3_dst_mod('result')
        assert any('omod == 2' in line and '*= 4.0f' in line for line in lines)

    def test_omod_3_multiplies_by_half(self):
        lines = vop3_dst_mod('result')
        assert any('omod == 3' in line and '*= 0.5f' in line for line in lines)

    def test_clamp_uses_float_literals(self):
        lines = vop3_dst_mod('result')
        assert any(
            'clamp' in line and '0.0f' in line and '1.0f' in line for line in lines
        )

    def test_varname_appears_in_all_lines(self):
        lines = vop3_dst_mod('myresult')
        assert all('myresult' in line for line in lines)

    def test_default_indent(self):
        lines = vop3_dst_mod('result')
        assert all(line.startswith('    ') for line in lines)

    def test_custom_indent(self):
        lines = vop3_dst_mod('result', indent='  ')
        assert all(line.startswith('  ') for line in lines)


class TestVop3DstModF64:
    """Tests for vop3_dst_mod_f64 (double output modifier: omod then clamp)."""

    def test_returns_four_lines(self):
        lines = vop3_dst_mod_f64('result')
        assert len(lines) == 4

    def test_omod_1_multiplies_by_2(self):
        lines = vop3_dst_mod_f64('result')
        assert any('omod == 1' in line and '*= 2.0;' in line for line in lines)

    def test_omod_2_multiplies_by_4(self):
        lines = vop3_dst_mod_f64('result')
        assert any('omod == 2' in line and '*= 4.0;' in line for line in lines)

    def test_omod_3_multiplies_by_half(self):
        lines = vop3_dst_mod_f64('result')
        assert any('omod == 3' in line and '*= 0.5;' in line for line in lines)

    def test_clamp_uses_double_literals(self):
        lines = vop3_dst_mod_f64('result')
        assert any(
            'clamp' in line and '0.0,' in line and '1.0)' in line for line in lines
        )

    def test_does_not_use_float_suffix(self):
        lines = vop3_dst_mod_f64('result')
        assert not any(
            '0.0f' in line or '1.0f' in line or '2.0f' in line for line in lines
        )

    def test_varname_appears_in_all_lines(self):
        lines = vop3_dst_mod_f64('myresult')
        assert all('myresult' in line for line in lines)

    def test_default_indent(self):
        lines = vop3_dst_mod_f64('result')
        assert all(line.startswith('    ') for line in lines)

    def test_custom_indent(self):
        lines = vop3_dst_mod_f64('result', indent='  ')
        assert all(line.startswith('  ') for line in lines)
