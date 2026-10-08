# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for VOP3 source and destination modifier helpers."""

import pytest

from amdisa.codegen.execute.vop3_modifiers import (
    vop3_dst_mod,
    vop3_src_mod,
)


class TestVop3SrcMod:
    """The generator forwards each encoding's modifier fields to the shared helper."""

    @pytest.mark.parametrize('has_abs', [False, True])
    @pytest.mark.parametrize('src_idx', [0, 1, 2])
    def test_forwards_source_index_and_available_fields(self, has_abs, src_idx):
        abs_field = 'inst_.abs' if has_abs else '0u'
        assert vop3_src_mod('source', src_idx, has_abs) == [
            '    source = amdgpu::source_modifier::apply_to_float('
            f'source, {src_idx}, {abs_field}, inst_.neg);'
        ]

    def test_custom_indent(self):
        lines = vop3_src_mod('source', 0, has_abs=True, indent='  ')
        assert lines[0].startswith('  source = ')


class TestVop3DstMod:
    """Tests for vop3_dst_mod (float output modifier: omod then clamp)."""

    def test_returns_six_lines(self):
        lines = vop3_dst_mod('result')
        assert len(lines) == 6

    def test_f32_omod_uses_f32_mode_policy(self):
        lines = vop3_dst_mod('result')
        assert 'effective_omod' in lines[0]
        assert 'fp_denorm_mode_f32()' in lines[0]

    def test_f16_omod_uses_f16_mode_policy(self):
        lines = vop3_dst_mod('result', omod_result_type='f16')
        assert 'effective_f16_omod' in lines[0]
        assert 'fp_denorm_mode_f16_f64()' in lines[0]
        assert 'false, inst_.omod' in lines[0]

    def test_rejects_bf16_until_post_narrow_finalization_is_owned(self):
        with pytest.raises(
            ValueError, match='unsupported VOP3 OMOD result policy type'
        ):
            vop3_dst_mod('result', omod_result_type='bf16')

    def test_rejects_unknown_result_type(self):
        with pytest.raises(
            ValueError, match='unsupported VOP3 OMOD result policy type'
        ):
            vop3_dst_mod('result', omod_result_type='f8')

    def test_omod_1_multiplies_by_2f(self):
        lines = vop3_dst_mod('result')
        assert any('omod == 1' in line and '*= 2.0f' in line for line in lines)

    def test_omod_2_multiplies_by_4f(self):
        lines = vop3_dst_mod('result')
        assert any('omod == 2' in line and '*= 4.0f' in line for line in lines)

    def test_omod_3_multiplies_by_half(self):
        lines = vop3_dst_mod('result')
        assert any('omod == 3' in line and '*= 0.5f' in line for line in lines)

    def test_clamp_uses_shared_architecture_policy(self):
        lines = vop3_dst_mod('result')
        assert any('clamp_floating_result(result, wf)' in line for line in lines)

    def test_active_omod_result_is_finalized(self):
        lines = vop3_dst_mod('result')
        assert any(
            'finalize_omod_f32(result, effective_omod)' in line for line in lines
        )

    def test_varname_appears_in_all_lines(self):
        lines = vop3_dst_mod('myresult')
        assert all('myresult' in line for line in lines[1:])

    def test_default_indent(self):
        lines = vop3_dst_mod('result')
        assert all(line.startswith('    ') for line in lines)

    def test_custom_indent(self):
        lines = vop3_dst_mod('result', indent='  ')
        assert all(line.startswith('  ') for line in lines)
