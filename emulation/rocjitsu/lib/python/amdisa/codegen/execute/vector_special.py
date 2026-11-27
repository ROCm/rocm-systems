# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""VM relative-move execution and conversion-layout metadata.

Numerical conversion, arithmetic and TRIG_PREOP execution route through rj-arith.
"""

from __future__ import annotations


def _resolved_vgpr_offset_call(opnd: str, uses_vgpr_msb_indexing: bool) -> str:
    if uses_vgpr_msb_indexing:
        return (
            f'Isa::resolved_vgpr_offset(wf, {opnd}.opr_type_, '
            f'{opnd}.encoding_value_, {opnd}.vgpr_msb_role())'
        )
    return f'Isa::resolved_vgpr_offset({opnd}.opr_type_, {opnd}.encoding_value_)'


def _move_lanes(dst: str, src: str, is_vop3: bool = False) -> str:
    """Copy the resolved/staged operands with the encoding's inherited modifiers."""
    if not is_vop3:
        return f'  amdgpu::arith::execute_move32(*this, wf, {dst}, {src});'
    modifiers = (
        '((inst_.abs & 1) ? RJ_ARITH_ALU_ABS_A : 0u) | ((inst_.neg & 1) ? RJ_ARITH_ALU_NEG_A : 0u)'
        if is_vop3
        else '0u'
    )
    return (
        '  amdgpu::arith::execute_unary32(wf, rj_arith_v_mov_b32_w32, rj_arith_v_mov_b32_w64,\n'
        f'      wf.cu().arithmetic_flags(), {modifiers},\n'
        f'      {dst}, {src});'
    )


def gen_vector_movrel(
    dst: list[str],
    src: list[str],
    op: str | None,
    uses_vgpr_msb_indexing: bool = False,
    supports_dpp: bool = False,
    supports_dpp8: bool = False,
    supports_sdwa: bool = False,
    inactive_uses_bound_ctrl: bool = False,
    is_vop3: bool = False,
    stage_after_destination: bool = False,
    apply_dpp_source_modifiers: bool = True,
) -> str:
    """Generate M0-relative VGPR move bodies."""
    if op not in ('src', 'dst', 'srcdst', 'srcdst2'):
        raise ValueError(f'unsupported MOVREL operation: {op}')

    L = []
    if op == 'src':
        rel_src_call = _resolved_vgpr_offset_call(src[0], uses_vgpr_msb_indexing)
        L.append(f'  std::optional<uint32_t> rel_src_base = {rel_src_call};')
        L.append(
            '  uint64_t rel_src_index = rel_src_base '
            '? static_cast<uint64_t>(*rel_src_base) + wf.m0() : UINT64_MAX;'
        )
        L.append(
            f'  bool rel_src_valid = amdgpu::relative_vgpr_index(rel_src_base, wf.m0(), '
            f'{src[0]}.vgpr_count(), wf.vgpr_alloc().count, wf.cu().arch()).has_value();'
        )
        L.append(
            f'  Operand rel_src({src[0]}.size_bits(), OperandType::OPR_VGPR, '
            f'static_cast<int>(rel_src_valid ? rel_src_index : 0u));'
        )
        _stage_movrel_source(
            L,
            supports_dpp,
            supports_dpp8,
            supports_sdwa,
            inactive_uses_bound_ctrl,
            not is_vop3 and apply_dpp_source_modifiers,
        )
        L.append(_move_lanes(dst[0], 'rel_src', is_vop3))
        return '\n'.join(L)

    if op in ('srcdst', 'srcdst2'):
        rel_dst_call = _resolved_vgpr_offset_call(dst[0], uses_vgpr_msb_indexing)
        rel_src_call = _resolved_vgpr_offset_call(src[0], uses_vgpr_msb_indexing)
        L.append(f'  std::optional<uint32_t> rel_dst_base = {rel_dst_call};')
        L.append(f'  std::optional<uint32_t> rel_src_base = {rel_src_call};')
        if op == 'srcdst':
            L.append('  uint32_t rel_src_offset = wf.m0();')
            L.append('  uint32_t rel_dst_offset = wf.m0();')
        else:
            L.append('  uint32_t rel_src_offset = wf.m0() & 0x3ffu;')
            L.append('  uint32_t rel_dst_offset = (wf.m0() >> 16) & 0x3ffu;')
        L.append(
            '  uint64_t rel_src_index = rel_src_base '
            '? static_cast<uint64_t>(*rel_src_base) + rel_src_offset : UINT64_MAX;'
        )
        L.append(
            '  uint64_t rel_dst_index = rel_dst_base '
            '? static_cast<uint64_t>(*rel_dst_base) + rel_dst_offset : UINT64_MAX;'
        )
        L.append(
            f'  bool rel_src_valid = amdgpu::relative_vgpr_index(rel_src_base, rel_src_offset, '
            f'{src[0]}.vgpr_count(), wf.vgpr_alloc().count, wf.cu().arch()).has_value();'
        )
        L.append(
            f'  bool rel_dst_valid = amdgpu::relative_vgpr_index(rel_dst_base, rel_dst_offset, '
            f'{dst[0]}.vgpr_count(), wf.vgpr_alloc().count, wf.cu().arch()).has_value();'
        )
        L.append('  if (!rel_dst_valid) return;')
        L.append(
            f'  Operand rel_src({src[0]}.size_bits(), OperandType::OPR_VGPR, '
            f'static_cast<int>(rel_src_valid ? rel_src_index : 0u));'
        )
        L.append(
            f'  Operand rel_dst({dst[0]}.size_bits(), OperandType::OPR_VGPR, static_cast<int>(rel_dst_index));'
        )
        _stage_movrel_source(
            L,
            supports_dpp,
            supports_dpp8,
            supports_sdwa,
            inactive_uses_bound_ctrl,
            not is_vop3 and apply_dpp_source_modifiers,
        )
        L.append(_move_lanes('rel_dst', 'rel_src', is_vop3))
        return '\n'.join(L)

    rel_dst_call = _resolved_vgpr_offset_call(dst[0], uses_vgpr_msb_indexing)
    L.append(f'  std::optional<uint32_t> rel_dst_base = {rel_dst_call};')
    L.append(
        '  uint64_t rel_dst_index = rel_dst_base '
        '? static_cast<uint64_t>(*rel_dst_base) + wf.m0() : UINT64_MAX;'
    )
    L.append(
        f'  bool rel_dst_valid = amdgpu::relative_vgpr_index(rel_dst_base, wf.m0(), '
        f'{dst[0]}.vgpr_count(), wf.vgpr_alloc().count, wf.cu().arch()).has_value();'
    )
    L.append('  if (!rel_dst_valid) return;')
    L.append(
        f'  Operand rel_dst({dst[0]}.size_bits(), OperandType::OPR_VGPR, static_cast<int>(rel_dst_index));'
    )
    source = src[0]
    if stage_after_destination and (supports_dpp or supports_dpp8 or supports_sdwa):
        L.append(f'  Operand &rel_src = {source};')
        _stage_movrel_source(
            L,
            supports_dpp,
            supports_dpp8,
            supports_sdwa,
            inactive_uses_bound_ctrl,
            not is_vop3 and apply_dpp_source_modifiers,
        )
        source = 'rel_src'
    L.append(_move_lanes('rel_dst', source, is_vop3))
    return '\n'.join(L)


def _stage_movrel_source(
    lines: list[str],
    supports_dpp: bool,
    supports_dpp8: bool,
    supports_sdwa: bool,
    inactive_uses_bound_ctrl: bool,
    apply_source_modifiers: bool,
) -> None:
    """Stage modifiers from the M0-adjusted source VGPR, not its encoded base."""
    if not (supports_dpp or supports_dpp8 or supports_sdwa):
        return
    lines.append('  std::optional<StagedOperand> rel_staged_src;')
    if supports_dpp:
        inactive_policy = str(inactive_uses_bound_ctrl).lower()
        lines.extend(
            [
                '  amdgpu::dpp::DppPlan rel_dpp_plan;',
                '  if (inst_.src0 == amdgpu::SRC_DPP) {',
                '    rel_dpp_plan = amdgpu::dpp::make_dpp_plan(',
                '        wf.wf_size(), dpp_ctrl_, dpp_row_mask_, dpp_bank_mask_,',
                '        dpp_bound_ctrl_, dpp_fi_, wf.exec(),',
                f'        {inactive_policy});',
                '    amdgpu::dpp::apply_dpp(',
                '        rel_src, rel_dpp_plan, wf.exec(), rel_staged_src, wf);',
                '  }',
            ]
        )
    if supports_dpp8:
        lines.extend(
            [
                '  if (amdgpu::dpp::is_src_dpp8(inst_.src0))',
                '    amdgpu::dpp::apply_dpp8(rel_src, dpp8_lane_sel_, dpp_fi_, rel_staged_src, wf);',
            ]
        )
    if supports_sdwa:
        lines.extend(
            [
                '  if (inst_.src0 == amdgpu::SRC_SDWA)',
                '    amdgpu::sdwa::stage_source(rel_src, sdwa_src0_sel_, sdwa_src0_sext_,',
                '                               sdwa_src0_neg_, sdwa_src0_abs_,',
                '                               amdgpu::sdwa::SourceModifierFormat::NONE,',
                '                               rel_staged_src, wf);',
            ]
        )
    if supports_dpp and apply_source_modifiers:
        lines.append(
            '  amdgpu::dpp::apply_source_modifiers(rel_src, rel_staged_src, wf, '
            '1ULL << 31, dpp_modifiers_, 0);'
        )
    lines.append(
        '  ScopedOperandDelegate rel_staged_src_binding('
        'rel_src, rel_staged_src ? &*rel_staged_src : nullptr);'
    )


def gen_vector_swaprel(
    dst: list[str],
    src: list[str],
    uses_vgpr_msb_indexing: bool = False,
) -> str:
    """Generate V_SWAPREL_B32 using the two unsigned M0 index fields."""
    d, s = dst[0], src[0]
    d_call = _resolved_vgpr_offset_call(d, uses_vgpr_msb_indexing)
    s_call = _resolved_vgpr_offset_call(s, uses_vgpr_msb_indexing)
    L = [
        f'  std::optional<uint32_t> rel_dst_base = {d_call};',
        f'  std::optional<uint32_t> rel_src_base = {s_call};',
        '  uint32_t rel_src_offset = wf.m0() & 0x3ffu;',
        '  uint32_t rel_dst_offset = (wf.m0() >> 16) & 0x3ffu;',
        '  uint64_t rel_src_index = rel_src_base ? static_cast<uint64_t>(*rel_src_base) + rel_src_offset : UINT64_MAX;',
        '  uint64_t rel_dst_index = rel_dst_base ? static_cast<uint64_t>(*rel_dst_base) + rel_dst_offset : UINT64_MAX;',
        f'  if (!amdgpu::relative_vgpr_index(rel_src_base, rel_src_offset, {s}.vgpr_count(), wf.vgpr_alloc().count, wf.cu().arch())) return;',
        f'  if (!amdgpu::relative_vgpr_index(rel_dst_base, rel_dst_offset, {d}.vgpr_count(), wf.vgpr_alloc().count, wf.cu().arch())) return;',
        f'  Operand rel_src({s}.size_bits(), OperandType::OPR_VGPR, static_cast<int>(rel_src_index));',
        f'  Operand rel_dst({d}.size_bits(), OperandType::OPR_VGPR, static_cast<int>(rel_dst_index));',
    ]
    L.append(
        '  amdgpu::arith::execute_swap<false>(wf, wf.cu().arithmetic_flags(), rel_dst, rel_src);'
    )
    return '\n'.join(L)


def cvt_scale_layout(source_format: str) -> str:
    """CDNA5 ISA 7.6.3 selects the E8M0 layout by the packed source format."""
    if source_format in ('fp8', 'bf8'):
        return 'f8'
    if source_format in ('fp4', 'fp6', 'bf6'):
        return 'f4_f6'
    raise ValueError(f'unsupported scaled low-precision format: {source_format}')


# ---------------------------------------------------------------------------
# Non-scaled FP8/BF8 pack/convert instructions (CDNA3/4, RDNA4)
# ---------------------------------------------------------------------------
