# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Vector special operation execute body generators.

Free functions that emit C++ execute_impl bodies for specialized vector
instructions, including relative moves, packed permutations, QSAD/MQSAD,
saturating packs, arithmetic, lane permutations, and packed type conversion.
"""

from __future__ import annotations

from amdisa.codegen.execute import input_policy
from amdisa.codegen.execute.conversion import (
    decode_fp8_expr,
    pack_b32_f16_expr,
    pk_rtz_f16_expr,
    pk_u8_expr,
    to_fp8_expr,
)
from amdisa.codegen.execute.fp8_formats import (
    fp8_helper_name,
    uses_shared_fp8_narrowing,
    uses_shared_fp8_widening,
)
from amdisa.codegen.execute.required_vgpr import append_required_vgpr_offset
from amdisa.codegen.execute.vop3_modifiers import (
    apply_output,
    output_policy_decl,
    vop3_src_mod,
    vop3_dst_mod,
)

_F32_TO_FP8_MODE_RNE = {
    'util::f32_to_fp8_e4m3_rne': 'util::f32_to_fp8_e4m3_rne_mode',
    'util::f32_to_bf8_e5m2_rne': 'util::f32_to_bf8_e5m2_rne_mode',
    'util::f32_to_fp8_e5m3_rne': 'util::f32_to_fp8_e5m3_rne_mode',
}

_F32_TO_FP8_MODE_SR = {
    'util::f32_to_fp8_e4m3_sr': 'util::f32_to_fp8_e4m3_sr_mode',
    'util::f32_to_bf8_e5m2_sr': 'util::f32_to_bf8_e5m2_sr_mode',
    'util::f32_to_fp8_e5m3_sr': 'util::f32_to_fp8_e5m3_sr_mode',
}


def _fp8_mode_rne_helper_name(arch_name: str, name: str) -> str:
    return fp8_helper_name(arch_name, _F32_TO_FP8_MODE_RNE[name])


def _fp8_mode_sr_helper_name(arch_name: str, name: str) -> str:
    return fp8_helper_name(arch_name, _F32_TO_FP8_MODE_SR[name])


def _fp8_rne_encode_call(cvt_fn: str, value_expr: str, use_fp16_ovfl: bool) -> str:
    if use_fp16_ovfl:
        return f'{cvt_fn}({value_expr}, wf.fp16_ovfl())'
    return f'{cvt_fn}({value_expr})'


def _shared_inst_operand(opnd: str) -> bool:
    return opnd.startswith('inst.')


def _default_vop3_opsel_expr(operands: list[str]) -> str:
    """Return the VOP3 OPSEL expression for local or shared execute bodies."""
    return (
        '::rocjitsu::amdgpu::vop3_opsel(inst.inst_)'
        if any(_shared_inst_operand(opnd) for opnd in operands)
        else 'amdgpu::vop3_opsel(inst_)'
    )


def _read_vop3_true16_src(opnd: str, opsel: str, src_idx: int) -> str:
    return (
        f'::rocjitsu::amdgpu::read_vop3_true16_src'
        f'({opnd}, wf, lane, {opsel}, {src_idx})'
    )


def _resolved_vgpr_offset_call(opnd: str, uses_vgpr_msb_indexing: bool) -> str:
    if uses_vgpr_msb_indexing:
        return (
            f'Isa::resolved_vgpr_offset(wf, {opnd}.opr_type_, '
            f'{opnd}.encoding_value_, {opnd}.vgpr_msb_role())'
        )
    return f'Isa::resolved_vgpr_offset({opnd}.opr_type_, {opnd}.encoding_value_)'


def _move_lanes(dst: str, src: str, use_rj_arith: bool, is_vop3: bool = False) -> str:
    if use_rj_arith:
        # Addressing and DPP source staging have already selected the operands.
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
    return '\n'.join(
        [
            '  uint64_t exec = wf.exec();',
            '  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {',
            '    if (!(exec & (1ULL << lane))) continue;',
            f'    amdgpu::RegisterAccess(wf).write_lane({dst}, lane, amdgpu::RegisterAccess(wf).read_lane({src}, lane));',
            '  }',
        ]
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
    use_rj_arith: bool = False,
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
            use_rj_arith and not is_vop3 and apply_dpp_source_modifiers,
        )
        L.append(_move_lanes(dst[0], 'rel_src', use_rj_arith, is_vop3))
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
            use_rj_arith and not is_vop3 and apply_dpp_source_modifiers,
        )
        L.append(_move_lanes('rel_dst', 'rel_src', use_rj_arith, is_vop3))
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
            use_rj_arith and not is_vop3,
        )
        source = 'rel_src'
    L.append(_move_lanes('rel_dst', source, use_rj_arith, is_vop3))
    return '\n'.join(L)


def _stage_movrel_source(
    lines: list[str],
    supports_dpp: bool,
    supports_dpp8: bool,
    supports_sdwa: bool,
    inactive_uses_bound_ctrl: bool,
    use_rj_arith: bool,
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
    if supports_dpp and use_rj_arith:
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
    use_rj_arith: bool = False,
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
    if use_rj_arith:
        L += [
            '  amdgpu::arith::execute_swap<false>(wf, wf.cu().arithmetic_flags(), rel_dst, rel_src);',
        ]
        return '\n'.join(L)
    L += [
        '  uint64_t exec = wf.exec();',
        '  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {',
        '    if (!(exec & (1ULL << lane))) continue;',
        '    uint32_t dst_value = amdgpu::RegisterAccess(wf).read_lane(rel_dst, lane);',
        '    uint32_t src_value = amdgpu::RegisterAccess(wf).read_lane(rel_src, lane);',
        '    amdgpu::RegisterAccess(wf).write_lane(rel_dst, lane, src_value);',
        '    amdgpu::RegisterAccess(wf).write_lane(rel_src, lane, dst_value);',
        '  }',
    ]
    return '\n'.join(L)


def gen_vector_sat_pack(
    dst: list[str], src: list[str], op: str | None, *, true16_dst: bool = False
) -> str:
    """Generate signed/unsigned saturating narrow-and-pack operations.

    V_SAT_PK_U8_I16 has a 16-bit result. With ``true16_dst`` its VOP3 form
    writes the half selected by OP_SEL[3] and keeps the other half.
    """
    if op not in ('u8_i16', 'i4_i8', 'u4_u8'):
        raise ValueError(f'unsupported saturating-pack operation: {op}')
    L = [
        '  uint64_t exec = wf.exec();',
        '  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {',
        '    if (!(exec & (1ULL << lane))) continue;',
        f'    uint32_t raw = amdgpu::RegisterAccess(wf).read_lane({src[0]}, lane);',
    ]
    if op == 'u8_i16':
        L.extend(
            [
                '    int32_t lo = static_cast<int16_t>(raw);',
                '    int32_t hi = static_cast<int16_t>(raw >> 16);',
                '    uint32_t result = static_cast<uint32_t>(std::clamp(lo, 0, 255)) |',
                '                      (static_cast<uint32_t>(std::clamp(hi, 0, 255)) << 8);',
            ]
        )
    elif op == 'i4_i8':
        L.extend(
            [
                '    uint32_t result = 0;',
                '    for (uint32_t i = 0; i < 4; ++i) {',
                '      int32_t value = static_cast<int8_t>(raw >> (i * 8));',
                '      result |= (static_cast<uint32_t>(std::clamp(value, -8, 7)) & 0xfu) << (i * 4);',
                '    }',
            ]
        )
    elif op == 'u4_u8':
        L.extend(
            [
                '    uint32_t result = 0;',
                '    for (uint32_t i = 0; i < 4; ++i)',
                '      result |= std::min((raw >> (i * 8)) & 0xffu, 15u) << (i * 4);',
            ]
        )
    if true16_dst and op == 'u8_i16':
        write = (
            f'    ::rocjitsu::amdgpu::write_vop3_true16_dst({dst[0]}, wf, lane, '
            'amdgpu::vop3_opsel(inst_), result, true);'
        )
    else:
        write = f'    amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, result);'
    L.extend(
        [
            write,
            '  }',
        ]
    )
    return '\n'.join(L)


def gen_vector_perm_pk16(
    dst: list[str], src: list[str], op: str | None, uses_vgpr_msb_indexing: bool
) -> str:
    """Generate the 16-selector packed table permutation family."""
    widths = {'b4': (4, 2), 'b6': (6, 3), 'b8': (8, 4)}
    if op not in widths:
        raise ValueError(f'unsupported PERM_PK16 operation: {op}')
    elem_bits, words = widths[op]
    L = [
        '  uint64_t exec = wf.exec();',
        '  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {',
        '    if (!(exec & (1ULL << lane))) continue;',
    ]
    if op == 'b4':
        L.extend(
            [
                f'    uint64_t table = amdgpu::RegisterAccess(wf).read_lane({src[1]}, lane) |',
                f'                     (static_cast<uint64_t>(amdgpu::RegisterAccess(wf).read_lane({src[0]}, lane)) << 32);',
                f'    uint64_t selectors = amdgpu::RegisterAccess(wf).read_lane64({src[2]}, lane);',
                '    uint64_t packed = 0;',
                '    for (uint32_t i = 0; i < 16; ++i) {',
                '      uint32_t index = (selectors >> (i * 4)) & 0xfu;',
                '      packed |= ((table >> (index * 4)) & 0xfu) << (i * 4);',
                '    }',
                f'    amdgpu::RegisterAccess(wf).write_lane64({dst[0]}, lane, packed);',
            ]
        )
    else:
        L.append(f'    uint32_t table[{words}] = {{}};')
        L.append(
            f'    uint64_t s1 = amdgpu::RegisterAccess(wf).read_lane64({src[1]}, lane);'
        )
        L.append('    table[0] = static_cast<uint32_t>(s1);')
        L.append('    table[1] = static_cast<uint32_t>(s1 >> 32);')
        if op == 'b6':
            L.append(
                f'    table[2] = amdgpu::RegisterAccess(wf).read_lane({src[0]}, lane);'
            )
        else:
            L.append(
                f'    uint64_t s0 = amdgpu::RegisterAccess(wf).read_lane64({src[0]}, lane);'
            )
            L.append('    table[2] = static_cast<uint32_t>(s0);')
            L.append('    table[3] = static_cast<uint32_t>(s0 >> 32);')
        L.extend(
            [
                f'    uint64_t selectors = amdgpu::RegisterAccess(wf).read_lane64({src[2]}, lane);',
                f'    uint32_t packed[{words}] = {{}};',
                '    for (uint32_t i = 0; i < 16; ++i) {',
                '      uint32_t index = (selectors >> (i * 4)) & 0xfu;',
                f'      uint32_t source_bit = index * {elem_bits}u;',
                '      uint32_t source_word = source_bit / 32;',
                '      uint32_t source_shift = source_bit % 32;',
                '      uint64_t pair = table[source_word];',
                f'      if (source_word + 1 < {words}u) pair |= static_cast<uint64_t>(table[source_word + 1]) << 32;',
                f'      uint32_t value = (pair >> source_shift) & ((1u << {elem_bits}) - 1u);',
                f'      uint32_t dest_bit = i * {elem_bits}u;',
                '      uint32_t dest_word = dest_bit / 32;',
                '      uint32_t dest_shift = dest_bit % 32;',
                '      packed[dest_word] |= value << dest_shift;',
                f'      if (dest_shift + {elem_bits}u > 32u)',
                '        packed[dest_word + 1] |= value >> (32u - dest_shift);',
                '    }',
                f'    Operand dst_word = {dst[0]};',
                '    dst_word.size_bits_ = 32;',
                f'    for (uint32_t word = 0; word < {words}u; ++word) {{',
                f'      dst_word.encoding_value_ = {dst[0]}.encoding_value_ + static_cast<int>(word);',
                '      amdgpu::RegisterAccess(wf).write_lane(dst_word, lane, packed[word]);',
                '    }',
            ]
        )
    L.append('  }')
    return '\n'.join(L)


def gen_vector_qsad(
    dst: list[str],
    src: list[str],
    op: str | None,
    uses_vgpr_msb_indexing: bool,
    integer_clamp: bool = False,
    whole_valu_source: bool = False,
) -> str:
    """Generate QSAD/MQSAD four-window sum-of-absolute-difference ops."""
    is_packed = op in ('sad_pk_u16', 'msad_pk_u16')
    is_masked = op in ('msad_pk_u16', 'msad_u32')
    if not is_packed and op != 'msad_u32':
        raise ValueError(f'unsupported QSAD operation: {op}')
    L = [
        '  uint64_t exec = wf.exec();',
        '  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {',
        '    if (!(exec & (1ULL << lane))) continue;',
        f'    uint64_t source = amdgpu::RegisterAccess(wf).read_lane64({src[0]}, lane);',
        f'    uint32_t reference = amdgpu::RegisterAccess(wf).read_lane({src[1]}, lane);',
    ]
    if is_packed:
        L.extend(
            [
                f'    uint64_t accum = amdgpu::RegisterAccess(wf).read_lane64({src[2]}, lane);',
                '    uint64_t packed_result = 0;',
            ]
        )
    else:
        L.extend(
            [
                '    uint32_t accum[4];',
                f'    if (std::optional<uint64_t> constant = {src[2]}.const_value()) {{',
                '      accum[0] = static_cast<uint32_t>(*constant);',
                '      accum[1] = static_cast<uint32_t>(*constant >> 32);',
                '      accum[2] = 0;',
                '      accum[3] = 0;',
                *(
                    [
                        f'    }} else if (auto region = amdgpu::RegisterAccess(wf).read_operand_region({src[2]}, uint64_t{{1}} << lane, 4)) {{',
                        '      for (uint32_t word = 0; word < 4; ++word)',
                        '        accum[word] = region->lane(word, lane);',
                    ]
                    if whole_valu_source
                    else []
                ),
                '    } else {',
                f'      Operand accum_word = {src[2]};',
                '      accum_word.size_bits_ = 32;',
                '      for (uint32_t window = 0; window < 4; ++window) {',
                f'        accum_word.encoding_value_ = {src[2]}.encoding_value_ + static_cast<int>(window);',
                '        accum[window] = amdgpu::RegisterAccess(wf).read_lane(accum_word, lane);',
                '      }',
                '    }',
                f'    Operand dst_word = {dst[0]};',
                '    dst_word.size_bits_ = 32;',
            ]
        )
    L.extend(
        [
            '    for (uint32_t window = 0; window < 4; ++window) {',
            '      uint32_t sum = 0;',
            '      for (uint32_t byte = 0; byte < 4; ++byte) {',
            '        uint32_t a = (source >> ((window + byte) * 8)) & 0xffu;',
            '        uint32_t b = (reference >> (byte * 8)) & 0xffu;',
        ]
    )
    if is_masked:
        L.append('        if (b != 0) sum += a > b ? a - b : b - a;')
    else:
        L.append('        sum += a > b ? a - b : b - a;')
    L.extend(['      }'])
    if is_packed:
        value_expr = (
            'inst_.clamp ? std::min(value, 0xffffu) : (value & 0xffffu)'
            if integer_clamp
            else 'value & 0xffffu'
        )
        L.extend(
            [
                '      uint32_t value = sum + ((accum >> (window * 16)) & 0xffffu);',
                f'      packed_result |= static_cast<uint64_t>({value_expr}) << (window * 16);',
            ]
        )
    else:
        value_expr = (
            'amdgpu::vop3_integer_add<uint32_t>(accum[window], sum, inst_.clamp)'
            if integer_clamp
            else 'accum[window] + sum'
        )
        L.extend(
            [
                f'      dst_word.encoding_value_ = {dst[0]}.encoding_value_ + static_cast<int>(window);',
                f'      amdgpu::RegisterAccess(wf).write_lane(dst_word, lane, {value_expr});',
            ]
        )
    L.append('    }')
    if is_packed:
        L.append(
            f'    amdgpu::RegisterAccess(wf).write_lane64({dst[0]}, lane, packed_result);'
        )
    L.append('  }')
    return '\n'.join(L)


_TRIG_PREOP_TWO_OVER_PI_CHUNKS = (
    0xA2F983,
    0x6E4E44,
    0x1529FC,
    0x2757D1,
    0xF534DD,
    0xC0DB62,
    0x95993C,
    0x439041,
    0xFE5163,
    0xABDEBB,
    0xC561B7,
    0x246E3A,
    0x424DD2,
    0xE00649,
    0x2EEA09,
    0xD1921C,
    0xFE1DEB,
    0x1CB129,
    0xA73EE8,
    0x8235F5,
    0x2EBB44,
    0x84E99C,
    0x7026B4,
    0x5F7E41,
    0x3991D6,
    0x398353,
    0x39F49C,
    0x845F8B,
    0xBDF928,
    0x3B1FF8,
    0x97FFDE,
    0x05980F,
    0xEF2F11,
    0x8B5A0A,
    0x6D1F6D,
    0x367ECF,
    0x27CB09,
    0xB74F46,
    0x3F669E,
    0x5FEA2D,
    0x7527BA,
    0xC7EBE5,
    0xF17B3D,
    0x0739F7,
    0x8A5292,
    0xEA6BFB,
    0x5FB11F,
    0x8D5D08,
    0x560330,
    0x46FC7B,
    0x6BABF0,
)
_TRIG_PREOP_CHUNK_BITS = 24
_TRIG_PREOP_VALID_BITS = 1201


def gen_vector_trig_preop(
    dst: list[str], src: list[str], is_vop3: bool, has_abs: bool
) -> str:
    """Generate V_TRIG_PREOP_F64's exact 53-bit 2/pi table segment lookup."""
    chunks = ', '.join(f'0x{chunk:06X}u' for chunk in _TRIG_PREOP_TWO_OVER_PI_CHUNKS)
    L = [
        '  // MSB-first 24-bit chunks of the binary expansion of 2/pi used by',
        '  // the ISA range-reduction lookup (the fdlibm/Payne-Hanek layout).',
        f'  static constexpr uint32_t kTwoOverPiChunks[] = {{{chunks}}};',
        f'  static constexpr uint32_t kChunkBits = {_TRIG_PREOP_CHUNK_BITS}u;',
        f'  static constexpr uint32_t kValidBits = {_TRIG_PREOP_VALID_BITS}u;',
        '  auto table_bit = [](uint32_t bit) -> uint64_t {',
        '    if (bit >= kValidBits) return 0;',
        '    uint32_t chunk = bit / kChunkBits;',
        '    if (chunk >= std::size(kTwoOverPiChunks)) return 0;',
        '    return (kTwoOverPiChunks[chunk] >>',
        '            (kChunkBits - 1u - bit % kChunkBits)) & 1u;',
        '  };',
        '  uint64_t exec = wf.exec();',
        '  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {',
        '    if (!(exec & (1ULL << lane))) continue;',
        f'    uint64_t raw0 = amdgpu::RegisterAccess(wf).read_lane64({src[0]}, lane);',
        '    double s0 = std::bit_cast<double>(raw0);',
    ]
    if is_vop3:
        L.extend(vop3_src_mod('s0', 0, has_abs))
    L.extend(
        [
            f'    uint32_t selector = amdgpu::RegisterAccess(wf).read_lane({src[1]}, lane) & 31u;',
            '    uint32_t exponent = static_cast<uint32_t>((std::bit_cast<uint64_t>(s0) >> 52) & 0x7ffu);',
            '    uint32_t shift = selector * 53u;',
            '    if (exponent > 1077u) shift += exponent - 1077u;',
            '    uint64_t segment = 0;',
            '    for (uint32_t bit = 0; bit < 53; ++bit)',
            '      segment = (segment << 1) | table_bit(shift + bit);',
            '    int scale = -53 - static_cast<int>(shift);',
            '    if (exponent >= 1968u) scale += 128;',
            '    uint64_t result = amdgpu::fp_mode::scale_u53_f64_rtz(segment, scale);',
        ]
    )
    if is_vop3:
        L.extend(
            [
                '    uint32_t effective_omod = amdgpu::fp_mode::effective_omod(',
                '        wf.cu().arch(), wf.fp_denorm_mode_f16_f64(), wf.ieee_mode(), inst_.omod);',
                '    result = amdgpu::fp_mode::finish_f64(',
                '        result, wf.fp_round_mode_f16_f64(), effective_omod, inst_.clamp,',
                '        amdgpu::floating_clamp_nan_to_zero(wf));',
            ]
        )
    L.extend(
        [
            f'    amdgpu::RegisterAccess(wf).write_lane64({dst[0]}, lane, result);',
            '  }',
        ]
    )
    return '\n'.join(L)


def gen_vector_mad_64_32(
    dst: list[str],
    src: list[str],
    dtype: str | None,
    result_writer: str | None = None,
    integer_clamp: bool = False,
) -> str:
    """Generate V_MAD_U64_U32 / V_MAD_I64_I32 body.

    D.i64 = S0.i32 * S1.i32 + S2.i64 (signed)
    D.u64 = S0.u32 * S1.u32 + S2.u64 (unsigned)

    Sources S0 and S1 are 32-bit; the accumulator S2 and result D are
    64-bit VGPR pairs. SDST holds unsigned carry-out or signed bit 64 of the
    65-bit sum, independently of integer saturation.
    """
    writes_carry = len(dst) > 1
    L = []
    L.append('  uint64_t exec = wf.exec();')
    if writes_carry:
        L.append('  uint64_t carry = 0;')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')
    if dtype == 'i64':
        L.append(
            f'    int64_t s0 = static_cast<int32_t>(amdgpu::RegisterAccess(wf).read_lane({src[0]}, lane));'
        )
        L.append(
            f'    int64_t s1 = static_cast<int32_t>(amdgpu::RegisterAccess(wf).read_lane({src[1]}, lane));'
        )
        L.append(
            f'    uint64_t s2 = amdgpu::RegisterAccess(wf).read_lane64({src[2]}, lane);'
        )
        L.append('    uint64_t product = static_cast<uint64_t>(s0 * s1);')
        L.append('    uint64_t result = product + s2;')
        if integer_clamp:
            L.append('    bool overflow = amdgpu::signed_add_overflows(product, s2);')
        if writes_carry:
            # SDST is bit 64 of the signed 65-bit sum, independent of saturation.
            L.append(
                '    if (((product >> 63) ^ (s2 >> 63) ^ (result < product)) != 0)'
            )
            L.append('      carry |= 1ULL << lane;')
        if integer_clamp:
            L.append('    if (inst_.clamp && overflow) {')
            L.append(
                '      result = (product & (1ULL << 63)) ? std::bit_cast<uint64_t>(std::numeric_limits<int64_t>::min()) : static_cast<uint64_t>(std::numeric_limits<int64_t>::max());'
            )
            L.append('    }')
        L.append(
            f'    amdgpu::RegisterAccess(wf).write_lane64({dst[0]}, lane, result);'
        )
    else:
        L.append(
            f'    uint64_t s0 = amdgpu::RegisterAccess(wf).read_lane({src[0]}, lane);'
        )
        L.append(
            f'    uint64_t s1 = amdgpu::RegisterAccess(wf).read_lane({src[1]}, lane);'
        )
        L.append(
            f'    uint64_t s2 = amdgpu::RegisterAccess(wf).read_lane64({src[2]}, lane);'
        )
        L.append('    uint64_t product = s0 * s1;')
        L.append('    uint64_t result = product + s2;')
        if writes_carry or integer_clamp:
            L.append('    bool overflow = result < product;')
        if writes_carry:
            L.append('    if (overflow)')
            L.append('      carry |= 1ULL << lane;')
        if integer_clamp:
            L.append(
                '    if (inst_.clamp && overflow) result = std::numeric_limits<uint64_t>::max();'
            )
        L.append(
            f'    amdgpu::RegisterAccess(wf).write_lane64({dst[0]}, lane, result);'
        )
    L.append('  }')
    if writes_carry:
        if result_writer is not None:
            L.append(f'  {result_writer}(carry);')
        else:
            L.append(f'  amdgpu::write_wave_mask_scalar({dst[1]}, wf, carry);')
    return '\n'.join(L)


def gen_vector_div_fixup(
    dst: list[str],
    src: list[str],
    dtype: str | None,
    is_vop3: bool = False,
    has_abs: bool = False,
) -> str:
    """Generate V_DIV_FIXUP body (corrects division result)."""
    if dtype != 'f16':
        return _gen_division_result(dst, src, dtype, 'fixup', is_vop3, has_abs)
    L = []
    opsel = _default_vop3_opsel_expr(dst + src) if is_vop3 and dtype == 'f16' else None
    L.append('  uint64_t exec = wf.exec();')
    if opsel is not None:
        L.append(f'  uint32_t opsel = {opsel};')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')
    if is_vop3:
        L.append(
            f'    float p = util::f16_to_f32(static_cast<uint16_t>({_read_vop3_true16_src(src[0], "opsel", 0)}));'
        )
        L.append(
            f'    float b = util::f16_to_f32(static_cast<uint16_t>({_read_vop3_true16_src(src[1], "opsel", 1)}));'
        )
        L.append(
            f'    float c = util::f16_to_f32(static_cast<uint16_t>({_read_vop3_true16_src(src[2], "opsel", 2)}));'
        )
    else:
        L.append(
            f'    float p = util::f16_to_f32(static_cast<uint16_t>(amdgpu::RegisterAccess(wf).read_lane({src[0]}, lane)));'
        )
        L.append(
            f'    float b = util::f16_to_f32(static_cast<uint16_t>(amdgpu::RegisterAccess(wf).read_lane({src[1]}, lane)));'
        )
        L.append(
            f'    float c = util::f16_to_f32(static_cast<uint16_t>(amdgpu::RegisterAccess(wf).read_lane({src[2]}, lane)));'
        )
    if is_vop3:
        L.extend(vop3_src_mod('p', 0, has_abs))
        L.extend(vop3_src_mod('b', 1, has_abs))
        L.extend(vop3_src_mod('c', 2, has_abs))
    L.append(
        '    float result = amdgpu::div_fixup_f16(p, b, c, wf.fp_round_mode_f16_f64(), wf.fp_denorm_mode_f16_f64());'
    )
    if is_vop3:
        L.extend(vop3_dst_mod('result', omod_result_type='f16'))
        L.append(
            '    uint32_t result_bits = amdgpu::narrow_div_fixup_f16(result, wf.fp16_ovfl());'
        )
        L.append(
            '    result_bits = amdgpu::fp_mode::finalize_omod_f16(result_bits, effective_omod);'
        )
        L.append(
            f'    ::rocjitsu::amdgpu::write_vop3_true16_dst({dst[0]}, wf, lane, opsel, result_bits, true);'
        )
    else:
        L.append(
            f'    amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, amdgpu::narrow_div_fixup_f16(result, wf.fp16_ovfl()));'
        )
    L.append('  }')
    return '\n'.join(L)


def gen_vector_div_scale(
    dst: list[str],
    src: list[str],
    dtype: str | None,
    is_vop3: bool = False,
    has_abs: bool = False,
    result_writer: str | None = None,
) -> str:
    """Generate V_DIV_SCALE through the shared division pre-scaling helper.

    S1 = denominator, S2 = numerator. S0 selects which to scale
    (S0==S1 → scale denominator, S0==S2 → scale numerator).
    VCC is set when V_DIV_FMAS must apply post-scaling.
    """
    is_f64 = dtype == 'f64'
    fp_type = 'double' if is_f64 else 'float'
    read_fn = 'read_lane64' if is_f64 else 'read_lane'
    write_fn = 'write_lane64' if is_f64 else 'write_lane'
    cast_to = 'uint64_t' if is_f64 else 'uint32_t'

    L = []
    L.append('  uint64_t exec = wf.exec();')
    L.append('  uint64_t vcc = wf.vcc();')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')
    L.append(
        f'    {fp_type} s0 = std::bit_cast<{fp_type}>(amdgpu::RegisterAccess(wf).{read_fn}({src[0]}, lane));'
    )
    L.append(
        f'    {fp_type} s1 = std::bit_cast<{fp_type}>(amdgpu::RegisterAccess(wf).{read_fn}({src[1]}, lane));'
    )
    L.append(
        f'    {fp_type} s2 = std::bit_cast<{fp_type}>(amdgpu::RegisterAccess(wf).{read_fn}({src[2]}, lane));'
    )
    if is_vop3:
        L.extend(vop3_src_mod('s0', 0, has_abs))
        L.extend(vop3_src_mod('s1', 1, has_abs))
        L.extend(vop3_src_mod('s2', 2, has_abs))
    mode = 'f16_f64' if is_f64 else 'f32'
    # RDNA3 preserves F64 signaling NaNs; physical gfx1100 and gfx1201 differ.
    quiet_nan = ', wf.cu().arch() != rocjitsu::IsaFamily::Rdna3' if is_f64 else ''
    L.append(
        f'    const amdgpu::DivisionScaleResult<{fp_type}> scaled = amdgpu::div_scale(s0, s1, s2, wf.fp_round_mode_{mode}(), wf.fp_denorm_mode_{mode}(){quiet_nan});'
    )
    L.append(f'    const {fp_type} result = scaled.value;')
    L.append('    const bool set_vcc = scaled.post_scale;')
    L.append('    if (set_vcc) vcc |= (1ULL << lane);')
    L.append('    else vcc &= ~(1ULL << lane);')
    L.append(
        f'    amdgpu::RegisterAccess(wf).{write_fn}({dst[0]}, lane, std::bit_cast<{cast_to}>(result));'
    )
    L.append('  }')
    if len(dst) > 1:
        if result_writer is not None:
            L.append(f'  {result_writer}(vcc);')
        else:
            L.append(f'  amdgpu::write_wave_mask_scalar({dst[1]}, wf, vcc);')
    else:
        L.append('  wf.set_vcc_mask(vcc);')
    return '\n'.join(L)


def gen_vector_div_fmas(
    dst: list[str],
    src: list[str],
    dtype: str | None,
    is_vop3: bool = False,
    has_abs: bool = False,
) -> str:
    """Generate a fused scaled FMA with explicit guest rounding controls."""
    return _gen_division_result(dst, src, dtype, 'fmas', is_vop3, has_abs)


def _gen_division_result(
    dst: list[str],
    src: list[str],
    dtype: str | None,
    operation: str,
    is_vop3: bool,
    has_abs: bool,
) -> str:
    is_f64 = dtype == 'f64'
    fp_type, bits = ('double', 'uint64_t') if is_f64 else ('float', 'uint32_t')
    read = 'read_lane64' if is_f64 else 'read_lane'
    write = 'write_lane64' if is_f64 else 'write_lane'
    mode = 'f16_f64' if is_f64 else 'f32'
    L = ['  uint64_t exec = wf.exec();']
    if operation == 'fmas':
        L.append('  const uint64_t vcc = wf.vcc_mask();')
    if is_vop3:
        L.append(output_policy_decl(dtype or 'f32'))
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')
    for index, operand in enumerate(src):
        L.append(
            f'    {fp_type} s{index} = std::bit_cast<{fp_type}>(amdgpu::RegisterAccess(wf).{read}({operand}, lane));'
        )
        if is_vop3:
            L.extend(vop3_src_mod(f's{index}', index, has_abs))
    post_scale = '(vcc & (1ULL << lane)) != 0, ' if operation == 'fmas' else ''
    L.append(
        f'    {fp_type} result = div_{operation}(s0, s1, s2, {post_scale}wf.fp_round_mode_{mode}(), wf.fp_denorm_mode_{mode}());'
    )
    result = f'std::bit_cast<{bits}>(result)'
    if is_vop3:
        # Both scale the rounded result, then clamp (checked on gfx1201).
        result = apply_output(dtype or 'f32', result)
    L.append(f'    amdgpu::RegisterAccess(wf).{write}({dst[0]}, lane, {result});')
    L.append('  }')
    return '\n'.join(L)


def gen_vector_dot2c_bf16(dst: list[str], src: list[str]) -> str:
    """Generate V_DOT2C_F32_BF16 body: D.f32 += A.bf16[0]*B.bf16[0] + A.bf16[1]*B.bf16[1]."""
    L = []
    d = dst[0] if dst else src[0]
    s0, s1 = (src[0], src[1]) if dst else (src[1], src[2])
    L.append('  uint64_t exec = wf.exec();')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')
    L.append(f'    uint32_t a = amdgpu::RegisterAccess(wf).read_lane({s0}, lane);')
    L.append(f'    uint32_t b = amdgpu::RegisterAccess(wf).read_lane({s1}, lane);')
    L.append(
        f'    float acc = std::bit_cast<float>(amdgpu::RegisterAccess(wf).read_lane({d}, lane));'
    )
    L.append('    float a0 = util::bf16_to_f32(static_cast<uint16_t>(a & 0xFFFF));')
    L.append(
        '    float a1 = util::bf16_to_f32(static_cast<uint16_t>((a >> 16) & 0xFFFF));'
    )
    L.append('    float b0 = util::bf16_to_f32(static_cast<uint16_t>(b & 0xFFFF));')
    L.append(
        '    float b1 = util::bf16_to_f32(static_cast<uint16_t>((b >> 16) & 0xFFFF));'
    )
    L.append('    acc += a0 * b0 + a1 * b1;')
    L.append(
        f'    amdgpu::RegisterAccess(wf).write_lane({d}, lane, std::bit_cast<uint32_t>(acc));'
    )
    L.append('  }')
    return '\n'.join(L)


def gen_vector_bitop3(
    dst: list[str],
    src: list[str],
    dtype: str | None,
    true16_opsel: str | None = None,
) -> str:
    """Generate V_BITOP3_B32/B16 body: 3-input LUT-based bitwise operation.

    The 8-bit truth table is packed into the VOP3 modifier fields:
      truth_table = (omod << 6) | (abs << 3) | neg
    NOT from any source operand value. Source modifiers are not applied.

    Index bit ordering:
      bit 2 = src0, bit 1 = src1, bit 0 = src2
    """
    L = []
    L.append('  uint8_t truth_table = static_cast<uint8_t>')
    L.append('      ((inst_.omod << 6) | (inst_.abs << 3) | inst_.neg);')
    L.append('  uint64_t exec = wf.exec();')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')
    if dtype == 'b16' and true16_opsel:
        helper = '::rocjitsu::amdgpu::read_vop3_true16_src'
        L.append(f'    uint32_t a = {helper}({src[0]}, wf, lane, {true16_opsel}, 0);')
        L.append(f'    uint32_t b = {helper}({src[1]}, wf, lane, {true16_opsel}, 1);')
        L.append(f'    uint32_t c = {helper}({src[2]}, wf, lane, {true16_opsel}, 2);')
    else:
        L.append(
            f'    uint32_t a = amdgpu::RegisterAccess(wf).read_lane({src[0]}, lane);'
        )
        L.append(
            f'    uint32_t b = amdgpu::RegisterAccess(wf).read_lane({src[1]}, lane);'
        )
        L.append(
            f'    uint32_t c = amdgpu::RegisterAccess(wf).read_lane({src[2]}, lane);'
        )
    mask = ' & 0xffffu' if dtype == 'b16' else ''
    L.append(f'    uint32_t result = amdgpu::bitop3_words(a, b, c, truth_table){mask};')
    if dtype == 'b16' and true16_opsel:
        L.append(
            f'    ::rocjitsu::amdgpu::write_vop3_true16_dst({dst[0]}, wf, lane, '
            f'{true16_opsel}, result, true);'
        )
    else:
        L.append(f'    amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, result);')
    L.append('  }')
    return '\n'.join(L)


def gen_vector_permlane_swap(dst: list[str], src: list[str], stride: int) -> str:
    """Generate V_PERMLANE{16,32}_SWAP_B32.

    The swap operates on every 2*stride-lane block of the wavefront, not just the
    first. Within each block starting at lane `base`:
      src0[base + i]          ← old vdst[base + stride + i]   (i in 0..stride-1)
      vdst[base + stride + i] ← old src0[base + i]
    So for the 16-lane form on a wave64 this swaps lanes 0-15<->16-31 AND
    32-47<->48-63 (all four groups); for the 32-lane form it swaps 0-31<->32-63.
    src0[base+stride..] and vdst[base..base+stride-1] within each block are
    UNCHANGED. Each output write is masked by its destination lane's EXEC bit.
    Both vdst and src0 are outputs (LLVM: returns {vdst_new, src0_new}).
    """
    L = []
    L.append('  uint64_t exec = wf.exec();')
    L.append('  uint32_t tmp_dst[64] = {}, tmp_src[64] = {};')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append(
        f'    tmp_dst[lane] = amdgpu::RegisterAccess(wf).read_lane({dst[0]}, lane);'
    )
    L.append(
        f'    tmp_src[lane] = amdgpu::RegisterAccess(wf).read_lane({dst[1]}, lane);'
    )
    L.append('  }')
    L.append(
        f'  for (uint32_t base = 0; base + {stride} < wf.wf_size(); base += 2u * {stride}) {{'
    )
    L.append(f'    for (uint32_t i = 0; i < {stride}; ++i) {{')
    L.append('      if (exec & (1ULL << (base + i)))')
    L.append(
        f'        amdgpu::RegisterAccess(wf).write_lane({dst[1]}, base + i, tmp_dst[base + {stride} + i]);'
    )
    L.append(f'      if (exec & (1ULL << (base + {stride} + i)))')
    L.append(
        f'        amdgpu::RegisterAccess(wf).write_lane({dst[0]}, base + {stride} + i, tmp_src[base + i]);'
    )
    L.append('    }')
    L.append('  }')
    return '\n'.join(L)


def gen_vector_permlane(
    dst: list[str],
    src: list[str],
    op: str | None,
    cross: bool,
    op_sel_expr: str = 'inst_.op_sel',
) -> str:
    """Generate V_PERMLANE16_B32 / V_PERMLANEX16_B32 (imm and var forms).

    For each lane i, read from lane (i & ~0xF) | selector[i & 0xF].
    Immediate form: lanesel = {S2, S1} forms a 64-bit value; each sub-lane i
      selects from lanesel[i*4+3 : i*4]. Sub-lanes 0-7 read from S1 (low 32),
      sub-lanes 8-15 read from S2 (high 32).
    Var form: selector from low 4 bits of src1 VGPR per lane.
    For permlanex16 (cross=True), fetch from the other 16-lane half of the
    same 32-lane row.
    """
    is_var = op == 'var'
    L = []
    L.append(f'  bool fi = ({op_sel_expr} & 0x1u) != 0;')
    L.append(f'  bool bound_ctrl = ({op_sel_expr} & 0x2u) != 0;')
    L.append('  uint64_t exec = wf.exec();')
    L.append('  uint32_t snap[64];')
    L.append('  for (uint32_t i = 0; i < wf.wf_size(); ++i)')
    L.append(f'    snap[i] = amdgpu::RegisterAccess(wf).read_lane({src[0]}, i);')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')
    if is_var:
        L.append(
            f'    uint32_t sel = amdgpu::RegisterAccess(wf).read_lane({src[1]}, lane) & 0xF;'
        )
    else:
        L.append('    uint32_t sub = lane & 0xF;')
        L.append(f'    uint32_t sel_word = (sub < 8u)')
        L.append(f'        ? amdgpu::RegisterAccess(wf).read_scalar({src[1]})')
        L.append(f'        : amdgpu::RegisterAccess(wf).read_scalar({src[2]});')
        L.append('    uint32_t sel = (sel_word >> ((sub & 7u) * 4u)) & 0xF;')
    kind = 'X16' if cross else 'Perm16'
    L.append(
        f'    uint32_t src_lane = amdgpu::valu_permutation_source(amdgpu::ValuPermutation::{kind}, lane, sel, 16, wf.wf_size());'
    )
    L.append('    if (src_lane >= wf.wf_size()) continue;')
    L.append('    bool src_active = (exec & (1ULL << src_lane)) != 0;')
    L.append('    if (!src_active && !fi) {')
    L.append('      if (bound_ctrl)')
    L.append(f'        amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, 0);')
    L.append('      continue;')
    L.append('    }')
    L.append(
        f'    amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, snap[src_lane]);'
    )
    L.append('  }')
    return '\n'.join(L)


def gen_vector_permlane64(dst: list[str], src: list[str]) -> str:
    """Generate V_PERMLANE64_B32: swap lane i with lane i ^ 32."""
    L = []
    L.append('  uint64_t exec = wf.exec();')
    L.append('  uint32_t snap[64];')
    L.append('  for (uint32_t i = 0; i < wf.wf_size(); ++i)')
    L.append(f'    snap[i] = amdgpu::RegisterAccess(wf).read_lane({src[0]}, i);')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')
    L.append(
        '    uint32_t partner = amdgpu::valu_permutation_source(amdgpu::ValuPermutation::Perm64, lane, 0, 0, wf.wf_size());'
    )
    L.append('    if (partner < wf.wf_size())')
    L.append(
        f'      amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, snap[partner]);'
    )
    L.append('  }')
    return '\n'.join(L)


def _permlane_group_width(src: list[str], operand: int) -> list[str]:
    return [
        f'  uint32_t lane_group_width = amdgpu::RegisterAccess(wf).read_scalar({src[operand]});',
        f'  if (exec && inst_.src{operand} < 256 && '
        '(!lane_group_width || (lane_group_width & (lane_group_width - 1))))',
        '    wf.report_undefined_behavior("permlane group width is not a power of two");',
    ]


def gen_vector_permlane_family(dst: list[str], src: list[str], op: str | None) -> str:
    """Generate gfx1250 V_PERMLANE_{BCAST,DOWN,UP,XOR}_B32."""
    if op not in ('bcast', 'down', 'up', 'xor'):
        raise ValueError(f'unsupported permlane family operation: {op}')
    L = []
    L.append('  uint64_t exec = wf.exec();')
    L.append('  uint32_t snap[64];')
    L.append('  for (uint32_t i = 0; i < wf.wf_size(); ++i)')
    L.append(f'    snap[i] = amdgpu::RegisterAccess(wf).read_lane({src[0]}, i);')
    L.append(f'  uint32_t selector = amdgpu::RegisterAccess(wf).read_scalar({src[1]});')
    L.extend(_permlane_group_width(src, 2))
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')
    L.append(
        f'    uint32_t src_lane = amdgpu::valu_permutation_source(amdgpu::ValuPermutation::{op.title()}, lane, selector, lane_group_width, wf.wf_size());'
    )
    L.append(
        f'    amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, snap[src_lane]);'
    )
    L.append('  }')
    return '\n'.join(L)


def gen_vector_permlane_idx_gen(dst: list[str], src: list[str]) -> str:
    """Generate V_PERMLANE_IDX_GEN_B32."""
    L = []
    L.append('  uint64_t exec = wf.exec();')
    L.extend(_permlane_group_width(src, 1))
    L.append('  const uint32_t group_mask = lane_group_width - 1u;')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')
    L.append(
        f'    const uint32_t index = amdgpu::RegisterAccess(wf).read_lane({src[0]}, lane);'
    )
    L.append(
        '    const uint32_t src_lane = (lane & ~group_mask) | (index & 63u & group_mask);'
    )
    L.append(
        f'    amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, src_lane << 2);'
    )
    L.append('  }')
    return '\n'.join(L)


def _gen_ocp_pk_fp8(
    L: list[str],
    dst: list[str],
    src: list[str],
    name: str,
    opsel: str,
    *,
    is_vop3: bool,
    has_abs: bool,
    e5m3_select: str | None = None,
) -> None:
    """Pack two F32 sources as OCP FP8/BF8 into the OPSEL[3] word, keeping the other."""
    L.insert(
        0, f'  const auto conversion = {to_fp8_expr(name, is_vop3, has_abs, "inst_")};'
    )
    L.append(f'    uint32_t s0 = amdgpu::RegisterAccess(wf).read_lane({src[0]}, lane);')
    L.append(f'    uint32_t s1 = amdgpu::RegisterAccess(wf).read_lane({src[1]}, lane);')
    if e5m3_select:
        # CDNA5 alone interprets CLAMP as a format selector on this opcode.
        L.append('    uint32_t packed;')
        L.append(f'    if ({e5m3_select}) {{')
        L.append(
            '      uint32_t lo = util::f32_to_fp8_e5m3_rne_mode(std::bit_cast<float>(s0), wf.fp16_ovfl());'
        )
        L.append(
            '      uint32_t hi = util::f32_to_fp8_e5m3_rne_mode(std::bit_cast<float>(s1), wf.fp16_ovfl());'
        )
        L.append('      packed = lo | (hi << 8);')
        L.append('    } else {')
        L.append('      packed = conversion(s0, s1);')
        L.append('    }')
    else:
        L.append('    uint32_t packed = conversion(s0, s1);')
    L.append(
        f'    ::rocjitsu::amdgpu::write_vop3_true16_dst({dst[0]}, wf, lane, ({opsel}) & 0x8u, packed, true);'
    )


def _gen_ocp_pk_f32_fp8(
    L: list[str], dst: list[str], src: list[str], name: str, opsel: str
) -> None:
    """Decode the two OCP FP8/BF8 bytes of the OPSEL[0] half into two F32 results."""
    L.append(
        f'    uint32_t packed = amdgpu::RegisterAccess(wf).read_lane({src[0]}, lane);'
    )
    half = 'packed' if opsel == '0u' else f'({opsel} & 1) ? (packed >> 16) : packed'
    L.append(f'    uint32_t half = {half};')
    low = decode_fp8_expr(name, 'half')
    high = decode_fp8_expr(name, 'half >> 8')
    L.append(
        f'    amdgpu::RegisterAccess(wf).write_lane64({dst[0]}, lane, '
        f'static_cast<uint64_t>({low}) | (static_cast<uint64_t>({high}) << 32));'
    )


def gen_vector_cvt_pk(
    dst: list[str],
    src: list[str],
    cls: str,
    op: str | None,
    *,
    opsel: str = '0u',
    dtype: str | None = None,
    is_vop3: bool = False,
    has_abs: bool = False,
    fp8_format_select: str | None = None,
    arch_name: str = '',
) -> str:
    """Generate pack/convert instructions."""
    L = []
    L.append('  uint64_t exec = wf.exec();')
    if cls == 'vector_cvt_pknorm' and dtype == 'f16':
        L.append(input_policy.policy_decl('f16'))
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')
    if cls == 'vector_cvt_pk_u8_f32' and len(src) > 2:
        # shared/conversion.h stages, resolved before the lane loop.
        L.insert(
            0, f'  const auto conversion = {pk_u8_expr(is_vop3, has_abs, "inst_")};'
        )
        L.extend(
            f'    uint32_t s{i} = amdgpu::RegisterAccess(wf).read_lane({src[i]}, lane);'
            for i in range(3)
        )
        L.append(
            f'    amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, conversion(s0, s1, s2));'
        )
    elif cls == 'vector_cvt_pk_u8_f32':
        L.append(
            f'    float fval = std::bit_cast<float>(amdgpu::RegisterAccess(wf).read_lane({src[0]}, lane));'
        )
        L.append(
            f'    uint32_t byte_sel = amdgpu::RegisterAccess(wf).read_lane({src[1]}, lane) & 3;'
        )
        # V_CVT_PKACCUM_U8_F32 (CDNA1-4) reads the packed bytes from its
        # destination. It keeps truncation and ignores ABS/NEG and input
        # flushing: no target with it has been captured.
        old_src = dst[0]
        L.append(
            f'    uint32_t old = amdgpu::RegisterAccess(wf).read_lane({old_src}, lane);'
        )
        L.append('    uint32_t byte = amdgpu::cvt_u8_f32(fval);')
        L.append('    uint32_t mask = ~(0xFFu << (byte_sel * 8));')
        L.append(
            f'    amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, (old & mask) | (byte << (byte_sel * 8)));'
        )
    elif cls == 'vector_cvt_pknorm':
        if dtype == 'f16':
            if is_vop3:
                reads = [_read_vop3_true16_src(src[i], opsel, i) for i in range(2)]
            else:
                reads = [
                    f'amdgpu::RegisterAccess(wf).read_lane({src[i]}, lane)'
                    for i in range(2)
                ]
            abs_field = 'inst_.abs' if has_abs else '0u'
            for i, read in enumerate(reads):
                L.append(
                    f'    uint32_t s{i}_bits = amdgpu::input_denormal::prepare<amdgpu::fp_format::F16>('
                    f'amdgpu::source_modifier::apply<amdgpu::fp_format::F16>('
                    f'static_cast<uint32_t>({read}), {i}, {abs_field}, inst_.neg), '
                    f'{input_policy.NAME});'
                )
                L.append(
                    f'    float s{i} = util::f16_to_f32(static_cast<uint16_t>(s{i}_bits));'
                )
        else:
            L.append(
                f'    float s0 = std::bit_cast<float>(amdgpu::RegisterAccess(wf).read_lane({src[0]}, lane));'
            )
            L.append(
                f'    float s1 = std::bit_cast<float>(amdgpu::RegisterAccess(wf).read_lane({src[1]}, lane));'
            )
        if is_vop3 and dtype != 'f16':
            L.extend(vop3_src_mod('s0', 0, has_abs))
            L.extend(vop3_src_mod('s1', 1, has_abs))
        # Round to nearest even under every MODE.FP_ROUND setting. The ISA applies
        # FP_ROUND to VALU float operations, and the packed NORM entries
        # (V_CVT_PK_NORM_* and V_CVT_PKNORM_*) give no exception. gfx1100 and
        # gfx1201 captures of V_CVT_PK_NORM_{I,U}16_{F16,F32} nevertheless do not
        # change with FP_ROUND. CDNA1-5, RDNA1-2 and RDNA3.5 reuse this rounding
        # without hardware verification.
        if op == 'i16':
            L.append('    auto cvt_i16 = [](float f) -> int16_t {')
            L.append('      if (std::isnan(f)) return 0;')
            L.append(
                '      return static_cast<int16_t>(util::rndne_scalar(std::clamp(static_cast<double>(f) * 32767.0, -32767.0, 32767.0)));'
            )
            L.append('    };')
            L.append('    int16_t lo = cvt_i16(s0);')
            L.append('    int16_t hi = cvt_i16(s1);')
        else:  # u16
            L.append('    auto cvt_u16 = [](float f) -> uint16_t {')
            L.append('      if (std::isnan(f)) return 0;')
            L.append(
                '      return static_cast<uint16_t>(util::rndne_scalar(std::clamp(static_cast<double>(f) * 65535.0, 0.0, 65535.0)));'
            )
            L.append('    };')
            L.append('    uint16_t lo = cvt_u16(s0);')
            L.append('    uint16_t hi = cvt_u16(s1);')
        L.append(
            f'    amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, (static_cast<uint32_t>(hi) << 16) | (static_cast<uint32_t>(lo) & 0xFFFF));'
        )
    elif cls == 'vector_cvt_pkrtz_f16_f32':
        # shared/conversion.h stages, resolved before the lane loop.
        L.insert(
            0,
            f'  const auto conversion = {pk_rtz_f16_expr(is_vop3, has_abs, "inst_")};',
        )
        L.append(
            f'    uint32_t s0 = amdgpu::RegisterAccess(wf).read_lane({src[0]}, lane);'
        )
        L.append(
            f'    uint32_t s1 = amdgpu::RegisterAccess(wf).read_lane({src[1]}, lane);'
        )
        L.append(
            f'    amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, conversion(s0, s1));'
        )
    elif cls == 'vector_cvt_pk':
        if op in ('fp8_f32', 'bf8_f32') and uses_shared_fp8_narrowing(arch_name):
            _gen_ocp_pk_fp8(
                L,
                dst,
                src,
                op[:3],
                opsel,
                is_vop3=is_vop3,
                has_abs=has_abs,
                e5m3_select=fp8_format_select if op == 'fp8_f32' else None,
            )
        elif op in ('f32_fp8', 'f32_bf8') and uses_shared_fp8_widening(arch_name):
            _gen_ocp_pk_f32_fp8(L, dst, src, op[4:], opsel)
        elif op in ('fp8_f32', 'bf8_f32', 'fp8_f16', 'bf8_f16'):
            # RDNA4 data-conversion prose scopes FP16_OVFL for FP8/BF8
            # destinations to conversions from F32 sources. Plain F16/BF16
            # sources convert the already-rounded 16-bit input value.
            use_fp16_ovfl = op.endswith('_f32')
            base_conv = (
                'util::f32_to_fp8_e4m3_rne'
                if op.startswith('fp8_')
                else 'util::f32_to_bf8_e5m2_rne'
            )
            conv = (
                _fp8_mode_rne_helper_name(arch_name, base_conv)
                if use_fp16_ovfl
                else fp8_helper_name(arch_name, base_conv)
            )
            conv_e5m3 = (
                (
                    'util::f32_to_fp8_e5m3_rne_mode'
                    if use_fp16_ovfl
                    else 'util::f32_to_fp8_e5m3_rne'
                )
                if op.startswith('fp8_') and fp8_format_select is not None
                else None
            )
            if op.endswith('_f32'):
                L.append(
                    f'    float s0 = std::bit_cast<float>(amdgpu::RegisterAccess(wf).read_lane({src[0]}, lane));'
                )
                L.append(
                    f'    float s1 = std::bit_cast<float>(amdgpu::RegisterAccess(wf).read_lane({src[1]}, lane));'
                )
            else:
                L.append(
                    f'    uint32_t raw = amdgpu::RegisterAccess(wf).read_lane({src[0]}, lane);'
                )
                L.append('    float s0 = util::f16_to_f32(static_cast<uint16_t>(raw));')
                L.append(
                    '    float s1 = util::f16_to_f32(static_cast<uint16_t>(raw >> 16));'
                )
            if is_vop3 and op.endswith('_f32'):
                L.extend(vop3_src_mod('s0', 0, has_abs))
                L.extend(vop3_src_mod('s1', 1, has_abs))
            elif is_vop3 and arch_name == 'cdna5' and op.endswith('_f16'):
                # Both packed F16 channels belong to SRC0 (CDNA5 opcodes 882/883).
                L.extend(vop3_src_mod('s0', 0, has_abs))
                L.extend(vop3_src_mod('s1', 0, has_abs))
            if conv_e5m3 is not None:
                L.append(
                    f"    uint32_t lo = ({fp8_format_select}) ? {_fp8_rne_encode_call(conv_e5m3, 's0', use_fp16_ovfl)} : {_fp8_rne_encode_call(conv, 's0', use_fp16_ovfl)};"
                )
                L.append(
                    f"    uint32_t hi = ({fp8_format_select}) ? {_fp8_rne_encode_call(conv_e5m3, 's1', use_fp16_ovfl)} : {_fp8_rne_encode_call(conv, 's1', use_fp16_ovfl)};"
                )
            else:
                L.append(
                    f"    uint32_t lo = {_fp8_rne_encode_call(conv, 's0', use_fp16_ovfl)};"
                )
                L.append(
                    f"    uint32_t hi = {_fp8_rne_encode_call(conv, 's1', use_fp16_ovfl)};"
                )
            L.append(
                '    uint32_t packed = static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 8);'
            )
            L.append(
                f'    ::rocjitsu::amdgpu::write_vop3_true16_dst({dst[0]}, wf, lane, ({opsel}) & 0x8u, packed);'
            )
        elif op in ('f32_fp8', 'f32_bf8', 'f16_fp8', 'f16_bf8'):
            conv = fp8_helper_name(
                arch_name,
                (
                    'util::fp8_e4m3_to_f32'
                    if op.endswith('_fp8')
                    else 'util::bf8_e5m2_to_f32'
                ),
            )
            L.append(
                f'    uint32_t packed = amdgpu::RegisterAccess(wf).read_lane({src[0]}, lane);'
            )
            L.append(f'    bool src_hi = {opsel} & 1;')
            L.append(
                '    uint32_t half = src_hi ? (packed >> 16) : (packed & 0xFFFFu);'
            )
            L.append(f'    float lo = {conv}(static_cast<uint8_t>(half & 0xFFu));')
            L.append(
                f'    float hi = {conv}(static_cast<uint8_t>((half >> 8) & 0xFFu));'
            )
            if op.startswith('f32_'):
                L.append('    uint32_t lo_bits = std::bit_cast<uint32_t>(lo);')
                L.append('    uint32_t hi_bits = std::bit_cast<uint32_t>(hi);')
                L.append(
                    f'    amdgpu::RegisterAccess(wf).write_lane64({dst[0]}, lane, static_cast<uint64_t>(lo_bits) | (static_cast<uint64_t>(hi_bits) << 32));'
                )
            else:
                L.append(
                    '    uint32_t lo_bits = util::f32_to_f16_mode(lo, wf.fp16_ovfl());'
                )
                L.append(
                    '    uint32_t hi_bits = util::f32_to_f16_mode(hi, wf.fp16_ovfl());'
                )
                L.append(
                    f'    amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, lo_bits | (hi_bits << 16));'
                )
        elif op in ('u16_f32', 'i16_f32'):
            L.append(
                f'    float f0 = std::bit_cast<float>(amdgpu::RegisterAccess(wf).read_lane({src[0]}, lane));'
            )
            L.append(
                f'    float f1 = std::bit_cast<float>(amdgpu::RegisterAccess(wf).read_lane({src[1]}, lane));'
            )
            if is_vop3:
                L.extend(vop3_src_mod('f0', 0, has_abs))
                L.extend(vop3_src_mod('f1', 1, has_abs))
            if op == 'u16_f32':
                L.append('    uint32_t lo = amdgpu::cvt_u16_f32(f0);')
                L.append('    uint32_t hi = amdgpu::cvt_u16_f32(f1);')
            else:
                L.append('    uint32_t lo = amdgpu::cvt_i16_f32(f0);')
                L.append('    uint32_t hi = amdgpu::cvt_i16_f32(f1);')
            L.append(
                f'    amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, (hi << 16) | (lo & 0xFFFFu));'
            )
        else:
            L.append(
                f'    uint32_t s0 = amdgpu::RegisterAccess(wf).read_lane({src[0]}, lane);'
            )
            L.append(
                f'    uint32_t s1 = amdgpu::RegisterAccess(wf).read_lane({src[1]}, lane);'
            )
            if op == 'u16_u32':
                L.append(
                    '    uint16_t lo = static_cast<uint16_t>(std::min(s0, 0xFFFFu));'
                )
                L.append(
                    '    uint16_t hi = static_cast<uint16_t>(std::min(s1, 0xFFFFu));'
                )
            else:  # i16_i32
                L.append(
                    '    int16_t lo = static_cast<int16_t>(std::clamp(static_cast<int32_t>(s0), -32768, 32767));'
                )
                L.append(
                    '    int16_t hi = static_cast<int16_t>(std::clamp(static_cast<int32_t>(s1), -32768, 32767));'
                )
            L.append(
                f'    amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, (static_cast<uint32_t>(static_cast<uint16_t>(hi)) << 16) | static_cast<uint32_t>(static_cast<uint16_t>(lo)));'
            )
    elif cls == 'vector_cvt_pk_f16_f32':
        L.append(
            f'    float s0 = std::bit_cast<float>(amdgpu::RegisterAccess(wf).read_lane({src[0]}, lane));'
        )
        L.append(
            f'    float s1 = std::bit_cast<float>(amdgpu::RegisterAccess(wf).read_lane({src[1]}, lane));'
        )
        if is_vop3 and arch_name == 'cdna4':
            L.extend(vop3_src_mod('s0', 0, has_abs))
            L.extend(vop3_src_mod('s1', 1, has_abs))
        L.append(f'    uint32_t lo = util::f32_to_f16_mode(s0, wf.fp16_ovfl());')
        L.append(f'    uint32_t hi = util::f32_to_f16_mode(s1, wf.fp16_ovfl());')
        packed = (
            'amdgpu::fp_mode::pack_f16_results(lo, hi, wf.fp_denorm_mode_f16_f64())'
            if arch_name == 'cdna5'
            else 'lo | (hi << 16)'
        )
        L.append(
            f'    amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, {packed});'
        )
    elif cls == 'vector_cvt_pk_bf16_f32':
        L.append(
            f'    float s0 = std::bit_cast<float>(amdgpu::RegisterAccess(wf).read_lane({src[0]}, lane));'
        )
        L.append(
            f'    float s1 = std::bit_cast<float>(amdgpu::RegisterAccess(wf).read_lane({src[1]}, lane));'
        )
        if is_vop3 and arch_name == 'cdna4':
            L.extend(vop3_src_mod('s0', 0, has_abs))
            L.extend(vop3_src_mod('s1', 1, has_abs))
        L.append(f'    uint32_t lo = util::f32_to_bf16_rne_mode(s0, wf.fp16_ovfl());')
        L.append(f'    uint32_t hi = util::f32_to_bf16_rne_mode(s1, wf.fp16_ovfl());')
        L.append(
            f'    amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, lo | (hi << 16));'
        )
    elif cls in ('vector_cvt_sr_pk_f16_f32', 'vector_cvt_sr_pk_bf16_f32'):
        helper = f'amdgpu::fp_mode::{cls.removeprefix("vector_")}'
        L.append(
            f'    uint32_t s0 = amdgpu::RegisterAccess(wf).read_lane({src[0]}, lane);'
        )
        L.append(
            f'    uint32_t s1 = amdgpu::RegisterAccess(wf).read_lane({src[1]}, lane);'
        )
        L.append(
            f'    uint32_t seed = amdgpu::RegisterAccess(wf).read_lane({src[2]}, lane);'
        )
        L.append(
            f'    uint32_t lo = {helper}(s0, static_cast<uint16_t>(seed), wf.fp16_ovfl());'
        )
        L.append(
            f'    uint32_t hi = {helper}(s1, static_cast<uint16_t>(seed >> 16), wf.fp16_ovfl());'
        )
        L.append(
            f'    amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, lo | (hi << 16));'
        )
    elif cls == 'vector_pack_b32_f16':
        # shared/conversion.h stages, resolved before the lane loop.
        L.insert(
            0,
            f'  const auto conversion = {pack_b32_f16_expr(is_vop3, has_abs, "inst_")};',
        )
        if is_vop3:
            L.append(f'    uint32_t s0 = {_read_vop3_true16_src(src[0], opsel, 0)};')
            L.append(f'    uint32_t s1 = {_read_vop3_true16_src(src[1], opsel, 1)};')
        else:
            L.append(
                f'    uint32_t s0 = amdgpu::RegisterAccess(wf).read_lane({src[0]}, lane);'
            )
            L.append(
                f'    uint32_t s1 = amdgpu::RegisterAccess(wf).read_lane({src[1]}, lane);'
            )
        L.append(
            f'    amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, conversion(s0, s1));'
        )
    elif cls == 'vector_cvt_sr_f16_f32':
        # Stochastic rounding: use src1 as random bits for rounding
        L.append(
            f'    float s0 = std::bit_cast<float>(amdgpu::RegisterAccess(wf).read_lane({src[0]}, lane));'
        )
        if is_vop3:
            L.extend(vop3_src_mod('s0', 0, has_abs))
        L.append(
            f'    uint32_t seed = amdgpu::RegisterAccess(wf).read_lane({src[1]}, lane);'
        )
        L.append(
            '    uint32_t result = static_cast<uint32_t>(util::f32_to_f16_sr_mode(s0, seed, wf.fp16_ovfl()));'
        )
        L.append(
            f'    ::rocjitsu::amdgpu::write_vop3_true16_dst({dst[0]}, wf, lane, {opsel}, result);'
        )
    elif cls == 'vector_cvt_sr_bf16_f32':
        L.append(
            f'    float s0 = std::bit_cast<float>(amdgpu::RegisterAccess(wf).read_lane({src[0]}, lane));'
        )
        if is_vop3:
            L.extend(vop3_src_mod('s0', 0, has_abs))
        L.append(
            f'    uint32_t seed = amdgpu::RegisterAccess(wf).read_lane({src[1]}, lane);'
        )
        L.append(
            '    uint32_t result = static_cast<uint32_t>(util::f32_to_bf16_sr_mode(s0, seed, wf.fp16_ovfl()));'
        )
        L.append(
            f'    ::rocjitsu::amdgpu::write_vop3_true16_dst({dst[0]}, wf, lane, {opsel}, result);'
        )
    elif cls == 'vector_cvt_sr_fp8_f16':
        # RDNA4 data-conversion prose scopes FP16_OVFL for FP8 destinations to
        # conversions from F32 sources; the F16-source SR form stays raw.
        L.append(
            f'    uint32_t raw = ::rocjitsu::amdgpu::read_vop3_true16_src({src[0]}, wf, lane, {opsel}, 0);'
        )
        L.append('    float s0 = util::f16_to_f32(static_cast<uint16_t>(raw));')
        L.append(
            f'    uint32_t seed = amdgpu::RegisterAccess(wf).read_lane({src[1]}, lane);'
        )
        if fp8_format_select is not None:
            cvt_fn = fp8_helper_name(arch_name, 'util::f32_to_fp8_e4m3_sr')
            L.append(
                f'    uint8_t result = ({fp8_format_select}) ? util::f32_to_fp8_e5m3_sr(s0, seed) : {cvt_fn}(s0, seed);'
            )
        else:
            cvt_fn = fp8_helper_name(arch_name, 'util::f32_to_fp8_e4m3_sr')
            L.append(f'    uint8_t result = {cvt_fn}(s0, seed);')
        L.append(f'    uint32_t dst_byte = ({opsel} >> 2) & 0x3;')
        L.append(
            f'    uint32_t old = amdgpu::RegisterAccess(wf).read_lane({dst[0]}, lane);'
        )
        L.append('    uint32_t mask = ~(0xFFu << (dst_byte * 8));')
        L.append(
            f'    amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, (old & mask) | (static_cast<uint32_t>(result) << (dst_byte * 8)));'
        )
    elif cls == 'vector_cvt_sr_bf8_f16':
        # RDNA4 data-conversion prose scopes FP16_OVFL for BF8 destinations to
        # conversions from F32 sources; the F16-source SR form stays raw.
        L.append(
            f'    uint32_t raw = ::rocjitsu::amdgpu::read_vop3_true16_src({src[0]}, wf, lane, {opsel}, 0);'
        )
        L.append('    float s0 = util::f16_to_f32(static_cast<uint16_t>(raw));')
        L.append(
            f'    uint32_t seed = amdgpu::RegisterAccess(wf).read_lane({src[1]}, lane);'
        )
        cvt_fn = fp8_helper_name(arch_name, 'util::f32_to_bf8_e5m2_sr')
        L.append(f'    uint8_t result = {cvt_fn}(s0, seed);')
        L.append(f'    uint32_t dst_byte = ({opsel} >> 2) & 0x3;')
        L.append(
            f'    uint32_t old = amdgpu::RegisterAccess(wf).read_lane({dst[0]}, lane);'
        )
        L.append('    uint32_t mask = ~(0xFFu << (dst_byte * 8));')
        L.append(
            f'    amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, (old & mask) | (static_cast<uint32_t>(result) << (dst_byte * 8)));'
        )
    L.append('  }')
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


def _opsel_field(ctx) -> str:
    """Return the correct opsel field name for the current ISA encoding.

    VOP3 encodings always have an opsel field in the machine instruction struct,
    but the ISA spec may not list it as a ucode field for every instruction.
    When encoding fields don't include it, fall back to the struct field name
    based on ISA convention (op_sel for CDNA/RDNA3, opsel for RDNA4/gfx1250).
    """
    if 'op_sel' in ctx.enc_field_names:
        return 'inst_.op_sel'
    if 'opsel' in ctx.enc_field_names:
        return 'inst_.opsel'
    if ctx.is_vop3:
        enc_map = getattr(ctx, 'encoding_map', None)
        if enc_map and ctx.enc_name in enc_map:
            fields = {f.name for f in enc_map[ctx.enc_name].ucode_fields}
            if 'opsel' in fields:
                return 'inst_.opsel'
            if 'op_sel' in fields:
                return 'inst_.op_sel'
        return 'inst_.op_sel'
    return '0u'


def gen_cvt_fp8(ctx) -> str:
    """Generate execute body for non-scaled V_CVT_{PK,SR}_{FP8,BF8}_F32
    and V_CVT_PK_F32_{FP8,BF8} instructions."""
    from amdisa.codegen.execute import ExecuteContext

    op = ctx.op
    dst = ctx.dst_ops
    src = ctx.src_ops
    is_vop3 = ctx.is_vop3
    opsel = _opsel_field(ctx) if is_vop3 else '0u'
    fp8_format_select = 'inst_.clamp' if is_vop3 and ctx.arch_name == 'cdna5' else None

    L = []
    L.append('  uint64_t exec = wf.exec();')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')

    shared_narrow = uses_shared_fp8_narrowing(ctx.arch_name)
    if shared_narrow and op in ('pk_fp8_f32', 'pk_bf8_f32', 'sr_fp8_f32', 'sr_bf8_f32'):
        gen = _gen_ocp_pk_fp8 if op.startswith('pk_') else _gen_ocp_sr_fp8
        gen(
            L,
            dst,
            src,
            op[3:6],
            opsel,
            is_vop3=is_vop3,
            has_abs=ctx.has_abs,
            e5m3_select=fp8_format_select if op[3:6] == 'fp8' else None,
        )
    elif uses_shared_fp8_widening(ctx.arch_name) and op in ('pk_f32_fp8', 'pk_f32_bf8'):
        _gen_ocp_pk_f32_fp8(L, dst, src, op[-3:], opsel)
    elif op == 'pk_fp8_f32':
        _gen_pk_narrow_fp8(
            L,
            dst,
            src,
            _fp8_mode_rne_helper_name(ctx.arch_name, 'util::f32_to_fp8_e4m3_rne'),
            opsel,
            apply_src_mod=is_vop3,
        )
    elif op == 'pk_bf8_f32':
        _gen_pk_narrow_fp8(
            L,
            dst,
            src,
            _fp8_mode_rne_helper_name(ctx.arch_name, 'util::f32_to_bf8_e5m2_rne'),
            opsel,
            apply_src_mod=is_vop3,
        )
    elif op == 'sr_fp8_f32':
        _gen_sr_narrow_fp8(
            L,
            dst,
            src,
            _fp8_mode_sr_helper_name(ctx.arch_name, 'util::f32_to_fp8_e4m3_sr'),
            opsel,
            apply_src_mod=is_vop3,
        )
    elif op == 'sr_bf8_f32':
        _gen_sr_narrow_fp8(
            L,
            dst,
            src,
            _fp8_mode_sr_helper_name(ctx.arch_name, 'util::f32_to_bf8_e5m2_sr'),
            opsel,
            apply_src_mod=is_vop3,
        )
    elif op == 'pk_f32_fp8':
        _gen_pk_widen_fp8(
            L,
            dst,
            src,
            fp8_helper_name(ctx.arch_name, 'util::fp8_e4m3_to_f32'),
            opsel,
        )
    elif op == 'pk_f32_bf8':
        _gen_pk_widen_fp8(
            L,
            dst,
            src,
            fp8_helper_name(ctx.arch_name, 'util::bf8_e5m2_to_f32'),
            opsel,
        )

    L.append('  }')
    return '\n'.join(L)


def _gen_pk_narrow_fp8(
    L: list[str],
    dst: list[str],
    src: list[str],
    cvt_fn: str,
    opsel: str = '0u',
    *,
    apply_src_mod: bool,
) -> None:
    """Pack 2×F32 → 2×FP8/BF8 with OPSEL[3] half selection, R-M-W."""
    L.append(
        f'    float s0 = std::bit_cast<float>(static_cast<uint32_t>(amdgpu::RegisterAccess(wf).read_lane({src[0]}, lane)));'
    )
    L.append(
        f'    float s1 = std::bit_cast<float>(static_cast<uint32_t>(amdgpu::RegisterAccess(wf).read_lane({src[1]}, lane)));'
    )
    if apply_src_mod:
        L.extend(vop3_src_mod('s0', 0, True))
        L.extend(vop3_src_mod('s1', 1, True))
    L.append(f'    uint8_t r0 = {cvt_fn}(s0, wf.fp16_ovfl());')
    L.append(f'    uint8_t r1 = {cvt_fn}(s1, wf.fp16_ovfl());')
    L.append(
        '    uint32_t packed = static_cast<uint32_t>(r0) | (static_cast<uint32_t>(r1) << 8);'
    )
    L.append(
        f'    ::rocjitsu::amdgpu::write_vop3_true16_dst({dst[0]}, wf, lane, ({opsel}) & 0x8u, packed);'
    )


def _gen_sr_narrow_fp8(
    L: list[str],
    dst: list[str],
    src: list[str],
    cvt_fn: str,
    opsel: str = '0u',
    *,
    apply_src_mod: bool,
) -> None:
    """Single F32 → FP8/BF8 with stochastic rounding, OPSEL[3:2] byte sel, R-M-W."""
    L.append(
        f'    float s0 = std::bit_cast<float>(static_cast<uint32_t>(amdgpu::RegisterAccess(wf).read_lane({src[0]}, lane)));'
    )
    L.append(
        f'    uint32_t seed = amdgpu::RegisterAccess(wf).read_lane({src[1]}, lane);'
    )
    if apply_src_mod:
        L.extend(vop3_src_mod('s0', 0, True))
    L.append(f'    uint8_t result = {cvt_fn}(s0, seed, wf.fp16_ovfl());')
    L.append(f'    uint32_t dst_byte = ({opsel} >> 2) & 0x3;')
    L.append(
        f'    uint32_t old = amdgpu::RegisterAccess(wf).read_lane({dst[0]}, lane);'
    )
    L.append('    uint32_t mask = ~(0xFFu << (dst_byte * 8));')
    L.append(
        f'    amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, (old & mask) | (static_cast<uint32_t>(result) << (dst_byte * 8)));'
    )


def _gen_ocp_sr_fp8(
    L: list[str],
    dst: list[str],
    src: list[str],
    name: str,
    opsel: str,
    *,
    is_vop3: bool,
    has_abs: bool,
    e5m3_select: str | None = None,
) -> None:
    """One F32 source to OCP FP8/BF8 with stochastic rounding, into the OPSEL[3:2] byte."""
    L.insert(
        0, f'  const auto conversion = {to_fp8_expr(name, is_vop3, has_abs, "inst_")};'
    )
    L.append(f'    uint32_t s0 = amdgpu::RegisterAccess(wf).read_lane({src[0]}, lane);')
    L.append(
        f'    uint32_t seed = amdgpu::RegisterAccess(wf).read_lane({src[1]}, lane);'
    )
    if e5m3_select:
        L.append(
            f'    uint32_t result = ({e5m3_select}) ? util::f32_to_fp8_e5m3_sr_mode(std::bit_cast<float>(s0), seed, wf.fp16_ovfl()) : conversion.stochastic(s0, seed);'
        )
    else:
        L.append('    uint32_t result = conversion.stochastic(s0, seed);')
    L.append(f'    uint32_t shift = (({opsel} >> 2) & 0x3u) * 8u;')
    L.append(
        f'    uint32_t old = amdgpu::RegisterAccess(wf).read_lane({dst[0]}, lane);'
    )
    L.append(
        f'    amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, (old & ~(0xFFu << shift)) | (result << shift));'
    )


def _gen_pk_widen_fp8(
    L: list[str], dst: list[str], src: list[str], cvt_fn: str, opsel: str = '0u'
) -> None:
    """Unpack 2×FP8/BF8 → 2×F32 with OPSEL[0] half selection."""
    L.append(
        f'    uint32_t packed = amdgpu::RegisterAccess(wf).read_lane({src[0]}, lane);'
    )
    L.append(f'    bool hi = {opsel} & 1;')
    L.append('    uint32_t half = hi ? (packed >> 16) : (packed & 0xFFFFu);')
    L.append(f'    float f0 = {cvt_fn}(static_cast<uint8_t>(half & 0xFF));')
    L.append(f'    float f1 = {cvt_fn}(static_cast<uint8_t>((half >> 8) & 0xFF));')
    L.append('    uint64_t result = static_cast<uint64_t>(std::bit_cast<uint32_t>(f0))')
    L.append('        | (static_cast<uint64_t>(std::bit_cast<uint32_t>(f1)) << 32);')
    L.append(f'    amdgpu::RegisterAccess(wf).write_lane64({dst[0]}, lane, result);')
