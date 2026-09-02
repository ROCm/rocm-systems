# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Vector special operation execute body generators.

Free functions that emit C++ execute_impl bodies for specialized vector
instructions, including relative moves, packed permutations, QSAD/MQSAD,
saturating packs, arithmetic, lane permutations, and packed type conversion.
"""

from __future__ import annotations

from amdisa.codegen.execute.fp8_formats import fp8_helper_name
from amdisa.codegen.execute.vop3_modifiers import (
    vop3_src_mod,
    vop3_dst_mod,
    vop3_dst_mod_f64,
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


def _narrow_rne_encode_call(fmt: str, cvt_fn: str, value_expr: str) -> str:
    if fmt in ('fp8', 'bf8'):
        return f'{cvt_fn}({value_expr}, wf.fp16_ovfl())'
    return f'{cvt_fn}({value_expr})'


def _narrow_sr_encode_call(
    fmt: str, cvt_fn: str, value_expr: str, seed_expr: str
) -> str:
    if fmt in ('fp8', 'bf8'):
        return f'{cvt_fn}({value_expr}, {seed_expr}, wf.fp16_ovfl())'
    return f'{cvt_fn}({value_expr}, {seed_expr})'


def _fp8_rne_encode_call(cvt_fn: str, value_expr: str, use_fp16_ovfl: bool) -> str:
    if use_fp16_ovfl:
        return f'{cvt_fn}({value_expr}, wf.fp16_ovfl())'
    return f'{cvt_fn}({value_expr})'


def _fp8_sr_encode_call(
    cvt_fn: str, value_expr: str, seed_expr: str, use_fp16_ovfl: bool
) -> str:
    if use_fp16_ovfl:
        return f'{cvt_fn}({value_expr}, {seed_expr}, wf.fp16_ovfl())'
    return f'{cvt_fn}({value_expr}, {seed_expr})'


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


def gen_vector_bcnt(dst: list[str], src: list[str]) -> str:
    """Generate V_BCNT_U32_B32 body: D.u32 = CountOneBits(S0.u32) + S1.u32."""
    L = []
    L.append('  uint64_t exec = wf.exec();')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')
    L.append(
        f'    uint32_t bits = amdgpu::RegisterAccess(wf).read_lane({src[0]}, lane);'
    )
    L.append(
        f'    uint32_t accum = amdgpu::RegisterAccess(wf).read_lane({src[1]}, lane);'
    )
    L.append(
        f'    amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane,'
        ' static_cast<uint32_t>(std::popcount(bits)) + accum);'
    )
    L.append('  }')
    return '\n'.join(L)


def gen_vector_mbcnt(dst: list[str], src: list[str], op: str | None) -> str:
    """Generate V_MBCNT_LO/HI_U32_B32 body."""
    L = []
    L.append('  uint64_t exec = wf.exec();')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')
    L.append(
        f'    uint32_t mask = amdgpu::RegisterAccess(wf).read_lane({src[0]}, lane);'
    )
    L.append(
        f'    uint32_t base = amdgpu::RegisterAccess(wf).read_lane({src[1]}, lane);'
    )
    if op == 'lo':
        L.append(
            '    uint32_t thread_mask = lane < 32 ? (1u << lane) - 1 : 0xFFFFFFFFu;'
        )
        L.append('    uint32_t count = std::popcount(mask & thread_mask);')
    else:  # hi
        L.append('    uint32_t shift = lane >= 32 ? lane - 32 : 0;')
        L.append('    uint32_t thread_mask = lane >= 32 ? (1u << shift) - 1 : 0;')
        L.append('    uint32_t count = std::popcount(mask & thread_mask);')
    L.append(
        f'    amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, base + count);'
    )
    L.append('  }')
    return '\n'.join(L)


def _resolved_vgpr_offset_call(opnd: str, uses_vgpr_msb_indexing: bool) -> str:
    if uses_vgpr_msb_indexing:
        return (
            f'Isa::resolved_vgpr_offset(wf, {opnd}.opr_type_, '
            f'{opnd}.encoding_value_, {opnd}.vgpr_msb_role())'
        )
    return f'Isa::resolved_vgpr_offset({opnd}.opr_type_, {opnd}.encoding_value_)'


def gen_vector_movrel(
    dst: list[str],
    src: list[str],
    op: str | None,
    uses_vgpr_msb_indexing: bool = False,
    supports_dpp: bool = False,
    supports_dpp8: bool = False,
    supports_sdwa: bool = False,
    inactive_uses_bound_ctrl: bool = False,
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
            f'  bool rel_src_valid = rel_src_base && wf.m0() <= 1023u && '
            f'rel_src_index + {src[0]}.vgpr_count() <= wf.vgpr_alloc().count;'
        )
        L.append(
            f'  Operand rel_src({src[0]}.size_bits(), OperandType::OPR_VGPR, '
            f'static_cast<int>(rel_src_valid ? rel_src_index : 0u));'
        )
        _stage_movrel_source(
            L, supports_dpp, supports_dpp8, supports_sdwa, inactive_uses_bound_ctrl
        )
        L.append('  uint64_t exec = wf.exec();')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')
        L.append(
            f'    amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, amdgpu::RegisterAccess(wf).read_lane(rel_src, lane));'
        )
        L.append('  }')
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
            f'  bool rel_src_valid = rel_src_base && rel_src_offset <= 1023u && '
            f'rel_src_index + {src[0]}.vgpr_count() <= wf.vgpr_alloc().count;'
        )
        L.append(
            f'  bool rel_dst_valid = rel_dst_base && rel_dst_offset <= 1023u && '
            f'rel_dst_index + {dst[0]}.vgpr_count() <= wf.vgpr_alloc().count;'
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
            L, supports_dpp, supports_dpp8, supports_sdwa, inactive_uses_bound_ctrl
        )
        L.append('  uint64_t exec = wf.exec();')
        L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        L.append('    if (!(exec & (1ULL << lane))) continue;')
        L.append(
            '    amdgpu::RegisterAccess(wf).write_lane(rel_dst, lane, '
            'amdgpu::RegisterAccess(wf).read_lane(rel_src, lane));'
        )
        L.append('  }')
        return '\n'.join(L)

    rel_dst_call = _resolved_vgpr_offset_call(dst[0], uses_vgpr_msb_indexing)
    L.append(f'  std::optional<uint32_t> rel_dst_base = {rel_dst_call};')
    L.append(
        '  uint64_t rel_dst_index = rel_dst_base '
        '? static_cast<uint64_t>(*rel_dst_base) + wf.m0() : UINT64_MAX;'
    )
    L.append(
        f'  bool rel_dst_valid = rel_dst_base && wf.m0() <= 1023u && '
        f'rel_dst_index + {dst[0]}.vgpr_count() <= wf.vgpr_alloc().count;'
    )
    L.append('  if (!rel_dst_valid) return;')
    L.append(
        f'  Operand rel_dst({dst[0]}.size_bits(), OperandType::OPR_VGPR, static_cast<int>(rel_dst_index));'
    )
    L.append('  uint64_t exec = wf.exec();')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')
    L.append(
        f'    amdgpu::RegisterAccess(wf).write_lane(rel_dst, lane, amdgpu::RegisterAccess(wf).read_lane({src[0]}, lane));'
    )
    L.append('  }')
    return '\n'.join(L)


def _stage_movrel_source(
    lines: list[str],
    supports_dpp: bool,
    supports_dpp8: bool,
    supports_sdwa: bool,
    inactive_uses_bound_ctrl: bool,
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
    lines.append(
        '  ScopedOperandDelegate rel_staged_src_binding('
        'rel_src, rel_staged_src ? &*rel_staged_src : nullptr);'
    )


def gen_vector_swaprel(
    dst: list[str], src: list[str], uses_vgpr_msb_indexing: bool = False
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
        f'  if (!rel_src_base || rel_src_index + {s}.vgpr_count() > wf.vgpr_alloc().count) return;',
        f'  if (!rel_dst_base || rel_dst_index + {d}.vgpr_count() > wf.vgpr_alloc().count) return;',
        f'  Operand rel_src({s}.size_bits(), OperandType::OPR_VGPR, static_cast<int>(rel_src_index));',
        f'  Operand rel_dst({d}.size_bits(), OperandType::OPR_VGPR, static_cast<int>(rel_dst_index));',
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


def gen_vector_sat_pack(dst: list[str], src: list[str], op: str | None) -> str:
    """Generate signed/unsigned saturating narrow-and-pack operations."""
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
    L.extend(
        [
            f'    amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, result);',
            '  }',
        ]
    )
    return '\n'.join(L)


def gen_vector_mullit(
    dst: list[str], src: list[str], is_vop3: bool, has_abs: bool
) -> str:
    """Generate V_MULLIT_F32 (the legacy lighting multiply)."""
    L = [
        '  uint64_t exec = wf.exec();',
        '  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {',
        '    if (!(exec & (1ULL << lane))) continue;',
        f'    float s0 = std::bit_cast<float>(amdgpu::RegisterAccess(wf).read_lane({src[0]}, lane));',
        f'    float s1 = std::bit_cast<float>(amdgpu::RegisterAccess(wf).read_lane({src[1]}, lane));',
        f'    float s2 = std::bit_cast<float>(amdgpu::RegisterAccess(wf).read_lane({src[2]}, lane));',
    ]
    if is_vop3:
        L.extend(vop3_src_mod('s0', 0, has_abs))
        L.extend(vop3_src_mod('s1', 1, has_abs))
        L.extend(vop3_src_mod('s2', 2, has_abs))
    L.extend(
        [
            '    float result;',
            '    if (s1 == -std::numeric_limits<float>::max() ||',
            '        s1 == -std::numeric_limits<float>::infinity() || std::isnan(s1) ||',
            '        s2 <= 0.0f || std::isnan(s2))',
            '      result = -std::numeric_limits<float>::max();',
            '    else',
            '      result = (s0 == 0.0f || s1 == 0.0f) ? 0.0f : s0 * s1;',
        ]
    )
    if is_vop3:
        L.extend(vop3_dst_mod('result'))
    L.extend(
        [
            f'    amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, std::bit_cast<uint32_t>(result));',
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
) -> str:
    """Generate V_MAD_U64_U32 / V_MAD_I64_I32 body.

    D.i64 = S0.i32 * S1.i32 + S2.i64 (signed)
    D.u64 = S0.u32 * S1.u32 + S2.u64 (unsigned)

    Sources S0 and S1 are 32-bit; the accumulator S2 and result D are
    64-bit VGPR pairs. The unsigned SDST-encoded form also writes the per-lane
    carry-out mask to SDST.
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
        L.append(
            '    uint64_t product = static_cast<uint64_t>(s0) * static_cast<uint64_t>(s1);'
        )
        L.append('    uint64_t result = product + s2;')
        if writes_carry:
            L.append('    constexpr uint64_t sign_bit = 1ULL << 63;')
            L.append(
                '    if (((~(product ^ s2) & (product ^ result)) & sign_bit) != 0)'
            )
            L.append('      carry |= 1ULL << lane;')
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
        if writes_carry:
            L.append('    if (result < product)')
            L.append('      carry |= 1ULL << lane;')
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


def gen_vector_mad_32_16(
    dst: list[str],
    src: list[str],
    dtype: str | None,
    is_vop3: bool = False,
    opsel: str | None = None,
) -> str:
    """Generate V_MAD_U32_U16 / V_MAD_I32_I16 body.

    D.u32 = S0.u16 * S1.u16 + S2.u32 (unsigned)
    D.i32 = S0.i16 * S1.i16 + S2.i32 (signed)
    """
    L = []
    if is_vop3 and opsel is None:
        opsel = _default_vop3_opsel_expr(dst + src)
    L.append('  uint64_t exec = wf.exec();')
    if is_vop3:
        L.append(f'  uint32_t opsel = {opsel};')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')
    if dtype == 'i32':
        if is_vop3:
            L.append(
                f'    uint32_t s0_raw = {_read_vop3_true16_src(src[0], "opsel", 0)};'
            )
            L.append(
                f'    uint32_t s1_raw = {_read_vop3_true16_src(src[1], "opsel", 1)};'
            )
        else:
            L.append(
                f'    uint32_t s0_raw = amdgpu::RegisterAccess(wf).read_lane({src[0]}, lane) & 0xFFFFu;'
            )
            L.append(
                f'    uint32_t s1_raw = amdgpu::RegisterAccess(wf).read_lane({src[1]}, lane) & 0xFFFFu;'
            )
        L.append('    int32_t s0 = static_cast<int16_t>(s0_raw);')
        L.append('    int32_t s1 = static_cast<int16_t>(s1_raw);')
        L.append(
            f'    int32_t s2 = static_cast<int32_t>(amdgpu::RegisterAccess(wf).read_lane({src[2]}, lane));'
        )
        result = (
            'amdgpu::vop3_integer_mad<int32_t, 16>('
            'static_cast<uint32_t>(s0), static_cast<uint32_t>(s1), '
            'static_cast<uint32_t>(s2), inst_.clamp)'
            if is_vop3
            else 'static_cast<uint32_t>(s0) * static_cast<uint32_t>(s1) + '
            'static_cast<uint32_t>(s2)'
        )
        L.append(
            f'    amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, {result});'
        )
    else:
        if is_vop3:
            L.append(f'    uint32_t s0 = {_read_vop3_true16_src(src[0], "opsel", 0)};')
            L.append(f'    uint32_t s1 = {_read_vop3_true16_src(src[1], "opsel", 1)};')
        else:
            L.append(
                f'    uint32_t s0 = amdgpu::RegisterAccess(wf).read_lane({src[0]}, lane) & 0xFFFFu;'
            )
            L.append(
                f'    uint32_t s1 = amdgpu::RegisterAccess(wf).read_lane({src[1]}, lane) & 0xFFFFu;'
            )
        L.append(
            f'    uint32_t s2 = amdgpu::RegisterAccess(wf).read_lane({src[2]}, lane);'
        )
        result = (
            'amdgpu::vop3_integer_mad<uint32_t, 16>(s0, s1, s2, inst_.clamp)'
            if is_vop3
            else 's0 * s1 + s2'
        )
        L.append(
            f'    amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, {result});'
        )
    L.append('  }')
    return '\n'.join(L)


def gen_vector_div_fixup(
    dst: list[str],
    src: list[str],
    dtype: str | None,
    is_vop3: bool = False,
    has_abs: bool = False,
) -> str:
    """Generate V_DIV_FIXUP body (corrects division result)."""
    L = []
    opsel = _default_vop3_opsel_expr(dst + src) if is_vop3 and dtype == 'f16' else None
    L.append('  uint64_t exec = wf.exec();')
    if opsel is not None:
        L.append(f'  uint32_t opsel = {opsel};')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')
    if dtype == 'f64':
        L.append(
            f'    double p = std::bit_cast<double>(amdgpu::RegisterAccess(wf).read_lane64({src[0]}, lane));'
        )
        L.append(
            f'    double b = std::bit_cast<double>(amdgpu::RegisterAccess(wf).read_lane64({src[1]}, lane));'
        )
        L.append(
            f'    double c = std::bit_cast<double>(amdgpu::RegisterAccess(wf).read_lane64({src[2]}, lane));'
        )
        if is_vop3:
            L.extend(vop3_src_mod('p', 0, has_abs))
            L.extend(vop3_src_mod('b', 1, has_abs))
            L.extend(vop3_src_mod('c', 2, has_abs))
        L.append('    double result;')
        L.append('    if (std::isnan(c)) result = c;')
        L.append('    else if (std::isnan(b)) result = b;')
        L.append(
            '    else if (c == 0.0 && b == 0.0) result = std::numeric_limits<double>::quiet_NaN();'
        )
        L.append(
            '    else if (std::isinf(c) && std::isinf(b)) result = std::numeric_limits<double>::quiet_NaN();'
        )
        L.append('    else if (b == 0.0) {')
        L.append(
            '      result = std::copysign(std::numeric_limits<double>::infinity(),'
        )
        L.append(
            '                             std::bit_cast<double>(std::bit_cast<uint64_t>(b) ^ std::bit_cast<uint64_t>(c)));'
        )
        L.append('    }')
        L.append(
            '    else if (c == 0.0) result = std::copysign(0.0, std::bit_cast<double>(std::bit_cast<uint64_t>(b) ^ std::bit_cast<uint64_t>(c)));'
        )
        L.append('    else if (std::isinf(c)) {')
        L.append(
            '      result = std::copysign(std::numeric_limits<double>::infinity(),'
        )
        L.append(
            '                             std::bit_cast<double>(std::bit_cast<uint64_t>(b) ^ std::bit_cast<uint64_t>(c)));'
        )
        L.append('    }')
        L.append(
            '    else if (std::isinf(b)) result = std::copysign(0.0, std::bit_cast<double>(std::bit_cast<uint64_t>(b) ^ std::bit_cast<uint64_t>(c)));'
        )
        L.append('    else result = p;')
        if is_vop3:
            L.extend(vop3_dst_mod_f64('result'))
        L.append(
            f'    amdgpu::RegisterAccess(wf).write_lane64({dst[0]}, lane, std::bit_cast<uint64_t>(result));'
        )
    elif dtype == 'f16':
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
        L.append('    float result;')
        L.append('    if (std::isnan(c)) result = c;')
        L.append('    else if (std::isnan(b)) result = b;')
        L.append(
            '    else if (c == 0.0f && b == 0.0f) result = std::numeric_limits<float>::quiet_NaN();'
        )
        L.append(
            '    else if (std::isinf(c) && std::isinf(b)) result = std::numeric_limits<float>::quiet_NaN();'
        )
        L.append('    else if (b == 0.0f) {')
        L.append('      result = std::copysign(std::numeric_limits<float>::infinity(),')
        L.append(
            '                             std::bit_cast<float>(std::bit_cast<uint32_t>(b) ^ std::bit_cast<uint32_t>(c)));'
        )
        L.append('    }')
        L.append(
            '    else if (c == 0.0f) result = std::copysign(0.0f, std::bit_cast<float>(std::bit_cast<uint32_t>(b) ^ std::bit_cast<uint32_t>(c)));'
        )
        L.append('    else if (std::isinf(c)) {')
        L.append('      result = std::copysign(std::numeric_limits<float>::infinity(),')
        L.append(
            '                             std::bit_cast<float>(std::bit_cast<uint32_t>(b) ^ std::bit_cast<uint32_t>(c)));'
        )
        L.append('    }')
        L.append(
            '    else if (std::isinf(b)) result = std::copysign(0.0f, std::bit_cast<float>(std::bit_cast<uint32_t>(b) ^ std::bit_cast<uint32_t>(c)));'
        )
        L.append('    else result = p;')
        if is_vop3:
            L.extend(vop3_dst_mod('result', omod_result_type='f16'))
            L.append(
                '    uint32_t result_bits = util::f32_to_f16_mode(result, wf.fp16_ovfl());'
            )
            L.append(
                '    result_bits = amdgpu::fp_mode::finalize_omod_f16(result_bits, effective_omod);'
            )
            L.append(
                f'    ::rocjitsu::amdgpu::write_vop3_true16_dst({dst[0]}, wf, lane, opsel, result_bits, true);'
            )
        else:
            L.append(
                f'    amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, util::f32_to_f16_mode(result, wf.fp16_ovfl()));'
            )
    else:
        L.append(
            f'    float p = std::bit_cast<float>(amdgpu::RegisterAccess(wf).read_lane({src[0]}, lane));'
        )
        L.append(
            f'    float b = std::bit_cast<float>(amdgpu::RegisterAccess(wf).read_lane({src[1]}, lane));'
        )
        L.append(
            f'    float c = std::bit_cast<float>(amdgpu::RegisterAccess(wf).read_lane({src[2]}, lane));'
        )
        if is_vop3:
            L.extend(vop3_src_mod('p', 0, has_abs))
            L.extend(vop3_src_mod('b', 1, has_abs))
            L.extend(vop3_src_mod('c', 2, has_abs))
        L.append('    float result;')
        L.append('    if (std::isnan(c)) result = c;')
        L.append('    else if (std::isnan(b)) result = b;')
        L.append(
            '    else if (c == 0.0f && b == 0.0f) result = std::numeric_limits<float>::quiet_NaN();'
        )
        L.append(
            '    else if (std::isinf(c) && std::isinf(b)) result = std::numeric_limits<float>::quiet_NaN();'
        )
        L.append('    else if (b == 0.0f) {')
        L.append('      result = std::copysign(std::numeric_limits<float>::infinity(),')
        L.append(
            '                             std::bit_cast<float>(std::bit_cast<uint32_t>(b) ^ std::bit_cast<uint32_t>(c)));'
        )
        L.append('    }')
        L.append(
            '    else if (c == 0.0f) result = std::copysign(0.0f, std::bit_cast<float>(std::bit_cast<uint32_t>(b) ^ std::bit_cast<uint32_t>(c)));'
        )
        L.append('    else if (std::isinf(c)) {')
        L.append('      result = std::copysign(std::numeric_limits<float>::infinity(),')
        L.append(
            '                             std::bit_cast<float>(std::bit_cast<uint32_t>(b) ^ std::bit_cast<uint32_t>(c)));'
        )
        L.append('    }')
        L.append(
            '    else if (std::isinf(b)) result = std::copysign(0.0f, std::bit_cast<float>(std::bit_cast<uint32_t>(b) ^ std::bit_cast<uint32_t>(c)));'
        )
        L.append('    else result = p;')
        if is_vop3:
            L.extend(vop3_dst_mod('result'))
        L.append(
            f'    amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, std::bit_cast<uint32_t>(result));'
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
    """Generate V_DIV_SCALE body per ISA pseudocode (CDNA4 p.363-365).

    S1 = denominator, S2 = numerator. S0 selects which to scale
    (S0==S1 → scale denominator, S0==S2 → scale numerator).
    VCC is set when V_DIV_FMAS must apply post-scaling.
    """
    is_f64 = dtype == 'f64'
    scale_exp = 128 if is_f64 else 64
    exp_threshold = 768 if is_f64 else 96
    tiny_exp = 53 if is_f64 else 23
    fp_type = 'double' if is_f64 else 'float'
    zero = '0.0' if is_f64 else '0.0f'
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
    L.append(f'    {fp_type} result = s0;')
    L.append('    bool set_vcc = false;')
    L.append(f'    if (s2 == {zero} || s1 == {zero}) {{')
    L.append(f'      // Zero numerator or denominator: pass through s0 unscaled.')
    L.append(f'      // Special-case handling (0/0, 0/x, x/0) is done by v_div_fixup.')
    L.append('    } else {')
    L.append('      int exp1 = 0, exp2 = 0;')
    L.append('      std::frexp(s1, &exp1);')
    L.append('      std::frexp(s2, &exp2);')
    L.append(f'      if (exp2 - exp1 >= {exp_threshold}) {{')
    L.append('        set_vcc = true;')
    L.append(f'        if (s0 == s1) result = std::ldexp(s0, {scale_exp});')
    L.append(f'      }} else if (std::fpclassify(s1) == FP_SUBNORMAL) {{')
    L.append(f'        result = std::ldexp(s0, {scale_exp});')
    if is_f64:
        L.append(f'      }} else if (std::fpclassify(1.0 / s1) == FP_SUBNORMAL &&')
        L.append(f'                 std::fpclassify(s2 / s1) == FP_SUBNORMAL) {{')
    else:
        L.append(
            f'      }} else if (std::fpclassify(1.0 / static_cast<double>(s1)) == FP_SUBNORMAL &&'
        )
        L.append(f'                 std::fpclassify(s2 / s1) == FP_SUBNORMAL) {{')
    L.append('        set_vcc = true;')
    L.append(f'        if (s0 == s1) result = std::ldexp(s0, {scale_exp});')
    if is_f64:
        L.append(f'      }} else if (std::fpclassify(1.0 / s1) == FP_SUBNORMAL) {{')
    else:
        L.append(
            f'      }} else if (std::fpclassify(1.0 / static_cast<double>(s1)) == FP_SUBNORMAL) {{'
        )
    L.append(f'        result = std::ldexp(s0, -{scale_exp});')
    L.append(f'      }} else if (std::fpclassify(s2 / s1) == FP_SUBNORMAL) {{')
    L.append('        set_vcc = true;')
    L.append(f'        if (s0 == s2) result = std::ldexp(s0, {scale_exp});')
    L.append(f'      }} else if (exp2 <= -{tiny_exp}) {{')
    L.append(f'        result = std::ldexp(s0, {scale_exp});')
    L.append('      }')
    L.append('    }')
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
    """Generate V_DIV_FMAS body (FMA with scale based on VCC)."""
    L = []
    L.append('  uint64_t exec = wf.exec();')
    L.append('  uint64_t vcc = wf.vcc();')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')
    if dtype == 'f64':
        L.append(
            f'    double s0 = std::bit_cast<double>(amdgpu::RegisterAccess(wf).read_lane64({src[0]}, lane));'
        )
        L.append(
            f'    double s1 = std::bit_cast<double>(amdgpu::RegisterAccess(wf).read_lane64({src[1]}, lane));'
        )
        L.append(
            f'    double s2 = std::bit_cast<double>(amdgpu::RegisterAccess(wf).read_lane64({src[2]}, lane));'
        )
        if is_vop3:
            L.extend(vop3_src_mod('s0', 0, has_abs))
            L.extend(vop3_src_mod('s1', 1, has_abs))
            L.extend(vop3_src_mod('s2', 2, has_abs))
        L.append('    double result = std::fma(s0, s1, s2);')
        L.append('    if (vcc & (1ULL << lane)) {')
        L.append('      result = std::ldexp(result, 64);')
        L.append('    }')
        L.append(
            f'    amdgpu::RegisterAccess(wf).write_lane64({dst[0]}, lane, std::bit_cast<uint64_t>(result));'
        )
    else:
        L.append(
            f'    float s0 = std::bit_cast<float>(amdgpu::RegisterAccess(wf).read_lane({src[0]}, lane));'
        )
        L.append(
            f'    float s1 = std::bit_cast<float>(amdgpu::RegisterAccess(wf).read_lane({src[1]}, lane));'
        )
        L.append(
            f'    float s2 = std::bit_cast<float>(amdgpu::RegisterAccess(wf).read_lane({src[2]}, lane));'
        )
        if is_vop3:
            L.extend(vop3_src_mod('s0', 0, has_abs))
            L.extend(vop3_src_mod('s1', 1, has_abs))
            L.extend(vop3_src_mod('s2', 2, has_abs))
        L.append('    float result = std::fma(s0, s1, s2);')
        L.append('    if (vcc & (1ULL << lane)) {')
        L.append('      result = std::ldexp(result, 32);')
        L.append('    }')
        L.append(
            f'    amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, std::bit_cast<uint32_t>(result));'
        )
    L.append('  }')
    return '\n'.join(L)


def gen_vector_dot(
    dst: list[str], src: list[str], op: str | None, dtype: str | None
) -> str:
    """Generate V_DOT*C body (dot product accumulate)."""
    L = []
    d = dst[0] if dst else src[0]
    s0, s1 = (src[0], src[1]) if dst else (src[1], src[2])
    L.append('  uint64_t exec = wf.exec();')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')
    L.append(f'    uint32_t a = amdgpu::RegisterAccess(wf).read_lane({s0}, lane);')
    L.append(f'    uint32_t b = amdgpu::RegisterAccess(wf).read_lane({s1}, lane);')
    L.append(f'    uint32_t acc = amdgpu::RegisterAccess(wf).read_lane({d}, lane);')
    if op == 'dot4c':
        L.append('    for (int i = 0; i < 4; ++i) {')
        L.append('      int8_t ea = static_cast<int8_t>((a >> (i * 8)) & 0xFF);')
        L.append('      int8_t eb = static_cast<int8_t>((b >> (i * 8)) & 0xFF);')
        L.append(
            '      acc += static_cast<uint32_t>(static_cast<int32_t>(ea) * static_cast<int32_t>(eb));'
        )
        L.append('    }')
    elif op == 'dot8c':
        L.append('    for (int i = 0; i < 8; ++i) {')
        L.append('      int32_t ea = static_cast<int32_t>((a >> (i * 4)) & 0xF);')
        L.append('      if (ea & 8) ea |= ~0xF;')
        L.append('      int32_t eb = static_cast<int32_t>((b >> (i * 4)) & 0xF);')
        L.append('      if (eb & 8) eb |= ~0xF;')
        L.append('      acc += static_cast<uint32_t>(ea * eb);')
        L.append('    }')
    elif op == 'dot2c' and dtype == 'f32':
        # V_DOT2C_F32_F16: D.f32 += f16_lo(A)*f16_lo(B) + f16_hi(A)*f16_hi(B)
        L.append('    float a0 = util::f16_to_f32(static_cast<uint16_t>(a & 0xFFFF));')
        L.append(
            '    float a1 = util::f16_to_f32(static_cast<uint16_t>((a >> 16) & 0xFFFF));'
        )
        L.append('    float b0 = util::f16_to_f32(static_cast<uint16_t>(b & 0xFFFF));')
        L.append(
            '    float b1 = util::f16_to_f32(static_cast<uint16_t>((b >> 16) & 0xFFFF));'
        )
        L.append('    float facc = std::bit_cast<float>(acc);')
        L.append('    facc += a0 * b0 + a1 * b1;')
        L.append('    acc = std::bit_cast<uint32_t>(facc);')
    elif op == 'dot2c' and dtype == 'i32':
        # V_DOT2C_I32_I16: D.i32 += i16_lo(A)*i16_lo(B) + i16_hi(A)*i16_hi(B)
        L.append('    int16_t a0 = static_cast<int16_t>(a & 0xFFFF);')
        L.append('    int16_t a1 = static_cast<int16_t>((a >> 16) & 0xFFFF);')
        L.append('    int16_t b0 = static_cast<int16_t>(b & 0xFFFF);')
        L.append('    int16_t b1 = static_cast<int16_t>((b >> 16) & 0xFFFF);')
        L.append(
            '    int64_t dot = static_cast<int64_t>(a0) * b0 + static_cast<int64_t>(a1) * b1;'
        )
        L.append('    acc += static_cast<uint32_t>(dot);')
    else:
        L.append(f'    (void)a; (void)b; // unhandled dot variant: {op}/{dtype}')
    L.append(f'    amdgpu::RegisterAccess(wf).write_lane({d}, lane, acc);')
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
    nbits = '16' if dtype == 'b16' else '32'
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
    L.append(f'    uint32_t result = 0;')
    L.append(f'    for (int i = 0; i < {nbits}; ++i) {{')
    L.append(
        '      uint32_t idx = (((a >> i) & 1) << 2) | (((b >> i) & 1) << 1) | ((c >> i) & 1);'
    )
    L.append('      result |= ((truth_table >> idx) & 1) << i;')
    L.append('    }')
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
    UNCHANGED. EXEC mask is IGNORED. Both vdst and src0 are outputs (LLVM:
    returns {vdst_new, src0_new}).
    """
    L = []
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
    L.append(
        f'      amdgpu::RegisterAccess(wf).write_lane({dst[1]}, base + i, tmp_dst[base + {stride} + i]);'
    )
    L.append(
        f'      amdgpu::RegisterAccess(wf).write_lane({dst[0]}, base + {stride} + i, tmp_src[base + i]);'
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
    if cross:
        L.append('    uint32_t row_base = lane & ~0x1Fu;')
        L.append('    uint32_t half = (lane ^ 0x10u) & 0x10u;')
        L.append('    uint32_t src_lane = row_base | half | sel;')
    else:
        L.append('    uint32_t src_lane = (lane & ~0xFu) | sel;')
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
    L.append('    uint32_t partner = lane ^ 32;')
    L.append('    if (partner < wf.wf_size())')
    L.append(
        f'      amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, snap[partner]);'
    )
    L.append('  }')
    return '\n'.join(L)


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
    L.append(
        f'  uint32_t lane_group_width = amdgpu::RegisterAccess(wf).read_scalar({src[2]});'
    )
    L.append(
        '  if (lane_group_width == 0 || (lane_group_width & (lane_group_width - 1)) != 0)'
    )
    L.append('    lane_group_width = wf.wf_size();')
    L.append('  lane_group_width = std::min(lane_group_width, wf.wf_size());')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')
    L.append('    uint32_t group_base = (lane / lane_group_width) * lane_group_width;')
    if op == 'bcast':
        L.append('    uint32_t src_offset = selector % lane_group_width;')
    elif op == 'down':
        L.append('    uint32_t offset = lane - group_base;')
        L.append('    uint32_t src_offset = offset + selector;')
    elif op == 'up':
        L.append('    uint32_t offset = lane - group_base;')
        L.append(
            '    uint32_t src_offset = (selector <= offset) ? (offset - selector) : lane_group_width;'
        )
    else:
        L.append('    uint32_t offset = lane - group_base;')
        L.append('    uint32_t src_offset = offset ^ selector;')
    L.append('    uint32_t src_lane = group_base + src_offset;')
    L.append('    if (src_offset >= lane_group_width || src_lane >= wf.wf_size()) {')
    L.append(f'      amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, 0);')
    L.append('      continue;')
    L.append('    }')
    L.append(
        f'    amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, snap[src_lane]);'
    )
    L.append('  }')
    return '\n'.join(L)


def gen_vector_permlane_idx_gen(dst: list[str], src: list[str]) -> str:
    """Generate V_PERMLANE_IDX_GEN_B32."""
    L = []
    L.append('  uint64_t exec = wf.exec();')
    L.append(f'  uint32_t selector = amdgpu::RegisterAccess(wf).read_scalar({src[1]});')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')
    L.append(
        f'    uint32_t value = amdgpu::RegisterAccess(wf).read_lane({src[0]}, lane);'
    )
    L.append(
        f'    amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, value ^ selector);'
    )
    L.append('  }')
    return '\n'.join(L)


def gen_vector_cvt_pk(
    dst: list[str],
    src: list[str],
    cls: str,
    op: str | None,
    *,
    opsel: str = '0u',
    dtype: str | None = None,
    is_vop3: bool = False,
    fp8_format_select: str | None = None,
    arch_name: str = '',
) -> str:
    """Generate pack/convert instructions."""
    L = []
    L.append('  uint64_t exec = wf.exec();')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')
    if cls == 'vector_cvt_pk_u8_f32':
        L.append(
            f'    float fval = std::bit_cast<float>(amdgpu::RegisterAccess(wf).read_lane({src[0]}, lane));'
        )
        L.append(
            f'    uint32_t byte_sel = amdgpu::RegisterAccess(wf).read_lane({src[1]}, lane) & 3;'
        )
        # V_CVT_PK_U8_F32 has 3 srcs; V_CVT_PKACCUM reads old from dst
        old_src = src[2] if len(src) > 2 else dst[0]
        L.append(
            f'    uint32_t old = amdgpu::RegisterAccess(wf).read_lane({old_src}, lane);'
        )
        L.append(
            '    uint32_t byte = static_cast<uint32_t>(std::clamp(fval, 0.0f, 255.0f));'
        )
        L.append('    uint32_t mask = ~(0xFFu << (byte_sel * 8));')
        L.append(
            f'    amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, (old & mask) | (byte << (byte_sel * 8)));'
        )
    elif cls == 'vector_cvt_pknorm':
        if dtype == 'f16':
            if is_vop3:
                L.append(
                    f'    float s0 = util::f16_to_f32(static_cast<uint16_t>({_read_vop3_true16_src(src[0], opsel, 0)}));'
                )
                L.append(
                    f'    float s1 = util::f16_to_f32(static_cast<uint16_t>({_read_vop3_true16_src(src[1], opsel, 1)}));'
                )
            else:
                L.append(
                    f'    float s0 = util::f16_to_f32(static_cast<uint16_t>(amdgpu::RegisterAccess(wf).read_lane({src[0]}, lane)));'
                )
                L.append(
                    f'    float s1 = util::f16_to_f32(static_cast<uint16_t>(amdgpu::RegisterAccess(wf).read_lane({src[1]}, lane)));'
                )
        else:
            L.append(
                f'    float s0 = std::bit_cast<float>(amdgpu::RegisterAccess(wf).read_lane({src[0]}, lane));'
            )
            L.append(
                f'    float s1 = std::bit_cast<float>(amdgpu::RegisterAccess(wf).read_lane({src[1]}, lane));'
            )
        if op == 'i16':
            L.append('    auto cvt_i16 = [](float f) -> int16_t {')
            L.append('      if (std::isnan(f)) return 0;')
            L.append(
                '      return static_cast<int16_t>(util::round_to_nearest_even(std::clamp(f * 32767.0f, -32768.0f, 32767.0f)));'
            )
            L.append('    };')
            L.append('    int16_t lo = cvt_i16(s0);')
            L.append('    int16_t hi = cvt_i16(s1);')
        else:  # u16
            L.append('    auto cvt_u16 = [](float f) -> uint16_t {')
            L.append('      if (std::isnan(f)) return 0;')
            L.append(
                '      return static_cast<uint16_t>(util::round_to_nearest_even(std::clamp(f * 65535.0f, 0.0f, 65535.0f)));'
            )
            L.append('    };')
            L.append('    uint16_t lo = cvt_u16(s0);')
            L.append('    uint16_t hi = cvt_u16(s1);')
        L.append(
            f'    amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, (static_cast<uint32_t>(hi) << 16) | (static_cast<uint32_t>(lo) & 0xFFFF));'
        )
    elif cls == 'vector_cvt_pkrtz_f16_f32':
        L.append(
            f'    float s0 = std::bit_cast<float>(amdgpu::RegisterAccess(wf).read_lane({src[0]}, lane));'
        )
        L.append(
            f'    float s1 = std::bit_cast<float>(amdgpu::RegisterAccess(wf).read_lane({src[1]}, lane));'
        )
        L.append(f'    uint32_t lo = util::f32_to_f16_rtz(s0);')
        L.append(f'    uint32_t hi = util::f32_to_f16_rtz(s1);')
        L.append(
            f'    amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, lo | (hi << 16));'
        )
    elif cls == 'vector_cvt_pk':
        if op in ('fp8_f32', 'bf8_f32', 'fp8_f16', 'bf8_f16'):
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
                f'    ::rocjitsu::amdgpu::write_vop3_true16_dst({dst[0]}, wf, lane, ({opsel}) & 0x8u, packed, true);'
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
            if op == 'u16_f32':
                L.append(
                    '    uint16_t lo = static_cast<uint16_t>(std::clamp(f0, 0.0f, 65535.0f));'
                )
                L.append(
                    '    uint16_t hi = static_cast<uint16_t>(std::clamp(f1, 0.0f, 65535.0f));'
                )
            else:
                L.append(
                    '    int16_t lo = static_cast<int16_t>(std::clamp(f0, -32768.0f, 32767.0f));'
                )
                L.append(
                    '    int16_t hi = static_cast<int16_t>(std::clamp(f1, -32768.0f, 32767.0f));'
                )
            L.append(
                f'    amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, (static_cast<uint32_t>(static_cast<uint16_t>(hi)) << 16) | static_cast<uint32_t>(static_cast<uint16_t>(lo)));'
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
        L.append(f'    uint32_t lo = util::f32_to_f16_mode(s0, wf.fp16_ovfl());')
        L.append(f'    uint32_t hi = util::f32_to_f16_mode(s1, wf.fp16_ovfl());')
        L.append(
            f'    amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, lo | (hi << 16));'
        )
    elif cls == 'vector_cvt_pk_bf16_f32':
        L.append(
            f'    float s0 = std::bit_cast<float>(amdgpu::RegisterAccess(wf).read_lane({src[0]}, lane));'
        )
        L.append(
            f'    float s1 = std::bit_cast<float>(amdgpu::RegisterAccess(wf).read_lane({src[1]}, lane));'
        )
        L.append(f'    uint32_t lo = util::f32_to_bf16_rne_mode(s0, wf.fp16_ovfl());')
        L.append(f'    uint32_t hi = util::f32_to_bf16_rne_mode(s1, wf.fp16_ovfl());')
        L.append(
            f'    amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, lo | (hi << 16));'
        )
    elif cls == 'vector_cvt_sr_pk_f16_f32':
        L.append(
            f'    float s0 = std::bit_cast<float>(amdgpu::RegisterAccess(wf).read_lane({src[0]}, lane));'
        )
        L.append(
            f'    float s1 = std::bit_cast<float>(amdgpu::RegisterAccess(wf).read_lane({src[1]}, lane));'
        )
        L.append(
            f'    uint32_t seed_lo = amdgpu::RegisterAccess(wf).read_lane({src[2]}, lane);'
        )
        L.append('    uint32_t seed_hi = util::prng_advance(seed_lo);')
        L.append(
            '    uint32_t lo = util::f32_to_f16_sr_mode(s0, seed_lo, wf.fp16_ovfl());'
        )
        L.append(
            '    uint32_t hi = util::f32_to_f16_sr_mode(s1, seed_hi, wf.fp16_ovfl());'
        )
        L.append(
            f'    amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, lo | (hi << 16));'
        )
    elif cls == 'vector_cvt_sr_pk_bf16_f32':
        L.append(
            f'    float s0 = std::bit_cast<float>(amdgpu::RegisterAccess(wf).read_lane({src[0]}, lane));'
        )
        L.append(
            f'    float s1 = std::bit_cast<float>(amdgpu::RegisterAccess(wf).read_lane({src[1]}, lane));'
        )
        L.append(
            f'    uint32_t seed_lo = amdgpu::RegisterAccess(wf).read_lane({src[2]}, lane);'
        )
        L.append('    uint32_t seed_hi = util::prng_advance(seed_lo);')
        L.append(
            '    uint32_t lo = util::f32_to_bf16_sr_mode(s0, seed_lo, wf.fp16_ovfl());'
        )
        L.append(
            '    uint32_t hi = util::f32_to_bf16_sr_mode(s1, seed_hi, wf.fp16_ovfl());'
        )
        L.append(
            f'    amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, lo | (hi << 16));'
        )
    elif cls == 'vector_pack_b32_f16':
        if is_vop3:
            L.append(f'    uint32_t s0 = {_read_vop3_true16_src(src[0], opsel, 0)};')
            L.append(f'    uint32_t s1 = {_read_vop3_true16_src(src[1], opsel, 1)};')
        else:
            L.append(
                f'    uint32_t s0 = amdgpu::RegisterAccess(wf).read_lane({src[0]}, lane) & 0xFFFF;'
            )
            L.append(
                f'    uint32_t s1 = amdgpu::RegisterAccess(wf).read_lane({src[1]}, lane) & 0xFFFF;'
            )
        L.append(
            f'    amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, s0 | (s1 << 16));'
        )
    elif cls == 'vector_cvt_sr_f16_f32':
        # Stochastic rounding: use src1 as random bits for rounding
        L.append(
            f'    float s0 = std::bit_cast<float>(amdgpu::RegisterAccess(wf).read_lane({src[0]}, lane));'
        )
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


def _scale_decode_call(fmt: str, raw_expr: str, arch_name: str = '') -> str:
    if fmt == 'fp4':
        return f'util::fp4_e2m1_to_f32(static_cast<uint8_t>({raw_expr}))'
    if fmt == 'fp6':
        return f'util::fp6_e2m3_to_f32(static_cast<uint8_t>({raw_expr}))'
    if fmt == 'bf6':
        return f'util::bf6_e3m2_to_f32(static_cast<uint8_t>({raw_expr}))'
    if fmt == 'fp8':
        return f"{fp8_helper_name(arch_name, 'util::fp8_e4m3_to_f32')}(static_cast<uint8_t>({raw_expr}))"
    if fmt == 'bf8':
        return f"{fp8_helper_name(arch_name, 'util::bf8_e5m2_to_f32')}(static_cast<uint8_t>({raw_expr}))"
    raise ValueError(f'unsupported scaled conversion input format: {fmt}')


def _scale_encode_call(fmt: str, value_expr: str, arch_name: str = '') -> str:
    if fmt == 'fp4':
        return f'util::f32_to_fp4_e2m1_rne({value_expr})'
    if fmt == 'fp6':
        return f'util::f32_to_fp6_e2m3_rne({value_expr})'
    if fmt == 'bf6':
        return f'util::f32_to_bf6_e3m2_rne({value_expr})'
    if fmt == 'fp8':
        return f"{_fp8_mode_rne_helper_name(arch_name, 'util::f32_to_fp8_e4m3_rne')}({value_expr}, wf.fp16_ovfl())"
    if fmt == 'bf8':
        return f"{_fp8_mode_rne_helper_name(arch_name, 'util::f32_to_bf8_e5m2_rne')}({value_expr}, wf.fp16_ovfl())"
    raise ValueError(f'unsupported scaled conversion output format: {fmt}')


def _scale_sr_encode_call(
    fmt: str, value_expr: str, seed_expr: str, arch_name: str = ''
) -> str:
    if fmt == 'fp4':
        return f'util::f32_to_fp4_e2m1_sr({value_expr}, {seed_expr})'
    if fmt == 'fp6':
        return f'util::f32_to_fp6_e2m3_sr({value_expr}, {seed_expr})'
    if fmt == 'bf6':
        return f'util::f32_to_bf6_e3m2_sr({value_expr}, {seed_expr})'
    if fmt == 'fp8':
        return f"{_fp8_mode_sr_helper_name(arch_name, 'util::f32_to_fp8_e4m3_sr')}({value_expr}, {seed_expr}, wf.fp16_ovfl())"
    if fmt == 'bf8':
        return f"{_fp8_mode_sr_helper_name(arch_name, 'util::f32_to_bf8_e5m2_sr')}({value_expr}, {seed_expr}, wf.fp16_ovfl())"
    raise ValueError(f'unsupported scaled SR conversion output format: {fmt}')


def _scale_lowp_bits(fmt: str) -> int:
    if fmt == 'fp4':
        return 4
    if fmt in ('fp6', 'bf6'):
        return 6
    if fmt in ('fp8', 'bf8'):
        return 8
    raise ValueError(f'unsupported scaled low-precision format: {fmt}')


def _scale_read_vgpr_base(L: list[str], var: str, operand: str) -> None:
    L.append(
        f'    uint32_t {var} = wf.vgpr_alloc().base + '
        f'*Isa::resolved_vgpr_offset(wf, {operand}.opr_type_, '
        f'{operand}.encoding_value_, {operand}.vgpr_msb_role());'
    )


def _scale_unpack_element_raw(fmt: str, arch_name: str = '') -> list[str]:
    bits = _scale_lowp_bits(fmt)
    mask = f'0x{((1 << bits) - 1):x}u'
    if bits == 4:
        return [
            '    auto read_scaled_src = [&](uint32_t index) -> float {',
            f'      uint32_t raw = (src_payload >> (index * 4u)) & {mask};',
            f"      return {_scale_decode_call(fmt, 'raw', arch_name)};",
            '    };',
        ]
    if bits == 8:
        return [
            '    auto read_scaled_src = [&](uint32_t index) -> float {',
            f'      uint32_t raw = static_cast<uint32_t>((src_payload >> (index * 8u)) & {mask});',
            f"      return {_scale_decode_call(fmt, 'raw', arch_name)};",
            '    };',
        ]
    return [
        '    auto read_scaled_src = [&](uint32_t index) -> float {',
        '      uint32_t bit = index * 6u;',
        '      uint32_t word = bit / 32u;',
        '      uint32_t shift = bit & 31u;',
        '      uint32_t raw = src_words[word] >> shift;',
        '      if (shift > 26u)',
        '        raw |= src_words[word + 1u] << (32u - shift);',
        f'      raw &= {mask};',
        f"      return {_scale_decode_call(fmt, 'raw', arch_name)};",
        '    };',
    ]


def _scale_source_reader(src_fmt: str) -> list[str]:
    if src_fmt == 'f32':
        return [
            '    auto read_scaled_input = [&](uint32_t index) -> float {',
            '      return std::bit_cast<float>(src_words[index]);',
            '    };',
        ]
    if src_fmt == 'f16':
        return [
            '    auto read_scaled_input = [&](uint32_t index) -> float {',
            '      uint32_t raw = src_words[index / 2u];',
            '      return util::f16_to_f32(static_cast<uint16_t>(raw >> ((index & 1u) * 16u)));',
            '    };',
        ]
    if src_fmt == 'bf16':
        return [
            '    auto read_scaled_input = [&](uint32_t index) -> float {',
            '      uint32_t raw = src_words[index / 2u];',
            '      return util::bf16_to_f32(static_cast<uint16_t>(raw >> ((index & 1u) * 16u)));',
            '    };',
        ]
    raise ValueError(f'unsupported scaled conversion source format: {src_fmt}')


def _scale_e8m0_unpack_scale(scale_src: str) -> list[str]:
    return [
        f'    uint32_t scale_payload = amdgpu::RegisterAccess(wf).read_lane({scale_src}, lane);',
        '    uint32_t scale_byte = (scale_payload >> ((inst_.opsel & 0x3u) * 8u)) & 0xffu;',
        '    float scale = util::e8m0_to_f32(static_cast<uint8_t>(scale_byte));',
    ]


def gen_vector_cvt_scale(
    dst: list[str],
    src: list[str],
    cls: str,
    op: str | None,
    arch_name: str = '',
) -> str:
    """Generate gfx1250 scaled packed low-precision conversions."""
    if cls != 'vector_cvt_scale' or op is None:
        raise ValueError('vector_cvt_scale requires an operation')

    stochastic = False
    parsed_op = op
    if parsed_op.startswith('sr_'):
        stochastic = True
        parsed_op = parsed_op[3:]

    parts = parsed_op.split('_')
    if len(parts) != 4 or parts[1] not in ('pk8', 'pk16'):
        raise ValueError(f'unsupported vector_cvt_scale operation: {op}')

    direction, pack_width, out_fmt, in_fmt = parts
    count = int(pack_width[2:])
    if direction == 'unpack':
        bits = _scale_lowp_bits(in_fmt)
        src_word_count = (count * bits + 31) // 32
        if out_fmt == 'f32':
            dst_word_count = count
        elif out_fmt in ('f16', 'bf16'):
            dst_word_count = count // 2
        else:
            raise ValueError(f'unsupported scaled unpack output format: {out_fmt}')
    elif direction == 'pack':
        bits = _scale_lowp_bits(out_fmt)
        src_word_count = count if in_fmt == 'f32' else count // 2
        dst_word_count = (count * bits + 31) // 32
    else:
        raise ValueError(f'unsupported vector_cvt_scale direction: {direction}')

    L: list[str] = []
    L.append('  uint64_t exec = wf.exec();')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')
    _scale_read_vgpr_base(L, 'dst_base', dst[0])
    _scale_read_vgpr_base(L, 'src_base', src[0])
    L.append('    amdgpu::RegisterAccess regs(wf);')
    # TODO(newling): Apply this check-before-observe pattern to other
    # multi-region instruction helpers, then make region acquisition require
    # successful preflight instead of returning validity-bearing views.
    L.append(
        f'    if (!regs.owns_vgpr_range(src_base, {src_word_count}u) || '
        f'!regs.owns_vgpr_range(dst_base, {dst_word_count}u))'
    )
    L.append('      continue;')
    scale_src = src[2] if stochastic else src[1]
    if direction == 'unpack':
        L.extend(_scale_e8m0_unpack_scale(scale_src))
    else:
        L.append(
            f'    float scale = std::bit_cast<float>(amdgpu::RegisterAccess(wf).read_lane({scale_src}, lane));'
        )
    if stochastic:
        L.append(
            f'    uint32_t seed = static_cast<uint32_t>(amdgpu::RegisterAccess(wf).read_lane({src[1]}, lane));'
        )

    if direction == 'unpack':
        L.append(
            f'    auto src_region = regs.read_vgpr_region('
            f'src_base, {src_word_count}u, 1ULL << lane);'
        )
        if count == 8 and bits == 4:
            L.append('    uint32_t src_payload = src_region.lane(0, lane);')
        elif count == 8 and bits == 8:
            L.append('    uint64_t src_payload = src_region.lane64(0, lane);')
        elif count == 16 and bits == 6:
            L.append('    uint32_t src_words[4] = {};')
            L.append('    for (uint32_t word = 0; word < 3u; ++word)')
            L.append('      src_words[word] = src_region.lane(word, lane);')
        else:
            raise ValueError(f'unsupported scaled unpack operation: {op}')

        L.extend(_scale_unpack_element_raw(in_fmt, arch_name))
        if out_fmt == 'f32':
            L.append(
                f'    auto dst_region = regs.write_vgpr_region('
                f'dst_base, {dst_word_count}u, 1ULL << lane);'
            )
            L.append(f'    for (uint32_t index = 0; index < {count}u; ++index) {{')
            L.append('      float value = read_scaled_src(index) * scale;')
            L.append(
                '      dst_region.set_lane(index, lane, std::bit_cast<uint32_t>(value));'
            )
            L.append('    }')
        elif out_fmt in ('f16', 'bf16'):
            conv = (
                'util::f32_to_f16_mode'
                if out_fmt == 'f16'
                else 'util::f32_to_bf16_rne_mode'
            )
            L.append(
                f'    auto dst_region = regs.write_vgpr_region('
                f'dst_base, {dst_word_count}u, 1ULL << lane);'
            )
            L.append(f'    uint32_t dst_words[{dst_word_count}] = {{}};')
            L.append(f'    for (uint32_t index = 0; index < {count}u; ++index) {{')
            L.append(
                f'      uint32_t bits = {conv}(read_scaled_src(index) * scale, wf.fp16_ovfl());'
            )
            L.append('      dst_words[index / 2u] |= bits << ((index & 1u) * 16u);')
            L.append('    }')
            L.append(f'    for (uint32_t word = 0; word < {dst_word_count}u; ++word)')
            L.append('      dst_region.set_lane(word, lane, dst_words[word]);')
    elif direction == 'pack':
        L.append(
            f'    auto src_region = regs.read_vgpr_region('
            f'src_base, {src_word_count}u, 1ULL << lane);'
        )
        L.append(
            f'    auto dst_region = regs.write_vgpr_region('
            f'dst_base, {dst_word_count}u, 1ULL << lane);'
        )
        L.append(f'    uint32_t src_words[{src_word_count}] = {{}};')
        L.append(f'    for (uint32_t word = 0; word < {src_word_count}u; ++word)')
        L.append('      src_words[word] = src_region.lane(word, lane);')
        L.extend(_scale_source_reader(in_fmt))
        L.append(f'    uint32_t dst_words[{dst_word_count}] = {{}};')
        L.append('    auto pack_scaled_dst = [&](uint32_t index, uint32_t code) {')
        L.append(f'      code &= 0x{((1 << bits) - 1):x}u;')
        L.append(f'      uint32_t bit = index * {bits}u;')
        L.append('      uint32_t word = bit / 32u;')
        L.append('      uint32_t shift = bit & 31u;')
        L.append('      dst_words[word] |= code << shift;')
        L.append(f'      if (shift + {bits}u > 32u)')
        L.append('        dst_words[word + 1u] |= code >> (32u - shift);')
        L.append('    };')
        L.append(f'    for (uint32_t index = 0; index < {count}u; ++index) {{')
        L.append('      float value = read_scaled_input(index) / scale;')
        if stochastic:
            L.append(
                f"      pack_scaled_dst(index, {_scale_sr_encode_call(out_fmt, 'value', 'seed', arch_name)});"
            )
            L.append('      seed = util::prng_advance(seed);')
        else:
            L.append(
                f"      pack_scaled_dst(index, {_scale_encode_call(out_fmt, 'value', arch_name)});"
            )
        L.append('    }')
        L.append(f'    for (uint32_t word = 0; word < {dst_word_count}u; ++word)')
        L.append('      dst_region.set_lane(word, lane, dst_words[word]);')

    L.append('  }')
    return '\n'.join(L)


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

    if op == 'pk_fp8_f32':
        _gen_pk_narrow_fp8(
            L,
            dst,
            src,
            _fp8_mode_rne_helper_name(ctx.arch_name, 'util::f32_to_fp8_e4m3_rne'),
            opsel,
            fp8_format_select=fp8_format_select,
            fp8_format_fn='util::f32_to_fp8_e5m3_rne_mode',
        )
    elif op == 'pk_bf8_f32':
        _gen_pk_narrow_fp8(
            L,
            dst,
            src,
            _fp8_mode_rne_helper_name(ctx.arch_name, 'util::f32_to_bf8_e5m2_rne'),
            opsel,
        )
    elif op == 'sr_fp8_f32':
        _gen_sr_narrow_fp8(
            L,
            dst,
            src,
            _fp8_mode_sr_helper_name(ctx.arch_name, 'util::f32_to_fp8_e4m3_sr'),
            opsel,
            fp8_format_select=fp8_format_select,
            fp8_format_fn='util::f32_to_fp8_e5m3_sr_mode',
        )
    elif op == 'sr_bf8_f32':
        _gen_sr_narrow_fp8(
            L,
            dst,
            src,
            _fp8_mode_sr_helper_name(ctx.arch_name, 'util::f32_to_bf8_e5m2_sr'),
            opsel,
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
    fp8_format_select: str | None = None,
    fp8_format_fn: str | None = None,
) -> None:
    """Pack 2×F32 → 2×FP8/BF8 with OPSEL[3] half selection, R-M-W."""
    L.append(
        f'    float s0 = std::bit_cast<float>(static_cast<uint32_t>(amdgpu::RegisterAccess(wf).read_lane({src[0]}, lane)));'
    )
    L.append(
        f'    float s1 = std::bit_cast<float>(static_cast<uint32_t>(amdgpu::RegisterAccess(wf).read_lane({src[1]}, lane)));'
    )
    if fp8_format_select is not None and fp8_format_fn is not None:
        L.append(
            f'    uint8_t r0 = ({fp8_format_select}) ? {fp8_format_fn}(s0, wf.fp16_ovfl()) : {cvt_fn}(s0, wf.fp16_ovfl());'
        )
        L.append(
            f'    uint8_t r1 = ({fp8_format_select}) ? {fp8_format_fn}(s1, wf.fp16_ovfl()) : {cvt_fn}(s1, wf.fp16_ovfl());'
        )
    else:
        L.append(f'    uint8_t r0 = {cvt_fn}(s0, wf.fp16_ovfl());')
        L.append(f'    uint8_t r1 = {cvt_fn}(s1, wf.fp16_ovfl());')
    L.append(
        '    uint32_t packed = static_cast<uint32_t>(r0) | (static_cast<uint32_t>(r1) << 8);'
    )
    L.append(
        f'    ::rocjitsu::amdgpu::write_vop3_true16_dst({dst[0]}, wf, lane, ({opsel}) & 0x8u, packed, true);'
    )


def _gen_sr_narrow_fp8(
    L: list[str],
    dst: list[str],
    src: list[str],
    cvt_fn: str,
    opsel: str = '0u',
    *,
    fp8_format_select: str | None = None,
    fp8_format_fn: str | None = None,
) -> None:
    """Single F32 → FP8/BF8 with stochastic rounding, OPSEL[3:2] byte sel, R-M-W."""
    L.append(
        f'    float s0 = std::bit_cast<float>(static_cast<uint32_t>(amdgpu::RegisterAccess(wf).read_lane({src[0]}, lane)));'
    )
    L.append(
        f'    uint32_t seed = amdgpu::RegisterAccess(wf).read_lane({src[1]}, lane);'
    )
    if fp8_format_select is not None and fp8_format_fn is not None:
        L.append(
            f'    uint8_t result = ({fp8_format_select}) ? {fp8_format_fn}(s0, seed, wf.fp16_ovfl()) : {cvt_fn}(s0, seed, wf.fp16_ovfl());'
        )
    else:
        L.append(f'    uint8_t result = {cvt_fn}(s0, seed, wf.fp16_ovfl());')
    L.append(f'    uint32_t dst_byte = ({opsel} >> 2) & 0x3;')
    L.append(
        f'    uint32_t old = amdgpu::RegisterAccess(wf).read_lane({dst[0]}, lane);'
    )
    L.append('    uint32_t mask = ~(0xFFu << (dst_byte * 8));')
    L.append(
        f'    amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, (old & mask) | (static_cast<uint32_t>(result) << (dst_byte * 8)));'
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


# ---------------------------------------------------------------------------
# Scaled FP8/BF8/FP4/FP6/BF6 format conversion instructions (CDNA4)
# ---------------------------------------------------------------------------

_NARROW_FMTS = frozenset({'fp8', 'bf8', 'fp4', 'fp6', 'bf6'})
_WIDE_FMTS = frozenset({'f32', 'f16', 'bf16'})

# Mapping from narrow format name → util:: conversion functions
_NARROW_TO_F32 = {
    'fp8': 'util::fp8_e4m3_to_f32',
    'bf8': 'util::bf8_e5m2_to_f32',
    'fp4': 'util::fp4_e2m1_to_f32',
    'fp6': 'util::fp6_e2m3_to_f32',
    'bf6': 'util::bf6_e3m2_to_f32',
}
_F32_TO_NARROW_RNE = {
    'fp8': 'util::f32_to_fp8_e4m3_rne',
    'bf8': 'util::f32_to_bf8_e5m2_rne',
    'fp4': 'util::f32_to_fp4_e2m1_rne',
    'fp6': 'util::f32_to_fp6_e2m3_rne',
    'bf6': 'util::f32_to_bf6_e3m2_rne',
}
_F32_TO_NARROW_SR = {
    'fp8': 'util::f32_to_fp8_e4m3_sr',
    'bf8': 'util::f32_to_bf8_e5m2_sr',
    'fp4': 'util::f32_to_fp4_e2m1_sr',
    'fp6': 'util::f32_to_fp6_e2m3_sr',
    'bf6': 'util::f32_to_bf6_e3m2_sr',
}


def _narrow_to_f32_name(arch_name: str, fmt: str) -> str:
    return fp8_helper_name(arch_name, _NARROW_TO_F32[fmt])


def _f32_to_narrow_rne_name(arch_name: str, fmt: str) -> str:
    return fp8_helper_name(arch_name, _F32_TO_NARROW_RNE[fmt])


def _f32_to_narrow_sr_name(arch_name: str, fmt: str) -> str:
    return fp8_helper_name(arch_name, _F32_TO_NARROW_SR[fmt])


def _f32_to_narrow_rne_mode_name(arch_name: str, fmt: str) -> str:
    # FP16_OVFL is applied at the final FP8/BF8 narrowing step after any
    # source widening or SCALEF32 adjustment has produced the float value. Plain
    # F16/BF16-source FP8/BF8 converts bypass this mode-aware helper, while
    # SCALEF32 converts use it because scale application creates an F32
    # intermediate before narrowing.
    if fmt == 'fp8':
        return _fp8_mode_rne_helper_name(arch_name, 'util::f32_to_fp8_e4m3_rne')
    if fmt == 'bf8':
        return _fp8_mode_rne_helper_name(arch_name, 'util::f32_to_bf8_e5m2_rne')
    return _f32_to_narrow_rne_name(arch_name, fmt)


def _f32_to_narrow_sr_mode_name(arch_name: str, fmt: str) -> str:
    # Keep SR behavior aligned with the RNE mode helper above: only conversions
    # that intentionally request final FP8/BF8 mode handling come through here.
    if fmt == 'fp8':
        return _fp8_mode_sr_helper_name(arch_name, 'util::f32_to_fp8_e4m3_sr')
    if fmt == 'bf8':
        return _fp8_mode_sr_helper_name(arch_name, 'util::f32_to_bf8_e5m2_sr')
    return _f32_to_narrow_sr_name(arch_name, fmt)


def _parse_scalef32_op(op: str):
    """Parse a CVT_SCALEF32 operation suffix into components.

    Returns (stochastic, mode, dst_fmt, src_fmt, direction).
    """
    s = op
    stochastic = False
    if s.startswith('sr_'):
        stochastic = True
        s = s[3:]

    if s.startswith('2xpk16_'):
        mode = '2xpk16'
        s = s[7:]
    elif s.startswith('pk32_'):
        mode = 'pk32'
        s = s[5:]
    elif s.startswith('pk_'):
        mode = 'pk'
        s = s[3:]
    else:
        mode = 'single'

    parts = s.split('_')
    assert len(parts) == 2, f'unexpected scalef32 suffix: {op}'
    dst_fmt, src_fmt = parts

    if dst_fmt in _NARROW_FMTS and src_fmt in _WIDE_FMTS:
        direction = 'narrow'
    elif dst_fmt in _WIDE_FMTS and src_fmt in _NARROW_FMTS:
        direction = 'widen'
    else:
        raise ValueError(f'cannot determine direction for {op}: {dst_fmt} vs {src_fmt}')

    return stochastic, mode, dst_fmt, src_fmt, direction


def _read_as_f32(src_name: str, src_fmt: str) -> str:
    """Return C++ expression to read a source value as float."""
    if src_fmt == 'f32':
        return f'std::bit_cast<float>(static_cast<uint32_t>(amdgpu::RegisterAccess(wf).read_lane({src_name}, lane)))'
    elif src_fmt == 'f16':
        return f'util::f16_to_f32(static_cast<uint16_t>(amdgpu::RegisterAccess(wf).read_lane({src_name}, lane) & 0xFFFF))'
    elif src_fmt == 'bf16':
        return f'util::bf16_to_f32(static_cast<uint16_t>(amdgpu::RegisterAccess(wf).read_lane({src_name}, lane) & 0xFFFF))'
    raise ValueError(f'unsupported source format: {src_fmt}')


def _write_as_fmt(dst_name: str, dst_fmt: str, val_expr: str) -> str:
    """Return C++ statement to write a float result in the target format."""
    if dst_fmt == 'f32':
        return f'amdgpu::RegisterAccess(wf).write_lane({dst_name}, lane, std::bit_cast<uint32_t>({val_expr}));'
    elif dst_fmt == 'f16':
        return f'amdgpu::RegisterAccess(wf).write_lane({dst_name}, lane, static_cast<uint32_t>(util::f32_to_f16_mode({val_expr}, wf.fp16_ovfl())));'
    elif dst_fmt == 'bf16':
        return f'amdgpu::RegisterAccess(wf).write_lane({dst_name}, lane, static_cast<uint32_t>(util::f32_to_bf16_rne_mode({val_expr}, wf.fp16_ovfl())));'
    raise ValueError(f'unsupported dst format: {dst_fmt}')


def _nan_for_fmt(dst_fmt: str) -> str:
    """Return the quiet-NaN bit pattern for a wide format, or 0 for narrow."""
    if dst_fmt in _NARROW_FMTS:
        return '0u'
    if dst_fmt == 'f32':
        return '0x7FC00000u'
    if dst_fmt == 'f16':
        return 'static_cast<uint32_t>(0x7E00u)'
    if dst_fmt == 'bf16':
        return 'static_cast<uint32_t>(0x7FC0u)'
    return '0x7FC00000u'


def gen_cvt_scalef32(ctx) -> str:
    """Generate execute body for V_CVT_SCALEF32_* instructions."""
    from amdisa.codegen.execute import ExecuteContext

    op = ctx.op
    dst = ctx.dst_ops
    src = ctx.src_ops
    stochastic, mode, dst_fmt, src_fmt, direction = _parse_scalef32_op(op)

    if mode in ('pk32', '2xpk16', 'sr_pk32'):
        return _gen_wide_scalef32(ctx, stochastic, mode, dst_fmt, src_fmt, direction)

    if direction == 'narrow':
        if stochastic:
            return _gen_sr_scalef32(ctx, mode, dst_fmt, src_fmt)
        return _gen_narrow_scalef32(ctx, mode, dst_fmt, src_fmt)
    else:
        return _gen_widen_scalef32(ctx, mode, dst_fmt, src_fmt)


def _scale_preamble(L: list[str], scale_src: str) -> None:
    """Emit per-lane scale factor extraction with E8M0 NaN check."""
    L.append(
        f'    uint32_t scale_bits = amdgpu::RegisterAccess(wf).read_lane({scale_src}, lane);'
    )
    L.append('    uint32_t biased_exp = (scale_bits >> 23) & 0xFFu;')


def _gen_narrow_scalef32(ctx, mode: str, dst_fmt: str, src_fmt: str) -> str:
    """Narrowing: F32/F16/BF16 → FP8/BF8/FP4, with scale, RNE."""
    dst = ctx.dst_ops
    src = ctx.src_ops
    cvt_fn = _f32_to_narrow_rne_mode_name(ctx.arch_name, dst_fmt)
    nan_val = _nan_for_fmt(dst_fmt)

    L = []
    L.append('  uint64_t exec = wf.exec();')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')

    if mode == 'pk' and src_fmt == 'f32' and dst_fmt != 'fp4':
        # 3-src: src0=F32[0], src1=F32[1], src2=scale
        scale_src = src[2]
        _scale_preamble(L, scale_src)
        L.append(
            f'    if (biased_exp == 0xFFu) {{ amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, amdgpu::RegisterAccess(wf).read_lane({dst[0]}, lane)); continue; }}'
        )
        L.append(
            '    double scale = std::ldexp(1.0, static_cast<int>(biased_exp) - 127);'
        )
        L.append(
            f'    float s0 = static_cast<float>(static_cast<double>(std::bit_cast<float>(static_cast<uint32_t>(amdgpu::RegisterAccess(wf).read_lane({src[0]}, lane)))) / scale);'
        )
        L.append(
            f'    float s1 = static_cast<float>(static_cast<double>(std::bit_cast<float>(static_cast<uint32_t>(amdgpu::RegisterAccess(wf).read_lane({src[1]}, lane)))) / scale);'
        )
        L.append(f"    uint8_t r0 = {_narrow_rne_encode_call(dst_fmt, cvt_fn, 's0')};")
        L.append(f"    uint8_t r1 = {_narrow_rne_encode_call(dst_fmt, cvt_fn, 's1')};")
        L.append(
            '    uint32_t packed = static_cast<uint32_t>(r0) | (static_cast<uint32_t>(r1) << 8);'
        )
        L.append('    bool hi = (inst_.op_sel >> 3) & 1;')
        L.append(
            f'    uint32_t old = amdgpu::RegisterAccess(wf).read_lane({dst[0]}, lane);'
        )
        L.append('    if (hi)')
        L.append(
            f'      amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, (old & 0xFFFFu) | (packed << 16));'
        )
        L.append('    else')
        L.append(
            f'      amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, (old & 0xFFFF0000u) | (packed & 0xFFFFu));'
        )
    elif mode == 'pk' and src_fmt in ('f16', 'bf16'):
        # 2-src: src0=packed 2xF16/BF16, src1=scale
        scale_src = src[1]
        _scale_preamble(L, scale_src)
        L.append(
            f'    if (biased_exp == 0xFFu) {{ amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, amdgpu::RegisterAccess(wf).read_lane({dst[0]}, lane)); continue; }}'
        )
        L.append(
            '    double scale = std::ldexp(1.0, static_cast<int>(biased_exp) - 127);'
        )
        L.append(
            f'    uint32_t packed_src = amdgpu::RegisterAccess(wf).read_lane({src[0]}, lane);'
        )
        if src_fmt == 'f16':
            L.append(
                '    float s0 = static_cast<float>(static_cast<double>(util::f16_to_f32(static_cast<uint16_t>(packed_src & 0xFFFF))) / scale);'
            )
            L.append(
                '    float s1 = static_cast<float>(static_cast<double>(util::f16_to_f32(static_cast<uint16_t>((packed_src >> 16) & 0xFFFF))) / scale);'
            )
        else:
            L.append(
                '    float s0 = static_cast<float>(static_cast<double>(util::bf16_to_f32(static_cast<uint16_t>(packed_src & 0xFFFF))) / scale);'
            )
            L.append(
                '    float s1 = static_cast<float>(static_cast<double>(util::bf16_to_f32(static_cast<uint16_t>((packed_src >> 16) & 0xFFFF))) / scale);'
            )
        L.append(f"    uint8_t r0 = {_narrow_rne_encode_call(dst_fmt, cvt_fn, 's0')};")
        L.append(f"    uint8_t r1 = {_narrow_rne_encode_call(dst_fmt, cvt_fn, 's1')};")
        if dst_fmt == 'fp4':
            L.append(
                '    uint32_t packed = static_cast<uint32_t>(r0 & 0xF) | (static_cast<uint32_t>(r1 & 0xF) << 4);'
            )
            L.append('    uint32_t dst_byte = (inst_.op_sel >> 2) & 0x3;')
            L.append(
                f'    uint32_t old = amdgpu::RegisterAccess(wf).read_lane({dst[0]}, lane);'
            )
            L.append('    uint32_t mask = ~(0xFFu << (dst_byte * 8));')
            L.append(
                f'    amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, (old & mask) | (packed << (dst_byte * 8)));'
            )
        else:
            L.append(
                '    uint32_t packed = static_cast<uint32_t>(r0) | (static_cast<uint32_t>(r1) << 8);'
            )
            L.append('    bool hi = (inst_.op_sel >> 3) & 1;')
            L.append(
                f'    uint32_t old = amdgpu::RegisterAccess(wf).read_lane({dst[0]}, lane);'
            )
            L.append('    if (hi)')
            L.append(
                f'      amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, (old & 0xFFFFu) | (packed << 16));'
            )
            L.append('    else')
            L.append(
                f'      amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, (old & 0xFFFF0000u) | (packed & 0xFFFFu));'
            )
    elif mode == 'pk' and dst_fmt == 'fp4' and src_fmt == 'f32':
        # 3-src: src0=F32[0], src1=F32[1], src2=scale → packed 2×FP4 (8-bit)
        scale_src = src[2]
        _scale_preamble(L, scale_src)
        L.append(
            f'    if (biased_exp == 0xFFu) {{ amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, amdgpu::RegisterAccess(wf).read_lane({dst[0]}, lane)); continue; }}'
        )
        L.append(
            '    double scale = std::ldexp(1.0, static_cast<int>(biased_exp) - 127);'
        )
        L.append(
            f'    float s0 = static_cast<float>(static_cast<double>(std::bit_cast<float>(static_cast<uint32_t>(amdgpu::RegisterAccess(wf).read_lane({src[0]}, lane)))) / scale);'
        )
        L.append(
            f'    float s1 = static_cast<float>(static_cast<double>(std::bit_cast<float>(static_cast<uint32_t>(amdgpu::RegisterAccess(wf).read_lane({src[1]}, lane)))) / scale);'
        )
        L.append(f"    uint8_t r0 = {_narrow_rne_encode_call(dst_fmt, cvt_fn, 's0')};")
        L.append(f"    uint8_t r1 = {_narrow_rne_encode_call(dst_fmt, cvt_fn, 's1')};")
        L.append(
            '    uint32_t packed = static_cast<uint32_t>(r0 & 0xF) | (static_cast<uint32_t>(r1 & 0xF) << 4);'
        )
        L.append('    uint32_t dst_byte = (inst_.op_sel >> 2) & 0x3;')
        L.append(
            f'    uint32_t old = amdgpu::RegisterAccess(wf).read_lane({dst[0]}, lane);'
        )
        L.append('    uint32_t mask = ~(0xFFu << (dst_byte * 8));')
        L.append(
            f'    amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, (old & mask) | (packed << (dst_byte * 8)));'
        )

    L.append('  }')
    return '\n'.join(L)


def _gen_sr_scalef32(ctx, mode: str, dst_fmt: str, src_fmt: str) -> str:
    """Narrowing with stochastic rounding: scale + SR."""
    dst = ctx.dst_ops
    src = ctx.src_ops
    cvt_fn = _f32_to_narrow_sr_mode_name(ctx.arch_name, dst_fmt)
    op = ctx.op

    L = []
    is_sr_pk_fp4_f32 = op == 'sr_pk_fp4_f32'
    if is_sr_pk_fp4_f32:
        L.append('  auto &cu = wf.cu();')
        L.append('  uint32_t vb = wf.vgpr_alloc().base;')
        L.append(
            f'  auto data_src_off = Isa::resolved_vgpr_offset({src[0]}.opr_type_, '
            f'{src[0]}.encoding_value_);'
        )
        L.append('  if (!data_src_off)')
        L.append('    throw util::UnimplementedInst(mnemonic());')
        L.append(
            f'  uint32_t data_src_base = vb + amdgpu::apply_gpr_idx('
            f'wf, *data_src_off, {src[0]}.vgpr_msb_role());'
        )
    L.append('  uint64_t exec = wf.exec();')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')

    if mode == 'pk' and dst_fmt == 'fp4':
        data_src, random_src, scale_src = src[0], src[1], src[2]
        _scale_preamble(L, scale_src)
        L.append(
            f'    if (biased_exp == 0xFFu) {{ amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, amdgpu::RegisterAccess(wf).read_lane({dst[0]}, lane)); continue; }}'
        )
        L.append(
            '    double scale = std::ldexp(1.0, static_cast<int>(biased_exp) - 127);'
        )
        if is_sr_pk_fp4_f32:
            L.append(
                f'    uint32_t seed_lo = static_cast<uint32_t>(amdgpu::RegisterAccess(wf).read_lane({random_src}, lane));'
            )
            L.append(
                f'    float val0 = std::bit_cast<float>(static_cast<uint32_t>(amdgpu::RegisterAccess(wf).read_lane({data_src}, lane)));'
            )
            L.append(
                '    float val1 = std::bit_cast<float>('
                'amdgpu::RegisterAccess(cu).read_vgpr(data_src_base + 1, lane));'
            )
        else:
            L.append(
                f'    uint32_t packed_src = amdgpu::RegisterAccess(wf).read_lane({data_src}, lane);'
            )
            L.append(
                f'    uint32_t seed_lo = amdgpu::RegisterAccess(wf).read_lane({random_src}, lane);'
            )
        if is_sr_pk_fp4_f32:
            L.append(
                '    float s0 = static_cast<float>(static_cast<double>(val0) / scale);'
            )
            L.append(
                '    float s1 = static_cast<float>(static_cast<double>(val1) / scale);'
            )
            L.append(
                f"    uint8_t r0 = {_narrow_sr_encode_call(dst_fmt, cvt_fn, 's0', 'seed_lo')};"
            )
            L.append('    uint32_t seed_hi = util::prng_advance(seed_lo);')
            L.append(
                f"    uint8_t r1 = {_narrow_sr_encode_call(dst_fmt, cvt_fn, 's1', 'seed_hi')};"
            )
        elif src_fmt == 'f16':
            L.append(
                '    float s0 = static_cast<float>(static_cast<double>(util::f16_to_f32(static_cast<uint16_t>(packed_src & 0xFFFF))) / scale);'
            )
            L.append(
                '    float s1 = static_cast<float>(static_cast<double>(util::f16_to_f32(static_cast<uint16_t>((packed_src >> 16) & 0xFFFF))) / scale);'
            )
            L.append(
                f"    uint8_t r0 = {_narrow_sr_encode_call(dst_fmt, cvt_fn, 's0', 'seed_lo')};"
            )
            L.append('    uint32_t seed_hi = util::prng_advance(seed_lo);')
            L.append(
                f"    uint8_t r1 = {_narrow_sr_encode_call(dst_fmt, cvt_fn, 's1', 'seed_hi')};"
            )
        elif src_fmt == 'bf16':
            L.append(
                '    float s0 = static_cast<float>(static_cast<double>(util::bf16_to_f32(static_cast<uint16_t>(packed_src & 0xFFFF))) / scale);'
            )
            L.append(
                '    float s1 = static_cast<float>(static_cast<double>(util::bf16_to_f32(static_cast<uint16_t>((packed_src >> 16) & 0xFFFF))) / scale);'
            )
            L.append(
                f"    uint8_t r0 = {_narrow_sr_encode_call(dst_fmt, cvt_fn, 's0', 'seed_lo')};"
            )
            L.append('    uint32_t seed_hi = util::prng_advance(seed_lo);')
            L.append(
                f"    uint8_t r1 = {_narrow_sr_encode_call(dst_fmt, cvt_fn, 's1', 'seed_hi')};"
            )
        L.append(
            '    uint32_t packed = static_cast<uint32_t>(r0 & 0xF) | (static_cast<uint32_t>(r1 & 0xF) << 4);'
        )
        L.append('    uint32_t dst_byte = (inst_.op_sel >> 2) & 0x3;')
        L.append(
            f'    uint32_t old = amdgpu::RegisterAccess(wf).read_lane({dst[0]}, lane);'
        )
        L.append('    uint32_t mask = ~(0xFFu << (dst_byte * 8));')
        L.append(
            f'    amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, (old & mask) | (packed << (dst_byte * 8)));'
        )
    else:
        # SR single FP8/BF8: src0=data, src1=random, src2=scale
        data_src, random_src, scale_src = src[0], src[1], src[2]
        _scale_preamble(L, scale_src)
        L.append(
            f'    if (biased_exp == 0xFFu) {{ amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, amdgpu::RegisterAccess(wf).read_lane({dst[0]}, lane)); continue; }}'
        )
        L.append(
            '    double scale = std::ldexp(1.0, static_cast<int>(biased_exp) - 127);'
        )
        if src_fmt == 'f32':
            L.append(
                f'    float val = std::bit_cast<float>(static_cast<uint32_t>(amdgpu::RegisterAccess(wf).read_lane({data_src}, lane)));'
            )
        elif src_fmt == 'f16':
            L.append(
                f'    uint32_t src_packed = amdgpu::RegisterAccess(wf).read_lane({data_src}, lane);'
            )
            L.append(
                '    float val = util::f16_to_f32(static_cast<uint16_t>(src_packed & 0xFFFF));'
            )
        else:
            L.append(
                f'    uint32_t src_packed = amdgpu::RegisterAccess(wf).read_lane({data_src}, lane);'
            )
            L.append(
                '    float val = util::bf16_to_f32(static_cast<uint16_t>(src_packed & 0xFFFF));'
            )
        L.append(
            '    float scaled = static_cast<float>(static_cast<double>(val) / scale);'
        )
        L.append(
            f'    uint32_t seed = amdgpu::RegisterAccess(wf).read_lane({random_src}, lane);'
        )
        L.append(
            f"    uint8_t result = {_narrow_sr_encode_call(dst_fmt, cvt_fn, 'scaled', 'seed')};"
        )
        L.append('    uint32_t dst_byte = (inst_.op_sel >> 2) & 0x3;')
        L.append(
            f'    uint32_t old = amdgpu::RegisterAccess(wf).read_lane({dst[0]}, lane);'
        )
        L.append('    uint32_t mask = ~(0xFFu << (dst_byte * 8));')
        L.append(
            f'    amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, (old & mask) | (static_cast<uint32_t>(result) << (dst_byte * 8)));'
        )

    L.append('  }')
    return '\n'.join(L)


def _gen_widen_scalef32(ctx, mode: str, dst_fmt: str, src_fmt: str) -> str:
    """Widening: FP8/BF8/FP4 → F32/F16/BF16, with scale."""
    dst = ctx.dst_ops
    src = ctx.src_ops
    cvt_fn = _narrow_to_f32_name(ctx.arch_name, src_fmt)
    nan_val = _nan_for_fmt(dst_fmt)

    L = []
    L.append('  uint64_t exec = wf.exec();')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')

    scale_src = src[1]
    _scale_preamble(L, scale_src)

    if mode == 'single':
        # Single widening: src0 = narrow value (byte selected by OPSEL[1:0])
        if dst_fmt in ('f16', 'bf16'):
            conv = (
                'util::f32_to_f16_mode'
                if dst_fmt == 'f16'
                else 'util::f32_to_bf16_rne_mode'
            )
            L.append('    bool hi = (inst_.op_sel >> 3) & 1;')
            L.append(
                f'    uint32_t old = amdgpu::RegisterAccess(wf).read_lane({dst[0]}, lane);'
            )
            L.append('    if (biased_exp == 0xFFu) {')
            L.append(f'      uint32_t nv = {nan_val};')
            L.append(
                f'      amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, hi ? ((old & 0xFFFFu) | (nv << 16)) : ((old & 0xFFFF0000u) | nv));'
            )
            L.append('      continue;')
            L.append('    }')
        else:
            L.append(
                f'    if (biased_exp == 0xFFu) {{ amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, {nan_val}); continue; }}'
            )
        L.append(
            '    double scale = std::ldexp(1.0, static_cast<int>(biased_exp) - 127);'
        )
        L.append(
            f'    uint32_t src_raw = amdgpu::RegisterAccess(wf).read_lane({src[0]}, lane);'
        )
        L.append('    uint32_t src_byte = inst_.op_sel & 0x3;')
        if src_fmt == 'fp4':
            L.append(
                '    uint8_t narrow_val = static_cast<uint8_t>((src_raw >> (src_byte * 8)) & 0xF);'
            )
        else:
            L.append(
                '    uint8_t narrow_val = static_cast<uint8_t>((src_raw >> (src_byte * 8)) & 0xFF);'
            )
        L.append(
            f'    float f = static_cast<float>(static_cast<double>({cvt_fn}(narrow_val)) * scale);'
        )
        if dst_fmt in ('f16', 'bf16'):
            L.append(
                f'    uint32_t bits = static_cast<uint32_t>({conv}(f, wf.fp16_ovfl()));'
            )
            L.append(
                f'    amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, hi ? ((old & 0xFFFFu) | (bits << 16)) : ((old & 0xFFFF0000u) | (bits & 0xFFFFu)));'
            )
        else:
            L.append(f'    {_write_as_fmt(dst[0], dst_fmt, "f")}')
    elif mode == 'pk':
        # Packed widening: 2 narrow values → 2 wide values
        if src_fmt == 'fp4':
            # OPSEL[1:0] selects source byte
            L.append(f'    if (biased_exp == 0xFFu) {{')
            if dst_fmt == 'f32':
                L.append(
                    f'      amdgpu::RegisterAccess(wf).write_lane64({dst[0]}, lane, 0x7FC000007FC00000ULL); continue;'
                )
            else:
                L.append(
                    f'      amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, {nan_val} | ({nan_val} << 16)); continue;'
                )
            L.append('    }')
            L.append(
                '    double scale = std::ldexp(1.0, static_cast<int>(biased_exp) - 127);'
            )
            L.append(
                f'    uint32_t src_raw = amdgpu::RegisterAccess(wf).read_lane({src[0]}, lane);'
            )
            L.append('    uint32_t src_byte = inst_.op_sel & 0x3;')
            L.append(
                '    uint8_t byte_val = static_cast<uint8_t>((src_raw >> (src_byte * 8)) & 0xFF);'
            )
            L.append(
                f'    float f0 = static_cast<float>(static_cast<double>({cvt_fn}(static_cast<uint8_t>(byte_val & 0xF))) * scale);'
            )
            L.append(
                f'    float f1 = static_cast<float>(static_cast<double>({cvt_fn}(static_cast<uint8_t>((byte_val >> 4) & 0xF))) * scale);'
            )
        else:
            # FP8/BF8: OPSEL[0] selects half
            L.append(f'    if (biased_exp == 0xFFu) {{')
            if dst_fmt == 'f32':
                L.append(
                    f'      amdgpu::RegisterAccess(wf).write_lane64({dst[0]}, lane, 0x7FC000007FC00000ULL); continue;'
                )
            else:
                L.append(
                    f'      amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, {nan_val} | ({nan_val} << 16)); continue;'
                )
            L.append('    }')
            L.append(
                '    double scale = std::ldexp(1.0, static_cast<int>(biased_exp) - 127);'
            )
            L.append(
                f'    uint32_t src_raw = amdgpu::RegisterAccess(wf).read_lane({src[0]}, lane);'
            )
            L.append('    bool src_hi = inst_.op_sel & 1;')
            L.append(
                '    uint32_t half = src_hi ? (src_raw >> 16) : (src_raw & 0xFFFFu);'
            )
            L.append(
                f'    float f0 = static_cast<float>(static_cast<double>({cvt_fn}(static_cast<uint8_t>(half & 0xFF))) * scale);'
            )
            L.append(
                f'    float f1 = static_cast<float>(static_cast<double>({cvt_fn}(static_cast<uint8_t>((half >> 8) & 0xFF))) * scale);'
            )

        if dst_fmt == 'f32':
            L.append(
                '    uint64_t result = static_cast<uint64_t>(std::bit_cast<uint32_t>(f0))'
            )
            L.append(
                '        | (static_cast<uint64_t>(std::bit_cast<uint32_t>(f1)) << 32);'
            )
            L.append(
                f'    amdgpu::RegisterAccess(wf).write_lane64({dst[0]}, lane, result);'
            )
        elif dst_fmt == 'f16':
            L.append(
                '    uint32_t result = static_cast<uint32_t>(util::f32_to_f16_mode(f0, wf.fp16_ovfl()))'
            )
            L.append(
                '        | (static_cast<uint32_t>(util::f32_to_f16_mode(f1, wf.fp16_ovfl())) << 16);'
            )
            L.append(
                f'    amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, result);'
            )
        else:
            L.append(
                '    uint32_t result = static_cast<uint32_t>(util::f32_to_bf16_rne_mode(f0, wf.fp16_ovfl()))'
            )
            L.append(
                '        | (static_cast<uint32_t>(util::f32_to_bf16_rne_mode(f1, wf.fp16_ovfl())) << 16);'
            )
            L.append(
                f'    amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, result);'
            )

    L.append('  }')
    return '\n'.join(L)


def _gen_wide_scalef32(
    ctx, stochastic: bool, mode: str, dst_fmt: str, src_fmt: str, direction: str
) -> str:
    """Wide (pk32/2xpk16) FP6/BF6 instructions — 32 elements across multi-VGPRs."""
    dst = ctx.dst_ops
    src = ctx.src_ops

    L = []
    L.append('  auto &cu = wf.cu();')
    L.append('  uint32_t vb = wf.vgpr_alloc().base;')
    direct_ops = [dst[0], src[0]]
    if direction == 'narrow' and mode == '2xpk16':
        direct_ops.append(src[1])
    for direct_op in dict.fromkeys(direct_ops):
        L.append(
            f'  auto {direct_op}_off = Isa::resolved_vgpr_offset('
            f'{direct_op}.opr_type_, {direct_op}.encoding_value_);'
        )
        L.append(f'  if (!{direct_op}_off)')
        L.append('    throw util::UnimplementedInst(mnemonic());')
        L.append(
            f'  uint32_t {direct_op}_base = vb + amdgpu::apply_gpr_idx('
            f'wf, *{direct_op}_off, {direct_op}.vgpr_msb_role());'
        )
    L.append('  auto rd = [&](uint32_t base, uint32_t lane, uint32_t dw) -> uint32_t {')
    L.append('    return amdgpu::RegisterAccess(cu).read_vgpr(base + dw, lane);')
    L.append('  };')
    L.append(
        '  auto wr = [&](uint32_t base, uint32_t lane, uint32_t dw, uint32_t val) {'
    )
    L.append('    amdgpu::RegisterAccess(cu).write_vgpr(base + dw, lane, val);')
    L.append('  };')

    L.append('  uint64_t exec = wf.exec();')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')

    scale_src = src[-1]
    if stochastic and len(src) >= 3:
        scale_src = src[2]
        random_src = src[1]
    elif len(src) >= 2:
        scale_src = src[-1]

    _scale_preamble(L, scale_src)
    L.append('    if (biased_exp == 0xFFu) continue;')
    L.append('    double scale = std::ldexp(1.0, static_cast<int>(biased_exp) - 127);')

    if direction == 'narrow':
        cvt_fn = (
            _f32_to_narrow_sr_mode_name(ctx.arch_name, dst_fmt)
            if stochastic
            else _f32_to_narrow_rne_mode_name(ctx.arch_name, dst_fmt)
        )
        n_elems = 32
        bits_per_elem = 6

        if stochastic:
            L.append(
                f'    uint32_t seed = amdgpu::RegisterAccess(wf).read_lane({random_src}, lane);'
            )

        L.append(f'    uint8_t vals[{n_elems}];')

        if mode == '2xpk16':
            # src0 = 16×F32 (lo), src1 = 16×F32 (hi), interleaved in output:
            # field[2k] = lo[k], field[2k+1] = hi[k]
            for half in range(2):
                s = src[half]
                for j in range(16):
                    idx = j * 2 + half
                    L.append(f'    {{')
                    L.append(
                        f'      float v = std::bit_cast<float>(rd({s}_base, lane, {j}));'
                    )
                    L.append(
                        f'      float sv = static_cast<float>(static_cast<double>(v) / scale);'
                    )
                    if stochastic:
                        L.append(
                            f"      vals[{idx}] = {_narrow_sr_encode_call(dst_fmt, cvt_fn, 'sv', 'seed')};"
                        )
                        L.append(f'      seed = util::prng_advance(seed);')
                    else:
                        L.append(
                            f"      vals[{idx}] = {_narrow_rne_encode_call(dst_fmt, cvt_fn, 'sv')};"
                        )
                    L.append(f'    }}')
        elif src_fmt == 'f32':
            # sr_pk32: src0 = 32×F32 (32 DWORDs)
            for j in range(32):
                L.append(f'    {{')
                L.append(
                    f'      float v = std::bit_cast<float>(rd({src[0]}_base, lane, {j}));'
                )
                L.append(
                    f'      float sv = static_cast<float>(static_cast<double>(v) / scale);'
                )
                if stochastic:
                    L.append(
                        f"      vals[{j}] = {_narrow_sr_encode_call(dst_fmt, cvt_fn, 'sv', 'seed')};"
                    )
                    L.append(f'      seed = util::prng_advance(seed);')
                else:
                    L.append(
                        f"      vals[{j}] = {_narrow_rne_encode_call(dst_fmt, cvt_fn, 'sv')};"
                    )
                L.append(f'    }}')
        else:
            # pk32 from F16/BF16: src0 = 32×F16/BF16 (16 DWORDs, 2 per DWORD)
            to_f32 = 'util::f16_to_f32' if src_fmt == 'f16' else 'util::bf16_to_f32'
            for j in range(16):
                L.append(f'    {{')
                L.append(f'      uint32_t dw = rd({src[0]}_base, lane, {j});')
                L.append(
                    f'      float v0 = {to_f32}(static_cast<uint16_t>(dw & 0xFFFF));'
                )
                L.append(
                    f'      float v1 = {to_f32}(static_cast<uint16_t>((dw >> 16) & 0xFFFF));'
                )
                L.append(
                    f'      float sv0 = static_cast<float>(static_cast<double>(v0) / scale);'
                )
                L.append(
                    f'      float sv1 = static_cast<float>(static_cast<double>(v1) / scale);'
                )
                if stochastic:
                    L.append(
                        f"      vals[{j*2}] = {_narrow_sr_encode_call(dst_fmt, cvt_fn, 'sv0', 'seed')};"
                    )
                    L.append(f'      seed = util::prng_advance(seed);')
                    L.append(
                        f"      vals[{j*2+1}] = {_narrow_sr_encode_call(dst_fmt, cvt_fn, 'sv1', 'seed')};"
                    )
                    L.append(f'      seed = util::prng_advance(seed);')
                else:
                    L.append(
                        f"      vals[{j*2}] = {_narrow_rne_encode_call(dst_fmt, cvt_fn, 'sv0')};"
                    )
                    L.append(
                        f"      vals[{j*2+1}] = {_narrow_rne_encode_call(dst_fmt, cvt_fn, 'sv1')};"
                    )
                L.append(f'    }}')

        L.append('    uint32_t dwords[6];')
        L.append('    util::pack_6bit(vals, dwords);')
        for d in range(6):
            L.append(f'    wr({dst[0]}_base, lane, {d}, dwords[{d}]);')

    else:
        # Widening: FP6/BF6 → F32/F16/BF16
        cvt_fn = _narrow_to_f32_name(ctx.arch_name, src_fmt)
        n_elems = 32

        L.append('    uint32_t src_dwords[6];')
        for d in range(6):
            L.append(f'    src_dwords[{d}] = rd({src[0]}_base, lane, {d});')
        L.append('    uint8_t vals[32];')
        L.append('    util::unpack_6bit(src_dwords, vals);')

        if dst_fmt == 'f32':
            for j in range(32):
                L.append(
                    f'    wr({dst[0]}_base, lane, {j}, std::bit_cast<uint32_t>(static_cast<float>(static_cast<double>({cvt_fn}(vals[{j}])) * scale)));'
                )
        elif dst_fmt == 'f16':
            to_narrow = 'util::f32_to_f16_mode'
            for j in range(16):
                L.append(f'    {{')
                L.append(
                    f'      float f0 = static_cast<float>(static_cast<double>({cvt_fn}(vals[{j*2}])) * scale);'
                )
                L.append(
                    f'      float f1 = static_cast<float>(static_cast<double>({cvt_fn}(vals[{j*2+1}])) * scale);'
                )
                L.append(
                    f'      wr({dst[0]}_base, lane, {j}, static_cast<uint32_t>({to_narrow}(f0, wf.fp16_ovfl())) | (static_cast<uint32_t>({to_narrow}(f1, wf.fp16_ovfl())) << 16));'
                )
                L.append(f'    }}')
        else:
            to_narrow = 'util::f32_to_bf16_rne_mode'
            for j in range(16):
                L.append(f'    {{')
                L.append(
                    f'      float f0 = static_cast<float>(static_cast<double>({cvt_fn}(vals[{j*2}])) * scale);'
                )
                L.append(
                    f'      float f1 = static_cast<float>(static_cast<double>({cvt_fn}(vals[{j*2+1}])) * scale);'
                )
                L.append(
                    f'      wr({dst[0]}_base, lane, {j}, static_cast<uint32_t>({to_narrow}(f0, wf.fp16_ovfl())) | (static_cast<uint32_t>({to_narrow}(f1, wf.fp16_ovfl())) << 16));'
                )
                L.append(f'    }}')

    L.append('  }')
    return '\n'.join(L)
