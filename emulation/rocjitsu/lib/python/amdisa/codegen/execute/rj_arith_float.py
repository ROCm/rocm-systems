# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""rj-arith floating arithmetic bindings with explicit guest-mode and state policy."""

from .rj_arith_names import wave_api


def gen_float_fma64(arch, mnemonic, dst, src, is_vop3):
    if arch not in ('rdna3', 'rdna4') or mnemonic != 'V_FMA_F64' or not is_vop3:
        return None
    if len(dst) != 1 or len(src) != 3:
        return None
    api = 'rj_arith_v_fma_f64' + ('_gfx12' if arch == 'rdna4' else '')
    return '\n'.join(
        [
            '  const uint64_t modifiers = ((inst_.abs & 1) ? RJ_ARITH_ALU_ABS_A : 0u) |',
            '      ((inst_.abs & 2) ? RJ_ARITH_ALU_ABS_B : 0u) |',
            '      ((inst_.abs & 4) ? RJ_ARITH_ALU_ABS_C : 0u) |',
            '      ((inst_.neg & 1) ? RJ_ARITH_ALU_NEG_A : 0u) |',
            '      ((inst_.neg & 2) ? RJ_ARITH_ALU_NEG_B : 0u) |',
            '      ((inst_.neg & 4) ? RJ_ARITH_ALU_NEG_C : 0u) |',
            '      (uint64_t(inst_.omod) * RJ_ARITH_ALU_OMOD_2) |',
            '      (inst_.clamp ? RJ_ARITH_ALU_CLAMP : 0u);',
            f'  amdgpu::arith::execute_float_ternary64(wf, {wave_api(api, 32)}, {wave_api(api, 64)},',
            '      wf.cu().arithmetic_flags(), modifiers,',
            f'      {dst[0]}, {src[0]}, {src[1]}, {src[2]}, {str(arch == "rdna3").lower()},',
            '      wf.cu().arithmetic_reports_fp_status());',
        ]
    )


FMA_F32 = {
    'V_FMA_F32',
    'V_FMAC_F32',
    'V_FMA_DX9_ZERO_F32',
    'V_FMAC_DX9_ZERO_F32',
    'V_FMAAK_F32',
    'V_FMAMK_F32',
}


MIXED_FMA_NAMES = {'V_FMA_MIX_F32', 'V_FMA_MIXLO_F16', 'V_FMA_MIXHI_F16'}


PACKED_FMA16_NAMES = {'V_PK_FMA_F16', 'V_PK_FMAC_F16'}


FMA_F16 = {'V_FMA_F16', 'V_FMAC_F16', 'V_FMAAK_F16', 'V_FMAMK_F16'}


def gen_float_fma32(arch, mnemonic, dst, src, is_vop3):
    if arch not in ('rdna3', 'rdna4') or mnemonic not in FMA_F32:
        return None
    if arch == 'rdna4' and mnemonic == 'V_FMAC_DX9_ZERO_F32':
        return None
    accumulate = mnemonic in ('V_FMAC_F32', 'V_FMAC_DX9_ZERO_F32')
    if len(dst) != 1 or len(src) != (2 if accumulate else 3):
        return None
    # The FMA API explicitly represents the accumulator/literal input, so
    # staging the destination cannot discard its old value.
    inputs = src + dst if accumulate else src
    dx9 = 'DX9_ZERO' in mnemonic
    lines = []
    modifiers = '0'
    if is_vop3:
        lines = [
            '  const uint64_t modifiers = ((inst_.abs & 1) ? RJ_ARITH_ALU_ABS_A : 0u) |',
            '      ((inst_.abs & 2) ? RJ_ARITH_ALU_ABS_B : 0u) |',
            '      ((inst_.neg & 1) ? RJ_ARITH_ALU_NEG_A : 0u) |',
            '      ((inst_.neg & 2) ? RJ_ARITH_ALU_NEG_B : 0u) |',
        ]
        if not accumulate:
            lines += [
                '      ((inst_.abs & 4) ? RJ_ARITH_ALU_ABS_C : 0u) |',
                '      ((inst_.neg & 4) ? RJ_ARITH_ALU_NEG_C : 0u) |',
            ]
        lines += [
            '      (uint64_t(inst_.omod) * RJ_ARITH_ALU_OMOD_2) |',
            '      (inst_.clamp ? RJ_ARITH_ALU_CLAMP : 0u);',
        ]
        modifiers = 'modifiers'
    reporting = 'false' if dx9 else 'wf.cu().arithmetic_reports_fp_status()'
    lines += [
        '  amdgpu::arith::execute_float_fma32(wf,',
        f'      wf.cu().arithmetic_flags(), {modifiers},',
        f'      {dst[0]}, {", ".join(inputs)},',
        f'      {str(arch == "rdna3").lower()}, {str(dx9).lower()}, {reporting});',
    ]
    return '\n'.join(lines)


def gen_mixed_fma(arch, mnemonic, dst, src, enc_name, opsel_exprs, hi_c_expr):
    if arch not in ('rdna3', 'rdna4') or mnemonic not in MIXED_FMA_NAMES:
        return None
    if enc_name != 'ENC_VOP3P' or len(dst) != 1 or len(src) != 3:
        return None
    low, high = opsel_exprs
    terms = ['(inst_.clamp ? RJ_ARITH_ALU_CLAMP : 0u)']
    narrow = []
    for index, source in enumerate('ABC'):
        selected = f'({high} & {1 << index})' if index < 2 else hi_c_expr
        terms += [
            f'({selected} ? RJ_ARITH_MIX_F16_{source} : 0u)',
            f'(({low} & {1 << index}) ? RJ_ARITH_ALU_HIGH_{source} : 0u)',
            f'((inst_.neg & {1 << index}) ? RJ_ARITH_ALU_NEG_{source} : 0u)',
            f'((inst_.neg_hi & {1 << index}) ? RJ_ARITH_ALU_ABS_{source} : 0u)',
        ]
        narrow += [
            f'({selected} && amdgpu::is_inline_float_src(inst_.src{index}) ? {1 << index}u : 0u)'
        ]
    api = 'rj_arith_' + mnemonic.lower() + ('_gfx12' if arch == 'rdna4' else '')
    byte_mask = (
        '0xF'
        if mnemonic == 'V_FMA_MIX_F32'
        else '0xC' if 'MIXHI' in mnemonic else '0x3'
    )
    return '\n'.join(
        [
            '  const uint64_t modifiers = ' + ' |\n      '.join(terms) + ';',
            '  const unsigned narrow_sources = ' + ' | '.join(narrow) + ';',
            f'  amdgpu::arith::execute_mixed_fma(wf, {wave_api(api, 32)}, {wave_api(api, 64)},',
            '      wf.cu().arithmetic_flags(), modifiers,',
            f'      {dst[0]}, {", ".join(src)}, narrow_sources, {byte_mask}, {str(arch == "rdna3").lower()},',
            '      wf.cu().arithmetic_reports_fp_status());',
        ]
    )


def gen_packed_fma16(arch, mnemonic, dst, src, enc_name, opsel_exprs, hi_c_expr):
    if arch not in ('rdna3', 'rdna4') or mnemonic not in PACKED_FMA16_NAMES:
        return None
    accumulate = mnemonic == 'V_PK_FMAC_F16'
    count = 2 if accumulate else 3
    if len(dst) != 1 or len(src) != count:
        return None
    if not accumulate and enc_name != 'ENC_VOP3P':
        return None
    lines = []
    terms = []
    if accumulate:
        if 'VOP3' in enc_name:
            terms = [
                '((inst_.abs & 1) ? RJ_ARITH_ALU_ABS_A : 0u)',
                '((inst_.abs & 2) ? RJ_ARITH_ALU_ABS_B : 0u)',
                '((inst_.neg & 1) ? RJ_ARITH_ALU_NEG_A : 0u)',
                '((inst_.neg & 2) ? RJ_ARITH_ALU_NEG_B : 0u)',
                '(inst_.clamp ? RJ_ARITH_ALU_CLAMP : 0u)',
            ]
    else:
        low, high = opsel_exprs
        terms = ['(inst_.clamp ? RJ_ARITH_PK_CLAMP : 0u)']
        for index, source in enumerate('ABC'):
            selected = f'({high} & {1 << index})' if index < 2 else hi_c_expr
            terms += [
                f'(({low} & {1 << index}) ? RJ_ARITH_PK_LO_{source}_HIGH : 0u)',
                f'({selected} ? 0u : RJ_ARITH_PK_HI_{source}_LOW)',
                f'((inst_.neg & {1 << index}) ? RJ_ARITH_PK_NEG_LO_{source} : 0u)',
                f'((inst_.neg_hi & {1 << index}) ? RJ_ARITH_PK_NEG_HI_{source} : 0u)',
            ]
    lines.append(
        '  const uint64_t modifiers = '
        + (' |\n      '.join(terms) if terms else '0')
        + ';'
    )
    narrow, replicate = [], []
    for index, operand in enumerate(src):
        # VOP2's B is a VGPR field, never an inline selector.
        if accumulate and 'VOP3' not in enc_name and index == 1:
            continue
        selector = f'inst_.src{index}'
        narrow.append(
            f'(amdgpu::pk16_src_needs_narrowing({selector}, {operand}.size_bits()) ? {1 << index}u : 0u)'
        )
        if accumulate:
            replicate.append(
                f'(amdgpu::dot2_src_needs_half_replication({selector}) ? {1 << index}u : 0u)'
            )
    lines.append('  const unsigned narrow_sources = ' + ' | '.join(narrow) + ';')
    lines.append(
        '  const unsigned replicate_sources = '
        + (' | '.join(replicate) if replicate else '0')
        + ';'
    )
    addend = dst[0] if accumulate else src[2]
    lines += [
        f'  amdgpu::arith::execute_packed_fma16<{str(accumulate).lower()}>(wf,',
        '      wf.cu().arithmetic_flags(), modifiers,',
        f'      {dst[0]}, {src[0]}, {src[1]}, {addend}, {str(arch == "rdna3").lower()},',
        '      narrow_sources, replicate_sources,',
        '      wf.cu().arithmetic_reports_fp_status());',
    ]
    return '\n'.join(lines)


def gen_float_fma16(arch, mnemonic, dst, src, is_vop3):
    if arch not in ('rdna3', 'rdna4') or mnemonic not in FMA_F16:
        return None
    accumulate = mnemonic == 'V_FMAC_F16'
    if len(dst) != 1 or len(src) != (2 if accumulate else 3):
        return None
    # Literal operands already appear in multiplication/addition order. FMAC
    # supplies old D explicitly so output staging cannot discard its addend.
    inputs = src + dst if accumulate else src
    api = 'rj_arith_v_fma_f16' + ('_gfx12' if arch == 'rdna4' else '')
    modifiers = '0'
    lines = []
    if is_vop3:
        terms = [
            '((inst_.abs & 1) ? RJ_ARITH_ALU_ABS_A : 0u)',
            '((inst_.abs & 2) ? RJ_ARITH_ALU_ABS_B : 0u)',
            '((inst_.neg & 1) ? RJ_ARITH_ALU_NEG_A : 0u)',
            '((inst_.neg & 2) ? RJ_ARITH_ALU_NEG_B : 0u)',
            '((opsel & 1) ? RJ_ARITH_ALU_HIGH_A : 0u)',
            '((opsel & 2) ? RJ_ARITH_ALU_HIGH_B : 0u)',
            '((opsel & 8) ? RJ_ARITH_ALU_HIGH_D : 0u)',
            '(uint64_t(inst_.omod) * RJ_ARITH_ALU_OMOD_2)',
            '(inst_.clamp ? RJ_ARITH_ALU_CLAMP : 0u)',
        ]
        if accumulate:
            terms += ['((opsel & 8) ? RJ_ARITH_ALU_HIGH_C : 0u)']
        else:
            terms += [
                '((opsel & 4) ? RJ_ARITH_ALU_HIGH_C : 0u)',
                '((inst_.abs & 4) ? RJ_ARITH_ALU_ABS_C : 0u)',
                '((inst_.neg & 4) ? RJ_ARITH_ALU_NEG_C : 0u)',
            ]
        lines += [
            '  const uint32_t opsel = amdgpu::vop3_opsel(inst_);',
            '  const uint64_t modifiers = ' + ' |\n      '.join(terms) + ';',
        ]
        modifiers = 'modifiers'
    lines += [
        f'  amdgpu::arith::execute_float_ternary16(wf, {wave_api(api, 32)}, {wave_api(api, 64)},',
        f'      wf.cu().arithmetic_flags(), {modifiers},',
        f'      {dst[0]}, {", ".join(inputs)}, {str(arch == "rdna3").lower()},',
        '      wf.cu().arithmetic_reports_fp_status());',
    ]
    return '\n'.join(lines)


FLOAT_NAMES = FMA_F16 | FMA_F32 | MIXED_FMA_NAMES | PACKED_FMA16_NAMES | {'V_FMA_F64'}
